import asyncio
import json
import logging
import os
import shutil
import tempfile
from os.path import join as pjoin

import numpy as np
import requests
from llm import AsyncResponse, Conversation, Prompt, Response
from tqdm import tqdm

from tales.logger import log


def mkdirs(dirpath: str) -> str:
    """Create a directory and all its parents.

    If the folder already exists, its path is returned without raising any exceptions.

    Arguments:
        dirpath: Path where a folder need to be created.

    Returns:
        Path to the (created) folder.
    """
    try:
        os.makedirs(dirpath)
    except FileExistsError:
        pass

    return dirpath


def download(url, dst, desc=None, force=False):
    """Download a remote file using HTTP get request.

    Args:
        url (str): URL where to get the file.
        dst (str): Destination folder where to save the file.
        force (bool, optional):
            Download again if it exists]. Defaults to False.

    Returns:
        str: Path to the downloaded file.

    Notes:
        This code is inspired by
        https://github.com/huggingface/transformers/blob/v4.0.0/src/transformers/file_utils.py#L1069
    """
    filename = url.split("/")[-1]
    path = pjoin(mkdirs(dst), filename)

    if os.path.isfile(path) and not force:
        return path

    # Download to a temp folder first to avoid corrupting the cache
    # with incomplete downloads.
    temp_dir = mkdirs(pjoin(tempfile.gettempdir(), "tales"))
    temp_path = pjoin(temp_dir, filename)
    with open(temp_path, "ab") as temp_file:
        headers = {}
        resume_size = temp_file.tell()
        if resume_size:
            headers["Range"] = f"bytes={resume_size}-"
            headers["x-ms-version"] = "2020-04-08"  # Needed for Range support.

        r = requests.get(url, stream=True, headers=headers)
        if r.headers.get("x-ms-error-code") == "InvalidRange" and r.headers[
            "Content-Range"
        ].rsplit("/", 1)[-1] == str(resume_size):
            shutil.move(temp_path, path)
            return path

        r.raise_for_status()  # Bad request.
        content_length = r.headers.get("Content-Length")
        total = resume_size + int(content_length)
        pbar = tqdm(
            unit="B",
            initial=resume_size,
            unit_scale=True,
            total=total,
            desc=desc or "Downloading {}".format(filename),
            leave=False,
        )

        for chunk in r.iter_content(chunk_size=1024):
            if chunk:  # filter out keep-alive new chunks
                pbar.update(len(chunk))
                temp_file.write(chunk)

    shutil.move(temp_path, path)

    pbar.close()
    return path


def merge_messages(messages):
    """Merge messages from the same role into a single message."""
    messages_out = [dict(messages[0])]
    for message in messages[1:]:
        if message["role"] == messages_out[-1]["role"]:
            messages_out[-1]["content"] += "\n\n" + message["content"]
        else:
            messages_out.append(dict(message))

    return messages_out


def inject_anthropic_cache_control(kwargs):
    """Add Anthropic ``cache_control`` breakpoints to a built request.

    The llm-anthropic plugin only marks ``cache_control`` on attachments or
    empty-text prompts, so pure-text conversations never get cached. We add
    two ephemeral breakpoints: one on the (static) system prompt and one on
    the last message's last content block, so the growing conversation prefix
    is cached incrementally across game steps.
    """
    # Cache the system prompt (static across all turns).
    system = kwargs.get("system")
    if isinstance(system, str) and system:
        kwargs["system"] = [
            {
                "type": "text",
                "text": system,
                "cache_control": {"type": "ephemeral"},
            }
        ]
    elif isinstance(system, list) and system and isinstance(system[-1], dict):
        system[-1]["cache_control"] = {"type": "ephemeral"}

    # Cache the conversation prefix: breakpoint on the last message's last block.
    messages = kwargs.get("messages")
    if messages:
        last = messages[-1]
        content = last.get("content")
        if isinstance(content, list) and content and isinstance(content[-1], dict):
            content[-1]["cache_control"] = {"type": "ephemeral"}
        elif isinstance(content, str) and content:
            last["content"] = [
                {
                    "type": "text",
                    "text": content,
                    "cache_control": {"type": "ephemeral"},
                }
            ]


def messages2conversation(model, messages):
    # llm >= 0.32 rebuilds the next turn's message chain from ``responses[-1]``
    # only, expecting that response's ``prompt.messages`` to already hold the
    # full prior chain (see llm.models._build_full_chain). Older llm walked the
    # whole ``responses`` list instead. To stay correct on both, every synthetic
    # response carries the accumulated canonical chain up to (and including) its
    # own user turn via an explicit ``messages=``; ``_messages_now()`` then
    # appends that turn's assistant output. Without this, all but the last turn
    # of history are silently dropped on the new llm core.
    from llm.parts import Message, TextPart

    messages = merge_messages(messages)  # Just in case.
    responses = []
    chain = []  # Canonical Message chain accumulated so far (excludes system).

    system = None
    for message in messages:
        if message["role"] == "system":
            system = message["content"]
            continue

        if message["role"] == "user":
            prompt = message["content"]
            chain.append(Message(role="user", parts=[TextPart(text=prompt)]))
            continue

        if message["role"] == "assistant":
            # Make a fake response object. Its prompt carries the full chain up
            # to this turn's user input (excluding this assistant output, which
            # ``_messages_now()`` derives from ``_chunks``).
            response = Response(
                model=model,
                prompt=Prompt(
                    prompt,
                    system=system,
                    model=model,
                    messages=list(chain),
                ),
                stream=False,
            )
            response._done = True
            response._chunks = [message["content"]]
            responses.append(response)
            chain.append(
                Message(role="assistant", parts=[TextPart(text=message["content"])])
            )

            system = None
            prompt = None

    return Conversation(model, responses=responses)


def format_messages_to_markdown(messages):
    """Concatenate messages into a single markdown string."""
    markdown_content = ""
    for message in messages:
        role = message["role"].capitalize()
        content = message["content"]
        markdown_content += f"#### {role}\n\n```\n{content}\n```\n\n"
    return markdown_content


def is_recoverable_error(exception):
    # List of exceptions thrown by various libraries that can be retried.
    recoverable_errors = [
        "openai.APIStatusError",
        "openai.PermissionDeniedError",
        "openai.APITimeoutError",
        "openai.error.Timeout",
        "openai.error.RateLimitError",
        "openai.error.ServiceUnavailableError",
        "openai.Timeout",
        "openai.APIError",
        "openai.APIConnectionError",
        "openai.RateLimitError",
        "openai.InternalServerError",
        "anthropic.error.RateLimitError",
        "anthropic.InternalServerError",
        "anthropic.APIConnectionError",
        "anthropic.OverloadedError",
        "anthropic.APIStatusError",
        "anthropic._exceptions.OverloadedError",
        "anthropic.APITimeoutError",
        "anthropic._exceptions.APITimeoutError",
        "llm.errors.ModelError",  # Gemini
        # Add more as needed
    ]
    exception_full_name = (
        f"{exception.__class__.__module__}.{exception.__class__.__name__}"
    )
    log.warning(f"Exception_full_name: {exception_full_name}")
    log.warning(f"Exception: {exception}")
    return exception_full_name in recoverable_errors


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.integer, np.floating)):
            return obj.item()
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NumpyEncoder, self).default(obj)
