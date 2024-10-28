import logging
import os
import shutil
import tempfile
from os.path import join as pjoin

import requests
import tiktoken
from llm import Conversation, Response
from tqdm import tqdm
from transformers import AutoTokenizer

log = logging.getLogger("tw-bench")


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
    temp_dir = mkdirs(pjoin(tempfile.gettempdir(), "twbench"))
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
        )

        for chunk in r.iter_content(chunk_size=1024):
            if chunk:  # filter out keep-alive new chunks
                pbar.update(len(chunk))
                temp_file.write(chunk)

    shutil.move(temp_path, path)

    pbar.close()
    return path


class TokenCounter:
    def __init__(self, model: str = "gpt-4o"):
        self.model = model
        try:
            self.tokenize = tiktoken.encoding_for_model(model).encode
        except KeyError:
            try:
                # Try to load from transformers.
                self.tokenize = AutoTokenizer.from_pretrained(model).tokenize
            except OSError:
                msg = (
                    f"Tokenizer not found for model {model},"
                    " make sure you have access to the model"
                    " (e.g., HuggingFace API key is correctly set)."
                )
                raise ValueError(msg)

    def __call__(self, *, messages=None, text=None):
        nb_tokens = 0
        if messages is not None:
            nb_tokens += sum(len(self.tokenize(msg["content"])) for msg in messages)

        if text is not None:
            nb_tokens += len(self.tokenize(text))

        return nb_tokens


def merge_messages(messages):
    """Merge messages from the same role into a single message."""
    messages_out = [dict(messages[0])]
    for message in messages[1:]:
        if message["role"] == messages_out[-1]["role"]:
            messages_out[-1]["content"] += "\n\n" + message["content"]
        else:
            messages_out.append(dict(message))

    return messages_out


def messages2conversation(model, messages):
    messages = merge_messages(messages)
    responses = []

    system = None
    for message in messages:
        if message["role"] == "system":
            system = message["content"]
            continue

        if message["role"] == "user":
            prompt = message["content"]
            continue

        if message["role"] == "assistant":
            response = message["content"]
            responses.append(
                Response.fake(model, prompt=prompt, system=system, response=response)
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
        "openai.APITimeoutError",
        "openai.error.Timeout",
        "openai.error.RateLimitError",
        "openai.error.ServiceUnavailableError",
        "openai.Timeout",
        "openai.APIError",
        "openai.APIConnectionError",
        "openai.RateLimitError",
        # Add more as needed
    ]
    exception_full_name = (
        f"{exception.__class__.__module__}.{exception.__class__.__name__}"
    )
    log.warning(f"Exception_full_name: {exception_full_name}")
    log.warning(f"Exception: {exception}")
    return exception_full_name in recoverable_errors
