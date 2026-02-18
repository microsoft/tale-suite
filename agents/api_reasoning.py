import argparse
import os
import time

import numpy as np
import requests
from tenacity import (
    retry,
    retry_if_exception,
    stop_after_attempt,
    wait_random_exponential,
)
from termcolor import colored

import tales
from tales.agent import register
from tales.utils import (
    format_messages_to_markdown,
    merge_messages,
)

SYSTEM_PROMPT = (
    "You are playing a text-based game and your goal is to finish it with the highest score."
    " Upon reading the text observation, provide a *single* short phrase to interact with the game, e.g. `get lamp` (without the backticks)."
    " When stuck, try using the `help` command to see what commands are available."
)


def _is_recoverable(exception):
    """Check if this is a transient HTTP/connection error worth retrying."""
    return isinstance(exception, (requests.exceptions.ConnectionError,
                                  requests.exceptions.Timeout,
                                  requests.exceptions.HTTPError))


class APIReasoningAgent(tales.Agent):

    def __init__(self, *args, **kwargs):
        self.model = kwargs["llm"]
        self.api_url = kwargs["api_url"]
        self.api_key = kwargs.get("key") or os.environ.get("TRITON_API_KEY", "")

        self.seed = kwargs["seed"]
        self.rng = np.random.RandomState(self.seed)

        self.history = []
        self.context_limit = kwargs["context_limit"]
        if self.context_limit is not None:
            assert self.context_limit > 0, "--context-limit must be greater than 0."

        self.act_temp = kwargs["act_temp"]
        self.cot_temp = kwargs["cot_temp"]
        self.reasoning_effort = kwargs["reasoning_effort"]
        self.conversation = kwargs["conversation"]

    @property
    def uid(self):
        return (
            f"APIReasoningAgent_{self.model}"
            f"_s{self.seed}"
            f"_c{self.context_limit}"
            f"_conv{self.conversation}"
            f"_actT{self.act_temp}"
            f"_cotT{self.cot_temp}"
            f"_effort{self.reasoning_effort}"
        )

    @property
    def params(self):
        return {
            "agent_type": "api_reasoning",
            "llm": self.model,
            "api_url": self.api_url,
            "seed": self.seed,
            "context_limit": self.context_limit,
            "conversation": self.conversation,
            "act_temp": self.act_temp,
            "cot_temp": self.cot_temp,
            "reasoning_effort": self.reasoning_effort,
        }

    @retry(
        retry=retry_if_exception(_is_recoverable),
        wait=wait_random_exponential(multiplier=1, max=40),
        stop=stop_after_attempt(100),
    )
    def _api_call(self, messages, **kwargs):
        """Make a chat completion request to the API."""
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }

        payload = {
            "model": self.model,
            "messages": messages,
        }

        # Add optional parameters.
        if "temperature" in kwargs:
            payload["temperature"] = kwargs["temperature"]
        if "max_tokens" in kwargs:
            payload["max_tokens"] = kwargs["max_tokens"]
        if "seed" in kwargs:
            payload["seed"] = kwargs["seed"]

        response = requests.post(
            self.api_url, headers=headers, json=payload, timeout=300
        )
        response.raise_for_status()

        data = response.json()
        return data

    def act(self, obs, reward, done, infos):
        call_kwargs = {
            "temperature": self.cot_temp,
            "seed": self.seed,
        }

        if isinstance(self.reasoning_effort, int):
            call_kwargs["max_tokens"] = self.reasoning_effort

        messages = self.build_messages(f"{obs}\n> ")

        # Try up to 10 times to get a non-empty response.
        response_text = ""
        response_data = None
        for _ in range(10):
            response_data = self._api_call(messages, **call_kwargs)
            response_text = response_data["choices"][0]["message"]["content"] or ""
            if response_text.strip():
                break

        action = response_text.strip()
        if action == "":
            action = "(empty)"

        thinking = None

        # Handle thinking models that wrap reasoning in <think> tags.
        if "<think>" in action and "</think>" in action:
            reasoning_end = action.find("</think>") + len("</think>")
            thinking = action[:reasoning_end].strip()
            action = action[reasoning_end:].strip()
        elif "<think>" in action and "</think>" not in action:
            # Thinking was cut off — send another request for just the action.
            messages.append(
                {
                    "role": "assistant",
                    "content": response_text.strip() + "</think>",
                }
            )
            followup_kwargs = {
                "max_tokens": 100,
                "temperature": self.act_temp,
            }
            followup_data = self._api_call(messages, **followup_kwargs)
            followup_text = followup_data["choices"][0]["message"]["content"] or ""
            full_text = response_text.strip() + "</think>" + followup_text
            reasoning_end = full_text.find("</think>") + len("</think>")
            thinking = full_text[:reasoning_end].strip()
            action = full_text[reasoning_end:].strip()

        self.history.append((f"{obs}\n> ", f"{action}\n"))

        # Compute usage statistics.
        stats = {
            "prompt": format_messages_to_markdown(messages),
            "thinking": thinking,
            "response": response_text,
        }

        usage = (response_data or {}).get("usage", {})
        stats["nb_tokens_prompt"] = usage.get("prompt_tokens", 0)
        stats["nb_tokens_response"] = usage.get("completion_tokens", 0)
        stats["nb_tokens_thinking"] = 0
        # Some APIs report reasoning tokens in completion_tokens_details.
        details = usage.get("completion_tokens_details") or {}
        if "reasoning_tokens" in details:
            stats["nb_tokens_thinking"] = details["reasoning_tokens"]

        stats["nb_tokens"] = (
            stats["nb_tokens_prompt"]
            + stats["nb_tokens_response"]
            + stats["nb_tokens_thinking"]
        )

        return action, stats

    def get_system_prompt(self):
        """Return the system prompt for the agent.

        Override this method in a subclass to customize the prompt.
        """
        return SYSTEM_PROMPT

    def build_messages(self, observation):
        messages = [{"role": "system", "content": self.get_system_prompt()}]
        limit = self.context_limit or len(self.history) + 1

        for i, (obs, action) in enumerate(self.history[-limit:]):
            if len(self.history) >= limit and i == 0:
                obs = (
                    f"// History has been truncated to the last {limit} steps.\n...\n> "
                )

            messages.append({"role": "user", "content": obs})
            messages.append({"role": "assistant", "content": action})

        messages.append({"role": "user", "content": observation})

        # Avoid having multiple messages from the same role.
        messages = merge_messages(messages)

        if not self.conversation:
            # Merge all messages into a single message except for the system.
            content = "".join([msg["content"] for msg in messages[1:]])
            messages = messages[:1] + [{"role": "user", "content": content}]

        return messages


def build_argparser(parser=None):
    parser = parser or argparse.ArgumentParser()
    group = parser.add_argument_group("APIReasoningAgent settings")

    group.add_argument(
        "--llm",
        default="api-llama-4-scout",
        help="Model name to use in the API request. Default: %(default)s",
    )
    group.add_argument(
        "--api-url",
        default="https://tritonai-api.ucsd.edu/v1/chat/completions",
        help="URL of the chat completions endpoint. Default: %(default)s",
    )
    group.add_argument(
        "--seed",
        type=int,
        default=20241001,
        help="Seed for LLM (not all endpoints support this). Default: %(default)s",
    )
    group.add_argument(
        "--act-temp",
        type=float,
        default=0.0,
        help="Temperature for LLM when taking actions. Default: %(default)s",
    )
    group.add_argument(
        "--cot-temp",
        type=float,
        default=0.0,
        help="Temperature for LLM when doing chain-of-thoughts. Default: %(default)s",
    )
    subgroup = group.add_mutually_exclusive_group(required=True)
    subgroup.add_argument(
        "--reasoning-effort",
        default="medium",
        dest="reasoning_effort",
        help="Reasoning effort for reasoning-type LLMs.",
    )
    subgroup.add_argument(
        "--cot-max-tokens",
        type=int,
        default=1024,
        dest="reasoning_effort",
        help="Maximum number of tokens for chain-of-thoughts. Default: %(default)s",
    )
    group.add_argument(
        "--context-limit",
        type=int,
        help="Limit context for LLM (in conversation turns). Default: no limit",
    )
    group.add_argument(
        "--conversation",
        required=True,
        action=argparse.BooleanOptionalAction,
        help="Enable conversation mode. Otherwise, use single prompt.",
    )

    return parser


register(
    name="api_reasoning",
    desc=(
        "This agent uses a remote OpenAI-compatible chat API (e.g. TritonAI) "
        "for reasoning and action selection."
    ),
    klass=APIReasoningAgent,
    add_arguments=build_argparser,
)
