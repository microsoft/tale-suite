import argparse

import llm
import numpy as np
from tenacity import (
    retry,
    retry_if_exception,
    stop_after_attempt,
    wait_random_exponential,
)
from termcolor import colored

import tales
from tales.agent import register
from tales.token import get_token_counter
from tales.utils import (
    format_messages_to_markdown,
    is_recoverable_error,
    merge_messages,
    messages2conversation,
)

SYSTEM_PROMPT = (
    "You are playing a text-based game and your goal is to finish it with the highest score."
    " Upon reading the text observation, provide a *single* short phrase to interact with the game, e.g. `get lamp` (without the backticks)."
    " When stuck, try using the `help` command to see what commands are available."
)

DEEPSEEK_CHAT_TEMPLATE_NO_THINK = "{% if not add_generation_prompt is defined %}{% set add_generation_prompt = false %}{% endif %}{% set ns = namespace(is_first=false, is_tool=false, is_output_first=true, system_prompt='') %}{%- for message in messages %}{%- if message['role'] == 'system' %}{% set ns.system_prompt = message['content'] %}{%- endif %}{%- endfor %}{{bos_token}}{{ns.system_prompt}}{%- for message in messages %}{%- if message['role'] == 'user' %}{%- set ns.is_tool = false -%}{{'<｜User｜>' + message['content']}}{%- endif %}{%- if message['role'] == 'assistant' and message['content'] is none %}{%- set ns.is_tool = false -%}{%- for tool in message['tool_calls']%}{%- if not ns.is_first %}{{'<｜Assistant｜><｜tool▁calls▁begin｜><｜tool▁call▁begin｜>' + tool['type'] + '<｜tool▁sep｜>' + tool['function']['name'] + '\\n' + '```json' + '\\n' + tool['function']['arguments'] + '\\n' + '```' + '<｜tool▁call▁end｜>'}}{%- set ns.is_first = true -%}{%- else %}{{'\\n' + '<｜tool▁call▁begin｜>' + tool['type'] + '<｜tool▁sep｜>' + tool['function']['name'] + '\\n' + '```json' + '\\n' + tool['function']['arguments'] + '\\n' + '```' + '<｜tool▁call▁end｜>'}}{{'<｜tool▁calls▁end｜><｜end▁of▁sentence｜>'}}{%- endif %}{%- endfor %}{%- endif %}{%- if message['role'] == 'assistant' and message['content'] is not none %}{%- if ns.is_tool %}{{'<｜tool▁outputs▁end｜>' + message['content'] + '<｜end▁of▁sentence｜>'}}{%- set ns.is_tool = false -%}{%- else %}{% set content = message['content'] %}{% if '</think>' in content %}{% set content = content.split('</think>')[-1] %}{% endif %}{{'<｜Assistant｜>' + content + '<｜end▁of▁sentence｜>'}}{%- endif %}{%- endif %}{%- if message['role'] == 'tool' %}{%- set ns.is_tool = true -%}{%- if ns.is_output_first %}{{'<｜tool▁outputs▁begin｜><｜tool▁output▁begin｜>' + message['content'] + '<｜tool▁output▁end｜>'}}{%- set ns.is_output_first = false %}{%- else %}{{'\\n<｜tool▁output▁begin｜>' + message['content'] + '<｜tool▁output▁end｜>'}}{%- endif %}{%- endif %}{%- endfor -%}{% if ns.is_tool %}{{'<｜tool▁outputs▁end｜>'}}{% endif %}{% if add_generation_prompt and not ns.is_tool %}{{'<｜Assistant｜><think>\\n</think>\\n'}}{% endif %}"


class ReasoningAgent(tales.Agent):

    def __init__(self, *args, **kwargs):
        self.llm = kwargs["llm"]
        self.model = llm.get_model(self.llm)
        self.token_counter = get_token_counter(self.model)
        self.allows_system_prompt = self.llm not in [
            "o1",
            "o1-mini",
            "o1-preview",
            "o3-mini",
            "o4-mini",
            "o3",
        ]

        # Provide the API key, if one is needed and has been provided
        self.model.key = llm.get_key(
            kwargs.get("key"), kwargs["llm"], self.model.key_env_var
        ) or llm.get_key(None, self.model.needs_key, self.model.key_env_var)

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
            f"ReasoningAgent_{self.llm}"
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
            "agent_type": "react",
            "llm": self.llm,
            "seed": self.seed,
            "context_limit": self.context_limit,
            "conversation": self.conversation,
            "act_temp": self.act_temp,
            "cot_temp": self.cot_temp,
            "reasoning_effort": self.reasoning_effort,
        }

    @retry(
        retry=retry_if_exception(is_recoverable_error),
        wait=wait_random_exponential(multiplier=1, max=40),
        stop=stop_after_attempt(100),
    )
    def _llm_call_from_conversation(self, conversation, *args, **kwargs):
        for i in range(10):
            response = conversation.prompt(*args, **kwargs)
            response.duration_ms()  # Forces the response to be computed.
            if response.text():
                return response  # Non-empty response, otherwise retry.

        return ""

    def _llm_call_from_messages(self, messages, *args, **kwargs):
        conversation = messages2conversation(self.model, messages)
        prompt = messages[-1]["content"]
        system = messages[0]["content"] if self.allows_system_prompt else None

        return self._llm_call_from_conversation(
            conversation, prompt=prompt, system=system, *args, **kwargs
        )

    def act(self, obs, reward, done, infos):
        llm_kwargs = {
            "temperature": self.cot_temp,
            "seed": self.seed,
            "stream": True,  # Should prevent openai.APITimeoutError
        }
        if isinstance(self.reasoning_effort, int):
            if self.llm in ["claude-3.7-sonnet"]:
                llm_kwargs["thinking_budget"] = self.reasoning_effort
            else:
                llm_kwargs["max_tokens"] = self.reasoning_effort

        elif self.llm in ["o1", "o1-preview", "o3-mini", "o4-mini", "o3", "gpt-5", "gpt-5-mini", "gpt-5-nano"]:
            llm_kwargs["reasoning_effort"] = self.reasoning_effort

        if self.llm in ["o1", "o1-mini", "o1-preview", "o3-mini", "o4-mini", "o3", "claude-3.7-sonnet", "gpt-5", "gpt-5-mini", "gpt-5-nano"]:
            # For these models, we cannot set the temperature.
            llm_kwargs.pop("temperature")

        if self.llm in ["o3-mini"]:
            llm_kwargs.pop("stream")

        if self.llm in ["claude-3.7-sonnet"]:
            llm_kwargs["thinking"] = 1
            llm_kwargs.pop("seed")

        if "gemini" in self.llm or "gemma" in self.llm:
            # For these models, we cannot set the seed and max_tokens has a different name.
            llm_kwargs.pop("seed")

        messages = self.build_messages(f"{obs}\n> ")
        response = self._llm_call_from_messages(messages, **llm_kwargs)
        response_text = response.text()
        action = response.text().strip()

        if action == "":
            # If the action is empty, we need to retry.
            action = "(empty)"

        thinking = None
        if "Qwen3" in self.llm:
            # Strip the reasoning <think> and </think>.
            reasoning_end = action.find("</think>")
            if reasoning_end == -1:
                # Send another request to get the action with the current reasoning.
                messages.append(
                    {
                        "role": "assistant",
                        "content": response_text.strip() + "</think>",
                    }
                )
                llm_kwargs["max_tokens"] = (
                    100  # Text actions should be short phrases but deepseek forces thought process by starting the generation with <think>.
                )
                llm_kwargs["temperature"] = self.act_temp
                llm_kwargs["extra_body"] = {"chat_template_kwargs": {"enable_thinking": False}}
                response = self._llm_call_from_messages(messages, **llm_kwargs)
                response_text += "</think>" + response.text()
                action = response_text.strip()
                reasoning_end = action.find("</think>") + len("</think>")
            else:
                reasoning_end += len("</think>")

            # Extract the reasoning part from the response.
            thinking = action[:reasoning_end].strip()
            # Extract the action part from the response.
            action = action[reasoning_end:].strip()

        if "DeepSeek-R1" in self.llm:
            # Strip the reasoning <think> and </think>.
            reasoning_end = action.find("</think>")
            if reasoning_end == -1:
                # Send another request to get the action with the current reasoning.
                messages.append(
                    {
                        "role": "assistant",
                        "content": "<think>\n" + response_text.strip() + "\n</think>",
                    }
                )
                # prompt = "// Thinking exceeded the length limit. Based on the thoughts so far, provide your chosen action on a single line while respecting the desired format.\n> "
                # messages.append({"role": "user", "content": prompt})
                llm_kwargs["max_tokens"] = (
                    100  # Text actions should be short phrases but deepseek forces thought process by starting the generation with <think>.
                )
                llm_kwargs["temperature"] = self.act_temp
                llm_kwargs["extra_body"] = {
                    "chat_template": DEEPSEEK_CHAT_TEMPLATE_NO_THINK,
                }
                response = self._llm_call_from_messages(messages, **llm_kwargs)
                response_text += "\n" + response.text()
                action = response.text().strip()
                reasoning_end = action.find("</think>")
                if reasoning_end == -1:
                    reasoning_end = (
                        0  # Give up and use the entire response as the action.
                    )
                else:
                    reasoning_end += len("</think>")
            else:
                reasoning_end += len("</think>")

            # Extract the reasoning part from the response.
            thinking = action[:reasoning_end].strip()
            # Extract the action part from the response.
            action = action[reasoning_end:].strip()

        elif self.llm in ["claude-3.7-sonnet"]:
            # Extract the thinking part from the response JSON.
            thinking = "".join(
                [item.get("thinking", "") for item in response.json()["content"]]
            )

        self.history.append((f"{obs}\n> ", f"{action}\n"))

        # Compute usage statistics
        stats = {
            "prompt": format_messages_to_markdown(messages),
            "thinking": thinking,
            "response": response_text,
        }

        if self.llm in ["gemini-2.5-pro-preview-03-25", "gemini-2.5-pro-preview-05-06"]:
            stats["nb_tokens_prompt"] = response.usage().input
            stats["nb_tokens_thinking"] = response.usage().details.get("thoughtsTokenCount", 0)
            stats["nb_tokens_response"] = response.usage().output

        elif self.llm in ["o1", "o1-mini", "o1-preview", "o3-mini", "o4-mini", "o3", "gpt-5", "gpt-5-mini", "gpt-5-nano"]:
            # stats["nb_tokens_prompt"] = self.token_counter(messages=messages),
            # stats["nb_tokens_response"] = self.token_counter(text=response_text)
            stats["nb_tokens_prompt"] = response.usage().input
            stats["nb_tokens_response"] = response.usage().output
            # For these models, we need to look at the API response
            #stats["nb_tokens_thinking"] = response.usage().details["completion_tokens_details"]["reasoning_tokens"]
            stats["nb_tokens_thinking"] = response.response_json["usage"]["completion_tokens_details"]["reasoning_tokens"]

        else:
            stats["nb_tokens_prompt"] = self.token_counter(messages=messages),
            stats["nb_tokens_thinking"] = self.token_counter(text=thinking) if thinking else 0
            stats["nb_tokens_response"] = self.token_counter(text=response_text)


        stats["nb_tokens"] = (
            stats["nb_tokens_prompt"]
            + stats["nb_tokens_response"]
            + stats["nb_tokens_thinking"]
        )

        return action, stats

    def build_messages(self, observation):
        messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        limit = self.context_limit or len(self.history) + 1

        for i, (obs, action) in enumerate(self.history[-limit:]):
            if len(self.history) >= limit and i == 0:
                # Add the current observation.
                obs = (
                    f"// History has been truncated to the last {limit} steps.\n...\n> "
                )

            messages.append({"role": "user", "content": obs})
            messages.append({"role": "assistant", "content": action})

        messages.append({"role": "user", "content": observation})

        # Just in case, let's avoid having multiple messages from the same role.
        messages = merge_messages(messages)

        if not self.conversation:
            # Merge all messages into a single message except for the system.
            content = "".join([msg["content"] for msg in messages[1:]])
            messages = messages[:1] + [{"role": "user", "content": content}]

        if not self.allows_system_prompt:
            # Make sure the system prompt is added to the following message.
            messages[1]["content"] = f"{SYSTEM_PROMPT}\n\n{messages[1]['content']}"
            messages.pop(0)

        return messages


def build_argparser(parser=None):
    parser = parser or argparse.ArgumentParser()
    group = parser.add_argument_group("LLMAgent settings")

    group.add_argument(
        "--llm",
        default="gpt-4o-mini",
        help="LLM to be used for evaluation. Default: %(default)s",
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
        help="Maximum number of token for chain-of-thoughts. Default: %(default)s",
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
    name="reasoning",
    desc=(
        "This agent uses reasoning LLM (o1/o3, deepseek-r1, etc.) to do CoT/thinking followed deciding which action to take."
    ),
    klass=ReasoningAgent,
    add_arguments=build_argparser,
)
