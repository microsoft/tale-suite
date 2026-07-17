import argparse
import os
import re

import llm
import numpy as np
from openai import OpenAI
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

# Suffix appended to the system prompt for models that need explicit
# thinking instructions (e.g. Magistral, which uses a native Mistral
# tokenizer and doesn't auto-inject thinking via chat template).
THINKING_INSTRUCTION = (
    "\n\nYour thinking process must follow the template below:\n"
    "<think>\n"
    "Your thoughts or draft.\n"
    "</think>\n\n"
    "After thinking, provide ONLY the game command."
)

DEEPSEEK_CHAT_TEMPLATE_NO_THINK = "{% if not add_generation_prompt is defined %}{% set add_generation_prompt = false %}{% endif %}{% set ns = namespace(is_first=false, is_tool=false, is_output_first=true, system_prompt='') %}{%- for message in messages %}{%- if message['role'] == 'system' %}{% set ns.system_prompt = message['content'] %}{%- endif %}{%- endfor %}{{bos_token}}{{ns.system_prompt}}{%- for message in messages %}{%- if message['role'] == 'user' %}{%- set ns.is_tool = false -%}{{'<｜User｜>' + message['content']}}{%- endif %}{%- if message['role'] == 'assistant' and message['content'] is none %}{%- set ns.is_tool = false -%}{%- for tool in message['tool_calls']%}{%- if not ns.is_first %}{{'<｜Assistant｜><｜tool▁calls▁begin｜><｜tool▁call▁begin｜>' + tool['type'] + '<｜tool▁sep｜>' + tool['function']['name'] + '\\n' + '```json' + '\\n' + tool['function']['arguments'] + '\\n' + '```' + '<｜tool▁call▁end｜>'}}{%- set ns.is_first = true -%}{%- else %}{{'\\n' + '<｜tool▁call▁begin｜>' + tool['type'] + '<｜tool▁sep｜>' + tool['function']['name'] + '\\n' + '```json' + '\\n' + tool['function']['arguments'] + '\\n' + '```' + '<｜tool▁call▁end｜>'}}{{'<｜tool▁calls▁end｜><｜end▁of▁sentence｜>'}}{%- endif %}{%- endfor %}{%- endif %}{%- if message['role'] == 'assistant' and message['content'] is not none %}{%- if ns.is_tool %}{{'<｜tool▁outputs▁end｜>' + message['content'] + '<｜end▁of▁sentence｜>'}}{%- set ns.is_tool = false -%}{%- else %}{% set content = message['content'] %}{% if '</think>' in content %}{% set content = content.split('</think>')[-1] %}{% endif %}{{'<｜Assistant｜>' + content + '<｜end▁of▁sentence｜>'}}{%- endif %}{%- endif %}{%- if message['role'] == 'tool' %}{%- set ns.is_tool = true -%}{%- if ns.is_output_first %}{{'<｜tool▁outputs▁begin｜><｜tool▁output▁begin｜>' + message['content'] + '<｜tool▁output▁end｜>'}}{%- set ns.is_output_first = false %}{%- else %}{{'\\n<｜tool▁output▁begin｜>' + message['content'] + '<｜tool▁output▁end｜>'}}{%- endif %}{%- endif %}{%- endfor -%}{% if ns.is_tool %}{{'<｜tool▁outputs▁end｜>'}}{% endif %}{% if add_generation_prompt and not ns.is_tool %}{{'<｜Assistant｜><think>\\n</think>\\n'}}{% endif %}"

CLAUDE_MODELS = [
    "claude-3.7-sonnet",
    "claude-4-sonnet",
    "claude-4-opus",
    "claude-sonnet-4.5",
    "claude-opus-4.5",
    "claude-opus-4.6",
    "claude-sonnet-4.6",
]

# Capture every <think>…</think> block so we can strip ALL of them from
# the final action.  Some chat templates (notably MiniMax-M2.5 with
# ``enable_thinking: True``) emit an empty ``<think></think>`` immediately
# followed by the model's *real* think block then the action — a single
# pass that stops at the first ``</think>`` would leak the second block
# into the action and the game would reject the multi-line input.
_THINK_RE = re.compile(r"<think>(.*?)</think>", re.DOTALL)
_GEMMA_CHANNEL_RE = re.compile(r"<\|channel>thought\n(.*?)<channel\|>", re.DOTALL)

OPENAI_MODELS = [
    "o1",
    "o1-mini",
    "o1-preview",
    "o3-mini",
    "o4-mini",
    "o3",
    "gpt-5.1",
    "gpt-5.2",
    "gpt-5.3",
    "gpt-5.4",
    "gpt-5",
    "gpt-5-mini",
    "gpt-5-nano",
]

GEMINI_MODELS = [
    "gemini-2.5-pro",
    "gemini-3-pro-preview",
]

# Models using Mistral native tokenizer — don't support chat_template_kwargs.
# Thinking is handled via --enable-reasoning --reasoning-parser at the server level
# and surfaces in the API response's `reasoning` field.
MISTRAL_NATIVE_MODELS = [
    "Magistral",
    "Mistral-Small-3",
    "Mistral-Large",
]


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
        reasoning_effort = kwargs["reasoning_effort"]
        # --reasoning-effort may be a numeric string (e.g., "1024"); convert to int.
        if isinstance(reasoning_effort, str) and reasoning_effort.isdigit():
            reasoning_effort = int(reasoning_effort)
        self.reasoning_effort = reasoning_effort
        self.conversation = kwargs["conversation"]

        # Detect vLLM-served models and use OpenAI client directly.
        # The `llm` library doesn't expose message.reasoning_content,
        # which vLLM populates when --reasoning-parser is configured.
        server_type = os.environ.get("SERVER_TYPE", "vllm")
        self.server_url = os.environ.get("SERVER_URL") or os.environ.get("VLLM_URL")
        self.reasoning_parser = os.environ.get("REASONING_PARSER")
        self.use_vllm_client = (
            self.server_url is not None
            and server_type == "vllm"
            and self.llm not in OPENAI_MODELS + CLAUDE_MODELS + GEMINI_MODELS
        )
        self._is_gptoss = "gpt-oss" in self.llm.lower()
        if self.use_vllm_client:
            api_key = os.environ.get("OPENAI_API_KEY", "not-needed")
            self.client = OpenAI(base_url=self.server_url, api_key=api_key)

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
        extra_body = kwargs.pop("extra_body", None)
        if extra_body:
            # Monkey-patch model.build_kwargs to inject extra_body into API call
            original_build_kwargs = self.model.__class__.build_kwargs

            def patched_build_kwargs(self_model, prompt, stream):
                result = original_build_kwargs(self_model, prompt, stream)
                result["extra_body"] = extra_body
                return result

            self.model.__class__.build_kwargs = patched_build_kwargs

        try:
            for i in range(10):
                response = conversation.prompt(*args, **kwargs)
                response.duration_ms()  # Forces the response to be computed.
                if response.text():
                    return response  # Non-empty response, otherwise retry.
                # Remove the failed empty response from conversation to prevent accumulation
                if conversation.responses:
                    conversation.responses.pop()
        finally:
            if extra_body:
                self.model.__class__.build_kwargs = original_build_kwargs

        return response  # Return last response even if empty

    def _llm_call_from_messages(self, messages, *args, **kwargs):
        conversation = messages2conversation(self.model, messages)
        prompt = messages[-1]["content"]
        system = messages[0]["content"] if self.allows_system_prompt else None

        return self._llm_call_from_conversation(
            conversation, prompt=prompt, system=system, *args, **kwargs
        )

    @retry(
        retry=retry_if_exception(is_recoverable_error),
        wait=wait_random_exponential(multiplier=1, max=40),
        stop=stop_after_attempt(100),
    )
    def _vllm_call(self, messages, **kwargs):
        """Call vLLM-served model via OpenAI-compatible client."""
        for _ in range(10):
            response = self.client.chat.completions.create(
                model=self.llm,
                messages=messages,
                **kwargs,
            )
            content = response.choices[0].message.content or ""
            if content.strip():
                return response
        return response

    def _act_vllm(self, messages):
        """Act using the vLLM OpenAI-compatible client (accesses reasoning field)."""
        kwargs = {"temperature": self.cot_temp, "seed": self.seed}
        extra_body = {}

        if self._is_gptoss:
            # gpt-oss: template-level reasoning effort control (Harmony channels)
            extra_body["chat_template_kwargs"] = {"reasoning_effort": "high"}
        else:
            is_mistral = any(m in self.llm for m in MISTRAL_NATIVE_MODELS)
            # Models using prompt-based thinking (no parser) don't need
            # enable_thinking — it can confuse templates that don't support it.
            uses_prompt_thinking = "nemotron" in self.llm.lower()
            if not is_mistral and not uses_prompt_thinking:
                extra_body.setdefault("chat_template_kwargs", {})[
                    "enable_thinking"
                ] = True
            if isinstance(self.reasoning_effort, int) and self.reasoning_parser:
                extra_body["thinking_token_budget"] = self.reasoning_effort

        # Gemma4: vLLM parser bug strips channel tokens; request them back
        is_gemma4 = "gemma-4" in self.llm.lower() or "gemma4" in self.llm.lower()
        if is_gemma4:
            extra_body["skip_special_tokens"] = False

        if isinstance(self.reasoning_effort, int):
            kwargs["max_tokens"] = self.reasoning_effort + 512
        else:
            kwargs["max_tokens"] = 2048

        if extra_body:
            kwargs["extra_body"] = extra_body

        response = self._vllm_call(messages, **kwargs)
        msg = response.choices[0].message

        # Extract thinking from API reasoning field (populated by --reasoning-parser).
        # vLLM versions use either "reasoning" or "reasoning_content".
        thinking = (
            getattr(msg, "reasoning", None)
            or getattr(msg, "reasoning_content", None)
            or None
        )
        action = (msg.content or "").strip()

        # Fallback: parse <think> tags if no reasoning from API
        if thinking is None and ("<think>" in action or "</think>" in action):
            closed_blocks = _THINK_RE.findall(action)
            if closed_blocks:
                non_empty = [b.strip() for b in closed_blocks if b.strip()]
                thinking = "\n\n".join(non_empty) if non_empty else ""
                action = _THINK_RE.sub("", action).strip()
            elif "</think>" in action:
                # Template-injected <think>: only </think> in output
                parts = action.split("</think>", 1)
                thinking = parts[0].strip()
                action = parts[1].strip() if len(parts) > 1 else ""

        # Fallback: gemma4 channel tokens (vLLM parser bug in v0.20.0)
        if thinking is None and "<|channel>" in action:
            m = _GEMMA_CHANNEL_RE.search(action)
            if m:
                thinking = m.group(1).strip()
                action = _GEMMA_CHANNEL_RE.sub("", action).strip()
                # Clean remaining special tokens from action
                for tag in ("<|channel>", "<channel|>", "<|turn>", "<turn|>"):
                    action = action.replace(tag, "")
                action = action.strip()

        # Recovery: if token budget exhausted with no action, follow-up call
        if not action and response.choices[0].finish_reason == "length":
            follow_up = messages.copy()
            if thinking:
                follow_up.append(
                    {"role": "assistant", "content": f"<think>\n{thinking}\n</think>"}
                )
            follow_up.append({"role": "user", "content": "> "})
            recovery_kwargs = {
                "temperature": self.act_temp,
                "seed": self.seed,
                "max_tokens": 100,
            }
            is_mistral = any(m in self.llm for m in MISTRAL_NATIVE_MODELS)
            if not is_mistral and not self._is_gptoss:
                recovery_kwargs["extra_body"] = {
                    "chat_template_kwargs": {"enable_thinking": False}
                }
            recovery = self._vllm_call(follow_up, **recovery_kwargs)
            action = (recovery.choices[0].message.content or "").strip()
            action = _THINK_RE.sub("", action).strip()

        if not action:
            action = "(empty)"

        # Compute usage statistics
        usage = response.usage
        response_text = msg.content or ""
        stats = {
            "prompt": format_messages_to_markdown(messages),
            "thinking": thinking,
            "response": response_text,
            "nb_tokens_prompt": usage.prompt_tokens if usage else 0,
            "nb_tokens_thinking": self.token_counter(text=thinking) if thinking else 0,
            "nb_tokens_response": usage.completion_tokens if usage else 0,
        }
        stats["nb_tokens"] = (
            stats["nb_tokens_prompt"]
            + stats["nb_tokens_response"]
            + stats["nb_tokens_thinking"]
        )

        return action, thinking, stats

    def act(self, obs, reward, done, infos):
        # --- vLLM path: use OpenAI client to access reasoning field ---
        if self.use_vllm_client:
            messages = self.build_messages(f"{obs}\n> ")
            action, thinking, stats = self._act_vllm(messages)

            # History management
            if any(m in self.llm for m in MISTRAL_NATIVE_MODELS) and thinking:
                stub = thinking[:120].rsplit(" ", 1)[0] + "..."
                history_action = f"<think>\n{stub}\n</think>\n{action}\n"
            else:
                history_action = f"{action}\n"
            self.history.append((f"{obs}\n> ", history_action))

            return action, stats

        # --- Existing llm library path (API models: OpenAI, Claude, Gemini) ---
        llm_kwargs = {
            "temperature": self.cot_temp,
            "seed": self.seed,
            "stream": True,  # Should prevent openai.APITimeoutError
        }
        if isinstance(self.reasoning_effort, int):
            if self.llm in CLAUDE_MODELS:
                llm_kwargs["thinking_budget"] = self.reasoning_effort
            else:
                llm_kwargs["max_tokens"] = self.reasoning_effort

        elif self.llm in OPENAI_MODELS:
            llm_kwargs["reasoning_effort"] = self.reasoning_effort

        elif self.llm in CLAUDE_MODELS:
            llm_kwargs["thinking_effort"] = self.reasoning_effort

        if self.llm in OPENAI_MODELS + CLAUDE_MODELS:
            # For these models, we cannot set the temperature.
            llm_kwargs.pop("temperature")

        if self.llm in ["o3-mini"]:
            llm_kwargs.pop("stream")

        if self.llm in CLAUDE_MODELS:
            llm_kwargs["thinking"] = 1
            llm_kwargs.pop("seed")

        if "gemini" in self.llm or "gemma" in self.llm:
            # For these models, we cannot set the seed and max_tokens has a different name.
            llm_kwargs.pop("seed")

        # For open models served via vLLM/SGLang, enable thinking mode explicitly.
        # Magistral/Mistral-native models use a non-Jinja2 tokenizer that doesn't
        # support chat_template_kwargs — they produce <think> tags naturally.
        if self.llm not in OPENAI_MODELS + CLAUDE_MODELS + GEMINI_MODELS:
            if not any(m in self.llm for m in MISTRAL_NATIVE_MODELS):
                llm_kwargs["extra_body"] = {
                    "chat_template_kwargs": {"enable_thinking": True}
                }

        messages = self.build_messages(f"{obs}\n> ")
        response = self._llm_call_from_messages(messages, **llm_kwargs)
        response_text = response.text()
        action = response.text().strip()

        if action == "":
            # If the action is empty, we need to retry.
            action = "(empty)"

        # --- Extract thinking from <think> tags (generic for all open models) ---
        thinking = None
        if "<think>" in action or "</think>" in action:
            # First pass: strip every closed <think>…</think> block.  Some
            # chat templates emit an empty ``<think></think>`` *and* the
            # real one; we must take all of them, not just the first.
            closed_blocks = _THINK_RE.findall(action)
            if closed_blocks:
                non_empty = [b.strip() for b in closed_blocks if b.strip()]
                thinking = "\n\n".join(non_empty) if non_empty else ""
                action = _THINK_RE.sub("", action).strip()
            elif "</think>" in action:
                # Template-injected <think>: only </think> in output
                parts = action.split("</think>", 1)
                thinking = parts[0].strip()
                action = parts[1].strip() if len(parts) > 1 else ""
            # If after stripping there's still an unclosed <think> tag
            # (token budget exhausted mid-thought), fall through to the
            # follow-up call below.
            if "<think>" in action and "</think>" not in action:
                # Thinking exceeded token budget — send follow-up to get action.
                # Capture the unclosed thinking content before recovery.
                _idx = action.index("<think>") + len("<think>")
                _unclosed = action[_idx:].strip()
                if _unclosed:
                    thinking = (
                        (thinking + "\n\n" + _unclosed).strip()
                        if thinking
                        else _unclosed
                    )
                if "DeepSeek-R1" in self.llm:
                    # DeepSeek requires a custom chat template to suppress thinking.
                    messages.append(
                        {
                            "role": "assistant",
                            "content": "<think>\n"
                            + response_text.strip()
                            + "\n</think>",
                        }
                    )
                    llm_kwargs["max_tokens"] = 100
                    llm_kwargs["temperature"] = self.act_temp
                    llm_kwargs["extra_body"] = {
                        "chat_template": DEEPSEEK_CHAT_TEMPLATE_NO_THINK,
                    }
                    response = self._llm_call_from_messages(messages, **llm_kwargs)
                    response_text += "\n" + response.text()
                    action = response.text().strip()
                elif any(m in self.llm for m in MISTRAL_NATIVE_MODELS):
                    # Mistral-native: just ask for action without thinking
                    messages.append(
                        {
                            "role": "assistant",
                            "content": response_text.strip() + "</think>",
                        }
                    )
                    messages.append({"role": "user", "content": "> "})
                    llm_kwargs["max_tokens"] = 100
                    llm_kwargs["temperature"] = self.act_temp
                    llm_kwargs.pop("extra_body", None)
                    response = self._llm_call_from_messages(messages, **llm_kwargs)
                    response_text += "</think>" + response.text()
                    action = response.text().strip()
                else:
                    # Generic: use chat_template_kwargs (Qwen3, MiniMax, etc.)
                    messages.append(
                        {
                            "role": "assistant",
                            "content": response_text.strip() + "</think>",
                        }
                    )
                    messages.append({"role": "user", "content": "> "})
                    llm_kwargs["max_tokens"] = 100
                    llm_kwargs["temperature"] = self.act_temp
                    llm_kwargs["extra_body"] = {
                        "chat_template_kwargs": {"enable_thinking": False}
                    }
                    response = self._llm_call_from_messages(messages, **llm_kwargs)
                    response_text += "</think>" + response.text()
                    action = response.text().strip()
                # Re-strip any think blocks that came back in the follow-up.
                followup_blocks = _THINK_RE.findall(action)
                if followup_blocks:
                    fu_non_empty = [b.strip() for b in followup_blocks if b.strip()]
                    if fu_non_empty:
                        joined = "\n\n".join(fu_non_empty)
                        thinking = (
                            (thinking + "\n\n" + joined).strip() if thinking else joined
                        )
                    action = _THINK_RE.sub("", action).strip()

        elif self.llm in CLAUDE_MODELS:
            # Extract the thinking part from the response JSON.
            thinking = "".join(
                [item.get("thinking", "") for item in response.json()["content"]]
            )

        # For Mistral-native models, keep a short thinking stub in history so
        # the model sees the <think> pattern and continues to reason on later
        # turns.  Full thinking would bloat context (~1 KiB/step × 100 steps).
        if any(m in self.llm for m in MISTRAL_NATIVE_MODELS) and thinking:
            stub = thinking[:120].rsplit(" ", 1)[0] + "..."
            history_action = f"<think>\n{stub}\n</think>\n{action}\n"
        else:
            history_action = f"{action}\n"
        self.history.append((f"{obs}\n> ", history_action))

        # Compute usage statistics
        stats = {
            "prompt": format_messages_to_markdown(messages),
            "thinking": thinking,
            "response": response_text,
        }

        if self.llm in GEMINI_MODELS:
            stats["nb_tokens_prompt"] = response.usage().input
            stats["nb_tokens_thinking"] = response.usage().details.get(
                "thoughtsTokenCount", 0
            )
            stats["nb_tokens_response"] = response.usage().output

        elif self.llm in OPENAI_MODELS:
            # stats["nb_tokens_prompt"] = self.token_counter(messages=messages),
            # stats["nb_tokens_response"] = self.token_counter(text=response_text)
            stats["nb_tokens_prompt"] = response.usage().input
            stats["nb_tokens_response"] = response.usage().output
            # For these models, we need to look at the API response
            # stats["nb_tokens_thinking"] = response.usage().details["completion_tokens_details"]["reasoning_tokens"]
            stats["nb_tokens_thinking"] = response.response_json["usage"][
                "completion_tokens_details"
            ]["reasoning_tokens"]

        elif self.llm in CLAUDE_MODELS:
            stats["nb_tokens_prompt"] = self.token_counter(messages=messages)
            stats["nb_tokens_response"] = self.token_counter(text=response_text)
            stats["nb_tokens_thinking"] = 0
            if thinking:
                stats["nb_tokens_thinking"] = (
                    response.usage().output - self.token_counter(text=response_text)
                )

        else:
            stats["nb_tokens_prompt"] = self.token_counter(messages=messages)
            stats["nb_tokens_thinking"] = (
                self.token_counter(text=thinking) if thinking else 0
            )
            stats["nb_tokens_response"] = self.token_counter(text=response_text)

        stats["nb_tokens"] = (
            stats["nb_tokens_prompt"]
            + stats["nb_tokens_response"]
            + stats["nb_tokens_thinking"]
        )

        return action, stats

    def build_messages(self, observation):
        system_prompt = SYSTEM_PROMPT
        # Models that need explicit thinking instructions in the prompt
        # because their tokenizer/template doesn't inject them automatically.
        needs_think_instruction = (
            any(m in self.llm for m in MISTRAL_NATIVE_MODELS)
            or "nemotron" in self.llm.lower()
        )
        if needs_think_instruction:
            system_prompt += THINKING_INSTRUCTION

        messages = [{"role": "system", "content": system_prompt}]
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
            messages[1]["content"] = f"{system_prompt}\n\n{messages[1]['content']}"
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
