import argparse
import json
import os
import re

import llm
import numpy as np
from llm.models import Conversation, Prompt, Response, ToolCall, ToolResult
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
    inject_anthropic_cache_control,
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
    "claude-opus-4.7",
    "claude-opus-4.8",
    "claude-opus-5",
    "claude-fable-5",
    "claude-fable-5.1",
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
    "gpt-5.6-sol",
    "gpt-5.6-luna",
    "gpt-5.6-terra",
    "gpt-5",
    "gpt-5-mini",
    "gpt-5-nano",
    # grok-4.3 and the kimi-k2.x models are served through the Azure OpenAI
    # plugin (AzureChat, reasoning: true), so they take the same
    # reasoning_effort kwarg and must not receive temperature or the vLLM-only
    # chat_template_kwargs extra_body.
    "grok-4.3",
    "kimi-k2.5",
    "kimi-k2.6",
]

GEMINI_MODELS = [
    "gemini-2.5-pro",
    "gemini-3-pro-preview",
]

# Non-reasoning chat models served through the Azure OpenAI plugin (AzureChat).
# They are flagged ``reasoning: true`` in the yaml but the endpoint does NOT
# accept a ``reasoning_effort`` argument (grok-4 rejects it outright; the gemma
# deployments ignore it). They behave like ordinary chat models: they take
# ``temperature`` and ``seed`` but, being Azure-hosted, reject the vLLM-only
# ``chat_template_kwargs`` extra_body and expose no tool support
# (supports_tools=False, so ``--tool-actions`` is unavailable).
AZURE_CHAT_MODELS = [
    "grok-4",
    "unsloth/gemma-3-4b",
    "unsloth/gemma-3-27b",
    "google/gemma-4-31B",
]

# Models using Mistral native tokenizer — don't support chat_template_kwargs.
# Thinking is handled via --enable-reasoning --reasoning-parser at the server level
# and surfaces in the API response's `reasoning` field.
MISTRAL_NATIVE_MODELS = [
    "Magistral",
    "Mistral-Small-3",
    "Mistral-Large",
]

# --tool-actions: instead of parsing the action from free text, the model must
# emit it via this tool.  The command argument is sent verbatim to the game
# parser, so hallucinated narration cannot leak into the action.
TOOL_NAME = "submit_command"
TOOL_DESCRIPTION = (
    "Submit the single text command to send to the text-adventure game's parser "
    "for this turn, e.g. 'open mailbox', 'go north', 'get lamp'. Call this exactly "
    "once per turn with one short command and no explanation."
)
TOOL_INPUT_SCHEMA = {
    "type": "object",
    "properties": {
        "command": {
            "type": "string",
            "description": (
                "A single short game command, e.g. 'open mailbox'. Must be one "
                "command only, with no explanation."
            ),
        }
    },
    "required": ["command"],
}
# Appended to SYSTEM_PROMPT when --tool-actions is enabled.
TOOL_INSTRUCTION = (
    "\n\nInstead of typing the command directly, submit it by calling the "
    f"`{TOOL_NAME}` tool with the `command` argument."
)


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
        self.include_reasoning_history = kwargs.get("include_reasoning_history", False)

        # --tool-actions: obtain the action from a `submit_command` tool call
        # rather than parsing it from free text.
        self.tool_actions = kwargs.get("tool_actions", False)
        if self.tool_actions:
            if not self.conversation:
                raise ValueError(
                    "--tool-actions requires conversation mode (tool_use/"
                    "tool_result blocks cannot be merged into a single prompt); "
                    "pass --conversation."
                )
            self._tool = llm.Tool(
                name=TOOL_NAME,
                description=TOOL_DESCRIPTION,
                input_schema=TOOL_INPUT_SCHEMA,
            )
            self._tool_system_prompt = SYSTEM_PROMPT + TOOL_INSTRUCTION

        # Detect OpenAI-compatible local servers and use their client directly.
        # The `llm` library doesn't expose message.reasoning_content, which
        # vLLM and SGLang populate when a reasoning parser is configured.
        server_type = os.environ.get("SERVER_TYPE", "vllm")
        self.server_url = os.environ.get("SERVER_URL") or os.environ.get("VLLM_URL")
        self.reasoning_parser = os.environ.get("REASONING_PARSER")
        self.use_vllm_client = (
            self.server_url is not None
            and server_type in ("vllm", "sglang")
            and self.llm
            not in OPENAI_MODELS + CLAUDE_MODELS + GEMINI_MODELS + AZURE_CHAT_MODELS
        )
        self._is_gptoss = "gpt-oss" in self.llm.lower()
        if self.use_vllm_client:
            api_key = os.environ.get("OPENAI_API_KEY", "not-needed")
            self.client = OpenAI(base_url=self.server_url, api_key=api_key)

    @property
    def uid(self):
        uid = (
            f"ReasoningAgent_{self.llm}"
            f"_s{self.seed}"
            f"_c{self.context_limit}"
            f"_conv{self.conversation}"
            f"_actT{self.act_temp}"
            f"_cotT{self.cot_temp}"
            f"_effort{self.reasoning_effort}"
        )
        if self.tool_actions:
            uid += "_tool1"
        if self.include_reasoning_history:
            uid += "_reasoningHistory1"
        return uid

    @property
    def params(self):
        params = {
            "agent_type": "react",
            "llm": self.llm,
            "seed": self.seed,
            "context_limit": self.context_limit,
            "conversation": self.conversation,
            "act_temp": self.act_temp,
            "cot_temp": self.cot_temp,
            "reasoning_effort": self.reasoning_effort,
        }
        if self.tool_actions:
            params["tool_actions"] = True
        if self.include_reasoning_history:
            params["include_reasoning_history"] = True
        return params

    @staticmethod
    def _unpack_history_entry(entry):
        if len(entry) == 2:
            observation, action = entry
            return observation, action, None
        return entry

    def _append_history(self, observation, action, thinking=None):
        if self.include_reasoning_history and thinking:
            self.history.append((observation, action, thinking))
        else:
            self.history.append((observation, action))

    def _history_text(self, action, thinking):
        if self.include_reasoning_history and thinking:
            return f"<think>\n{thinking}\n</think>\n{action}"
        return action

    def _history_assistant_message(self, action, thinking):
        message = {"role": "assistant", "content": action}
        if not (self.include_reasoning_history and thinking):
            return message

        is_mistral = any(model in self.llm for model in MISTRAL_NATIVE_MODELS)
        if self.use_vllm_client and not is_mistral:
            message["reasoning_content"] = thinking
        else:
            message["content"] = self._history_text(action, thinking)
        return message

    @retry(
        retry=retry_if_exception(is_recoverable_error),
        wait=wait_random_exponential(multiplier=1, max=40),
        stop=stop_after_attempt(100),
    )
    def _llm_call_from_conversation(self, conversation, *args, **kwargs):
        extra_body = kwargs.pop("extra_body", None)
        # In tool-actions mode a valid response carries a tool_use block rather
        # than text (a pure tool call streams as a single space), so validity
        # must be checked on tool_calls() instead of text().
        is_valid = kwargs.pop("is_valid", None)
        if is_valid is None:
            is_valid = lambda response: bool(response.text())
        # The llm-anthropic plugin only sets cache_control for attachments or
        # empty-text prompts, so for our pure-text game conversations cache=True
        # is a no-op. Inject cache_control ourselves via the same monkey-patch.
        inject_cache = bool(kwargs.get("cache")) and self.llm in CLAUDE_MODELS
        needs_patch = bool(extra_body) or inject_cache
        if needs_patch:
            # Monkey-patch model.build_kwargs to inject extra_body / cache_control.
            original_build_kwargs = self.model.__class__.build_kwargs

            def patched_build_kwargs(self_model, prompt, stream):
                result = original_build_kwargs(self_model, prompt, stream)
                if extra_body:
                    result["extra_body"] = extra_body
                if inject_cache:
                    inject_anthropic_cache_control(result)
                return result

            self.model.__class__.build_kwargs = patched_build_kwargs

        try:
            for i in range(10):
                response = conversation.prompt(*args, **kwargs)
                response.duration_ms()  # Forces the response to be computed.
                if is_valid(response):
                    return response  # Valid response, otherwise retry.
                # Remove the failed empty response from conversation to prevent accumulation
                if conversation.responses:
                    conversation.responses.pop()
        finally:
            if needs_patch:
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

    def _completion_token_stats(self, usage, thinking):
        if usage is None:
            return 0, self.token_counter(text=thinking) if thinking else 0

        completion_tokens = usage.completion_tokens or 0
        reasoning_tokens = getattr(usage, "reasoning_tokens", None)

        if reasoning_tokens is None:
            details = getattr(usage, "completion_tokens_details", None)
            if isinstance(details, dict):
                reasoning_tokens = details.get("reasoning_tokens")
            elif details is not None:
                reasoning_tokens = getattr(details, "reasoning_tokens", None)

        if reasoning_tokens is None:
            model_extra = getattr(usage, "model_extra", None) or {}
            reasoning_tokens = model_extra.get("reasoning_tokens")

        if reasoning_tokens is None:
            reasoning_tokens = self.token_counter(text=thinking) if thinking else 0

        reasoning_tokens = max(int(reasoning_tokens or 0), 0)
        if completion_tokens:
            reasoning_tokens = min(reasoning_tokens, completion_tokens)
        response_tokens = max(completion_tokens - reasoning_tokens, 0)
        return response_tokens, reasoning_tokens

    @staticmethod
    def _reasoning_from_message(msg):
        reasoning = getattr(msg, "reasoning", None) or getattr(
            msg, "reasoning_content", None
        )
        if reasoning:
            return reasoning
        model_extra = getattr(msg, "model_extra", None) or {}
        return model_extra.get("reasoning") or model_extra.get("reasoning_content")

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
        thinking = self._reasoning_from_message(msg)
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
        nb_response, nb_thinking = self._completion_token_stats(usage, thinking)
        stats = {
            "prompt": format_messages_to_markdown(messages),
            "thinking": thinking,
            "response": response_text,
            "nb_tokens_prompt": usage.prompt_tokens if usage else 0,
            "nb_tokens_thinking": nb_thinking,
            "nb_tokens_response": nb_response,
        }
        stats["nb_tokens"] = (
            stats["nb_tokens_prompt"]
            + stats["nb_tokens_response"]
            + stats["nb_tokens_thinking"]
        )

        return action, thinking, stats

    def _build_llm_kwargs(self):
        """Build the per-model keyword arguments for the `llm` library path.

        Shared by the free-text ``act`` path and the ``--tool-actions`` path so
        thinking/effort/cache/temperature handling stays identical.
        """
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
            llm_kwargs["cache"] = True  # Enable prompt caching to reduce cost
            llm_kwargs.pop("seed")

        if self.llm in GEMINI_MODELS:
            # Google Gemini API rejects `seed` and renames max_tokens.
            # (Azure-served gemma models in AZURE_CHAT_MODELS accept `seed`.)
            llm_kwargs.pop("seed")

        # For open models served via vLLM/SGLang, enable thinking mode explicitly.
        # Magistral/Mistral-native models use a non-Jinja2 tokenizer that doesn't
        # support chat_template_kwargs — they produce <think> tags naturally.
        # Azure-hosted chat models (AZURE_CHAT_MODELS) reject this extra_body.
        if (
            self.llm
            not in OPENAI_MODELS + CLAUDE_MODELS + GEMINI_MODELS + AZURE_CHAT_MODELS
        ):
            if not any(m in self.llm for m in MISTRAL_NATIVE_MODELS):
                llm_kwargs["extra_body"] = {
                    "chat_template_kwargs": {"enable_thinking": True}
                }

        return llm_kwargs

    def build_tool_conversation(self):
        """Rebuild a fresh ``llm`` Conversation from ``self.history`` for the
        ``--tool-actions`` path.

        Each history entry becomes a user turn followed by an assistant turn
        carrying a ``submit_command`` tool_use. When reasoning history is
        enabled, the assistant turn also contains its prior reasoning trace. The first
        retained turn's user content is the plain-text observation (no dangling
        ``tool_result`` whose ``tool_use`` was truncated away); every subsequent
        turn's user content is a ``tool_result`` referencing the previous
        command's ``tool_call_id``.

        llm >= 0.32 reconstructs the next turn's chain from ``responses[-1]``
        only, requiring that response's ``prompt.messages`` to already hold the
        full prior chain (see llm.models._build_full_chain); older llm walked the
        whole ``responses`` list. To stay correct on both, every synthetic
        response carries the accumulated canonical chain up to (and including)
        its own user turn via an explicit ``messages=``. Without this, only the
        last synthetic turn survives, orphaning its ``tool_result`` and raising
        "unexpected tool_use_id found in tool_result blocks".

        Returns ``(conversation, last_tool_call_id)`` where ``last_tool_call_id``
        is ``None`` when there is no history (the current turn is the very first
        user message).
        """
        from llm.parts import Message, TextPart, ToolCallPart, ToolResultPart

        limit = self.context_limit or len(self.history) + 1
        kept = self.history[-limit:]
        truncated = len(self.history) > len(kept)

        responses = []
        chain = []  # Canonical Message chain accumulated so far.
        prev_id = None
        for i, entry in enumerate(kept):
            obs, command, thinking = self._unpack_history_entry(entry)
            tool_call_id = f"toolu_{i:04d}"
            if i == 0:
                text = obs
                if truncated:
                    text = (
                        f"// History has been truncated to the last {limit} "
                        f"steps.\n...\n{obs}"
                    )
                chain.append(Message(role="user", parts=[TextPart(text=text)]))
            else:
                chain.append(
                    Message(
                        role="tool",
                        parts=[
                            ToolResultPart(
                                name=TOOL_NAME, output=obs, tool_call_id=prev_id
                            )
                        ],
                    )
                )
            prompt = Prompt(
                None,
                model=self.model,
                tools=[self._tool],
                messages=list(chain),
            )
            response = Response(prompt=prompt, model=self.model, stream=False)
            response._done = True
            response._chunks = [""]
            response._tool_calls = [
                ToolCall(TOOL_NAME, {"command": command}, tool_call_id=tool_call_id)
            ]
            responses.append(response)
            assistant_parts = []
            if self.include_reasoning_history and thinking:
                assistant_parts.append(TextPart(text=f"<think>\n{thinking}\n</think>"))
            assistant_parts.append(
                ToolCallPart(
                    name=TOOL_NAME,
                    arguments={"command": command},
                    tool_call_id=tool_call_id,
                )
            )
            chain.append(Message(role="assistant", parts=assistant_parts))
            prev_id = tool_call_id

        return Conversation(self.model, responses=responses), prev_id

    @staticmethod
    def _command_from_tool_response(response):
        """Return the first non-empty `submit_command` command, or None."""
        for call in response.tool_calls():
            arguments = call.arguments or {}
            command = (arguments.get("command") or "").strip()
            if command:
                return command
        return None

    def _act_tool_llm(self, obs, reward, done, infos):
        """`--tool-actions` path for API models via the `llm` library."""
        llm_kwargs = self._build_llm_kwargs()
        conversation, last_tool_call_id = self.build_tool_conversation()
        first_turn = last_tool_call_id is None

        # A pure tool call streams as a single space, so validate on either a
        # tool call or non-empty text; retry only truly empty responses.
        def is_valid(response):
            return bool(response.tool_calls()) or bool(response.text().strip())

        if first_turn:
            prompt_kwargs = {"prompt": obs}
        else:
            prompt_kwargs = {
                "prompt": None,
                "tool_results": [
                    ToolResult(TOOL_NAME, output=obs, tool_call_id=last_tool_call_id)
                ],
            }

        response = self._llm_call_from_conversation(
            conversation,
            tools=[self._tool],
            system=self._tool_system_prompt,
            is_valid=is_valid,
            **prompt_kwargs,
            **llm_kwargs,
        )
        command = self._command_from_tool_response(response)

        # No usable command → re-prompt once with an explicit nudge, then fall
        # back to the first line of the free-text reply.
        if command is None:
            nudge = (
                f"Please respond by calling the `{TOOL_NAME}` tool with a single "
                "`command` argument. Do not answer in plain text."
            )
            # Rebuild a clean conversation for the nudge from history alone. The
            # failed attempt is discarded so the chain again ends on the last
            # historical `submit_command` tool_use, which -- unless this is the
            # very first turn -- MUST be answered by a tool_result in the next
            # message, otherwise Anthropic rejects the request with "tool_use
            # ids were found without tool_result blocks". Delivering the nudge
            # (and re-sending the current observation) through that tool_result
            # guarantees no tool_use is ever left dangling, whether the failed
            # attempt was text-only or an empty tool call.
            conversation, last_tool_call_id = self.build_tool_conversation()
            nudged_obs = f"{obs}\n\n{nudge}"
            if last_tool_call_id is None:
                nudge_kwargs = {"prompt": nudged_obs}
            else:
                nudge_kwargs = {
                    "prompt": None,
                    "tool_results": [
                        ToolResult(
                            TOOL_NAME,
                            output=nudged_obs,
                            tool_call_id=last_tool_call_id,
                        )
                    ],
                }
            response = self._llm_call_from_conversation(
                conversation,
                tools=[self._tool],
                system=self._tool_system_prompt,
                is_valid=is_valid,
                **nudge_kwargs,
                **llm_kwargs,
            )
            command = self._command_from_tool_response(response)

        if command is None:
            text = (response.text() or "").strip()
            command = text.splitlines()[0].strip() if text else ""
        if not command:
            command = "(empty)"

        # Extract thinking (Claude exposes it in the response content blocks).
        thinking = None
        if self.llm in CLAUDE_MODELS:
            content = (response.json() or {}).get("content", []) or []
            thinking = (
                "".join(
                    item.get("thinking", "")
                    for item in content
                    if isinstance(item, dict)
                )
                or None
            )

        self._append_history(obs, command, thinking)

        stats = self._tool_stats(response, obs, command, thinking)
        return command, stats

    def _tool_stats(self, response, obs, command, thinking):
        """Compute usage statistics for a tool-actions turn (llm library)."""
        prompt_md = format_messages_to_markdown(
            [{"role": "system", "content": self._tool_system_prompt}]
            + [
                m
                for entry in self.history[:-1]
                for o, c, t in [self._unpack_history_entry(entry)]
                for m in (
                    {"role": "user", "content": o},
                    {
                        "role": "assistant",
                        "content": self._history_text(c, t),
                    },
                )
            ]
            + [{"role": "user", "content": obs}]
        )
        stats = {
            "prompt": prompt_md,
            "thinking": thinking,
            "response": command,
        }
        try:
            usage = response.usage()
        except Exception:
            usage = None
        nb_prompt = usage.input if usage and usage.input else 0
        nb_output = usage.output if usage and usage.output else 0

        if self.llm in CLAUDE_MODELS:
            # Mirror the free-text path: count the full prompt/response with the
            # tokenizer (usage.input is discounted by prompt caching) and derive
            # thinking tokens from the model-reported output count.
            stats["nb_tokens_prompt"] = self.token_counter(text=prompt_md)
            stats["nb_tokens_response"] = self.token_counter(text=command)
            stats["nb_tokens_thinking"] = (
                max(nb_output - stats["nb_tokens_response"], 0) if nb_output else 0
            )
        elif self.llm in GEMINI_MODELS:
            stats["nb_tokens_prompt"] = nb_prompt
            stats["nb_tokens_response"] = nb_output
            stats["nb_tokens_thinking"] = (
                usage.details.get("thoughtsTokenCount", 0) if usage else 0
            )
        elif self.llm in OPENAI_MODELS:
            stats["nb_tokens_prompt"] = nb_prompt
            stats["nb_tokens_response"] = nb_output
            reasoning_tokens = 0
            try:
                reasoning_tokens = response.response_json["usage"][
                    "completion_tokens_details"
                ]["reasoning_tokens"]
            except (KeyError, TypeError):
                reasoning_tokens = 0
            stats["nb_tokens_thinking"] = reasoning_tokens
        else:
            stats["nb_tokens_prompt"] = nb_prompt or self.token_counter(text=prompt_md)
            stats["nb_tokens_response"] = nb_output or self.token_counter(text=command)
            stats["nb_tokens_thinking"] = (
                self.token_counter(text=thinking) if thinking else 0
            )

        stats["nb_tokens"] = (
            stats["nb_tokens_prompt"]
            + stats["nb_tokens_response"]
            + stats["nb_tokens_thinking"]
        )
        return stats

    @retry(
        retry=retry_if_exception(is_recoverable_error),
        wait=wait_random_exponential(multiplier=1, max=40),
        stop=stop_after_attempt(100),
    )
    def _vllm_tool_call(self, messages, tools, **kwargs):
        """Call a vLLM-served model expecting a tool call in the reply."""
        return self.client.chat.completions.create(
            model=self.llm,
            messages=messages,
            tools=tools,
            tool_choice="auto",
            **kwargs,
        )

    def _build_vllm_tool_messages(self, obs):
        """Rebuild OpenAI-style chat messages (with tool_calls/tool results)
        from ``self.history`` for the ``--tool-actions`` vLLM path."""
        limit = self.context_limit or len(self.history) + 1
        kept = self.history[-limit:]
        truncated = len(self.history) > len(kept)

        messages = [{"role": "system", "content": self._tool_system_prompt}]
        prev_id = None
        for i, entry in enumerate(kept):
            obs_i, command_i, thinking_i = self._unpack_history_entry(entry)
            call_id = f"call_{i:04d}"
            if i == 0:
                text = obs_i
                if truncated:
                    text = (
                        f"// History has been truncated to the last {limit} "
                        f"steps.\n...\n{obs_i}"
                    )
                messages.append({"role": "user", "content": text})
            else:
                messages.append(
                    {"role": "tool", "tool_call_id": prev_id, "content": obs_i}
                )
            assistant_message = {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": call_id,
                        "type": "function",
                        "function": {
                            "name": TOOL_NAME,
                            "arguments": json.dumps({"command": command_i}),
                        },
                    }
                ],
            }
            if self.include_reasoning_history and thinking_i:
                assistant_message["reasoning_content"] = thinking_i
            messages.append(assistant_message)
            prev_id = call_id

        if prev_id is None:
            messages.append({"role": "user", "content": obs})
        else:
            messages.append({"role": "tool", "tool_call_id": prev_id, "content": obs})

        return messages

    def _act_tool_vllm(self, obs, reward, done, infos):
        """`--tool-actions` path for OpenAI-compatible local servers."""
        tools = [
            {
                "type": "function",
                "function": {
                    "name": TOOL_NAME,
                    "description": TOOL_DESCRIPTION,
                    "parameters": TOOL_INPUT_SCHEMA,
                },
            }
        ]

        kwargs = {"temperature": self.cot_temp, "seed": self.seed}
        extra_body = {}
        if self._is_gptoss:
            extra_body["chat_template_kwargs"] = {"reasoning_effort": "high"}
        else:
            is_mistral = any(m in self.llm for m in MISTRAL_NATIVE_MODELS)
            uses_prompt_thinking = "nemotron" in self.llm.lower()
            if not is_mistral and not uses_prompt_thinking:
                extra_body.setdefault("chat_template_kwargs", {})[
                    "enable_thinking"
                ] = True
            if isinstance(self.reasoning_effort, int) and self.reasoning_parser:
                extra_body["thinking_token_budget"] = self.reasoning_effort
        if "gemma-4" in self.llm.lower() or "gemma4" in self.llm.lower():
            extra_body["skip_special_tokens"] = False
        if isinstance(self.reasoning_effort, int):
            kwargs["max_tokens"] = self.reasoning_effort + 512
        else:
            kwargs["max_tokens"] = 2048
        if extra_body:
            kwargs["extra_body"] = extra_body

        messages = self._build_vllm_tool_messages(obs)
        try:
            response = self._vllm_tool_call(messages, tools, **kwargs)
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(
                "--tool-actions failed against the local model server. The served model "
                "or chat template may not support tool/function calling. Original "
                f"error: {exc}"
            ) from exc

        msg = response.choices[0].message
        command = self._command_from_vllm_message(msg)

        # No tool call → re-prompt once with a nudge, then fall back to content.
        if command is None:
            messages.append({"role": "assistant", "content": (msg.content or "")})
            messages.append(
                {
                    "role": "user",
                    "content": (
                        f"Please respond by calling the `{TOOL_NAME}` tool with a "
                        "single `command` argument. Do not answer in plain text."
                    ),
                }
            )
            response = self._vllm_tool_call(messages, tools, **kwargs)
            msg = response.choices[0].message
            command = self._command_from_vllm_message(msg)

        if command is None:
            text = (msg.content or "").strip()
            command = text.splitlines()[0].strip() if text else ""
        if not command:
            command = "(empty)"

        thinking = self._reasoning_from_message(msg)

        self._append_history(obs, command, thinking)

        usage = response.usage
        nb_response, nb_thinking = self._completion_token_stats(usage, thinking)
        stats = {
            "prompt": format_messages_to_markdown(
                [
                    {
                        "role": m["role"],
                        "content": m["content"] if m["content"] is not None else "",
                    }
                    for m in messages
                ]
            ),
            "thinking": thinking,
            "response": command,
            "nb_tokens_prompt": usage.prompt_tokens if usage else 0,
            "nb_tokens_thinking": nb_thinking,
            "nb_tokens_response": nb_response,
        }
        stats["nb_tokens"] = (
            stats["nb_tokens_prompt"]
            + stats["nb_tokens_response"]
            + stats["nb_tokens_thinking"]
        )
        return command, stats

    @staticmethod
    def _command_from_vllm_message(msg):
        """Return the first non-empty command from OpenAI-style tool calls, or None."""
        tool_calls = getattr(msg, "tool_calls", None)
        if not tool_calls:
            return None
        for call in tool_calls:
            try:
                arguments = json.loads(call.function.arguments or "{}")
            except (json.JSONDecodeError, TypeError, AttributeError):
                continue
            command = (arguments.get("command") or "").strip()
            if command:
                return command
        return None

    def act(self, obs, reward, done, infos):
        # --- Tool-actions mode: obtain the action from a structured tool call.
        if self.tool_actions:
            if self.use_vllm_client:
                return self._act_tool_vllm(obs, reward, done, infos)
            return self._act_tool_llm(obs, reward, done, infos)

        # --- vLLM path: use OpenAI client to access reasoning field ---
        if self.use_vllm_client:
            messages = self.build_messages(f"{obs}\n> ")
            action, thinking, stats = self._act_vllm(messages)

            # History management
            if (
                not self.include_reasoning_history
                and any(m in self.llm for m in MISTRAL_NATIVE_MODELS)
                and thinking
            ):
                stub = thinking[:120].rsplit(" ", 1)[0] + "..."
                history_action = f"<think>\n{stub}\n</think>\n{action}\n"
            else:
                history_action = f"{action}\n"
            self._append_history(f"{obs}\n> ", history_action, thinking)

            return action, stats

        # --- Existing llm library path (API models: OpenAI, Claude, Gemini) ---
        llm_kwargs = self._build_llm_kwargs()

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
        if (
            not self.include_reasoning_history
            and any(m in self.llm for m in MISTRAL_NATIVE_MODELS)
            and thinking
        ):
            stub = thinking[:120].rsplit(" ", 1)[0] + "..."
            history_action = f"<think>\n{stub}\n</think>\n{action}\n"
        else:
            history_action = f"{action}\n"
        self._append_history(f"{obs}\n> ", history_action, thinking)

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

        elif self.llm in OPENAI_MODELS + AZURE_CHAT_MODELS:
            # stats["nb_tokens_prompt"] = self.token_counter(messages=messages),
            # stats["nb_tokens_response"] = self.token_counter(text=response_text)
            stats["nb_tokens_prompt"] = response.usage().input
            stats["nb_tokens_response"] = response.usage().output
            # For these models, we need to look at the API response. Some Azure
            # endpoints (e.g. kimi-k2.x) omit completion_tokens_details, so fall
            # back to 0 rather than raising KeyError.
            try:
                stats["nb_tokens_thinking"] = response.response_json["usage"][
                    "completion_tokens_details"
                ]["reasoning_tokens"]
            except (KeyError, TypeError):
                stats["nb_tokens_thinking"] = 0

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

        for i, entry in enumerate(self.history[-limit:]):
            obs, action, thinking = self._unpack_history_entry(entry)
            if len(self.history) >= limit and i == 0:
                # Add the current observation.
                obs = (
                    f"// History has been truncated to the last {limit} steps.\n...\n> "
                )

            messages.append({"role": "user", "content": obs})
            messages.append(self._history_assistant_message(action, thinking))

        messages.append({"role": "user", "content": observation})

        # Just in case, let's avoid having multiple messages from the same role.
        messages = merge_messages(messages)

        if not self.conversation:
            # Merge all messages into a single message except for the system.
            content = "".join(
                [
                    self._history_text(msg["content"], msg.get("reasoning_content"))
                    for msg in messages[1:]
                ]
            )
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
    group.add_argument(
        "--tool-actions",
        dest="tool_actions",
        default=False,
        action=argparse.BooleanOptionalAction,
        help=(
            "Obtain the action from a `submit_command` tool call instead of "
            "parsing free text. Requires --conversation. Default: %(default)s"
        ),
    )
    group.add_argument(
        "--include-reasoning-history",
        default=False,
        action=argparse.BooleanOptionalAction,
        help=(
            "Include each prior reasoning trace in subsequent conversation "
            "history. Default: %(default)s"
        ),
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
