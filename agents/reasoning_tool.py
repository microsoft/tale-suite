"""ReasoningAgent with a terminal `game_action` tool, for locally served models (vLLM with --enable-auto-tool-choice).

Mirrors the game_action interface ssc1 used for Astra (2026-10-23 spec): game_action is the only tool, tool_choice
"auto"; each step the user message is f"{obs}\n> ", the assistant answers with one game_action call, the history
gets a tool result "Action recorded." and the next observation comes as the next user message (never as the tool
result). Past reasoning goes back with each assistant turn (--keep-reasoning), as in the plain-text runs.

Same as Astra:
- A reply with no tool call: its text is the action (action_source "text"; Beam's wire-token junk is cut as in
  plain-text mode).
Difference from Astra: a malformed reply (several calls, a call to another tool, unparsable or non-string arguments)
is not retried by default (TOOL_RETRIES = 0; Astra retried). The step takes the first game_action call's action, else
the text. With TOOL_RETRIES > 0, each retry uses a new sampling seed (vLLM is seeded, so an identical retry would
repeat the same output); the 500-step Beam runs used TOOL_RETRIES = 4. TALES' own system prompt is unchanged (our
runs don't block state commands, and Break-In's title Enter is sent by the environment).

The step's Response holds a JSON record: source, content, tool_calls, retries (re-requests made) and the malformed
replies.
Requests are non-streaming, straight to the server's /v1/chat/completions (the `llm` library doesn't pass tools here).

Muse Spark (llama.cpp, --reasoning-format none; 2026-10-24): llama.cpp's Muse handler constrains generation with a
grammar once a tool call starts (tool_choice auto/required). The vLLM Beam runs and Astra were unconstrained, so Muse
requests send tool_choice "none": the template still renders the tool (llama.cpp keeps `tools` for the template), no
grammar is applied, and the raw output comes back in the content. It is split here: to=self = thinking, one
to=game_action message = the call (its <atem:invoke name="game_action"> action parameter), to=user = the text
fallback (as in the final-only plain-text runs), to=commentary dropped. Past reasoning goes back as
reasoning_content, the field llama-server reads.
"""

import json
import logging
import re
import time
import urllib.request

from agents.reasoning import (
    BEAM_MODELS,
    LOCAL_REASONING_MODELS,
    MUSE_RAW_MODELS,
    SYSTEM_PROMPT,
    THINK_CAP_PROMPT,
    THINK_CAP_RETRY_TOKENS,
    ReasoningAgent,
    build_argparser,
    cut_beam_junk,
)
from tales.agent import register
from tales.utils import format_messages_to_markdown, merge_messages

GAME_ACTION_TOOL = {
    "type": "function",
    "function": {
        "name": "game_action",
        "description": (
            "Submit exactly one command to the current text-adventure game. For an Enter, Return, or blank-line"
            " prompt, submit <ENTER>. The action is passed to the benchmark's existing parser, so provide only the"
            " command and no commentary."
        ),
        "parameters": {
            "type": "object",
            "properties": {"action": {"type": "string", "description": "The exact text-adventure command to execute."}},
            "required": ["action"],
            "additionalProperties": False,
        },
    },
}
TOOL_RESULT = "Action recorded."
TOOL_RETRIES = 0  # Extra attempts after a malformed reply (the 500-step Beam runs used 4).
RETRY_SEED_STEP = 1_000_003
MUSE_MESSAGE_ANY = re.compile(
    r"(?:<\|start\|>assistant)?\s*to=([\w.\-]+)<\|message\|>(.*?)(?:<\|eom\|>|<\|eot\|>|<\|end\|>|(?=<\|start\|>)|$)", re.S
)
ATEM_INVOKE = re.compile(r'<atem:invoke name="([^"]*)">(.*?)</atem:invoke>', re.S)
ATEM_PARAM = re.compile(r'<atem:parameter name="([^"]*)">(.*?)</atem:parameter>', re.S)


def parse_muse(raw):
    """(thinking, tool messages [(recipient, body)], to=user text) from raw Muse output."""
    messages = [(r, b) for r, b in MUSE_MESSAGE_ANY.findall(raw)]
    thinking = "\n\n".join(b.strip() for r, b in messages if r == "self") or None
    tools = [(r, b) for r, b in messages if r not in ("self", "user", "commentary")]
    user = "\n\n".join(b.strip() for r, b in messages if r == "user")
    return thinking, tools, user


class ReasoningToolAgent(ReasoningAgent):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.llm not in LOCAL_REASONING_MODELS:
            raise ValueError("reasoning-tool is only wired for LOCAL_REASONING_MODELS (vLLM).")
        self.api_base = getattr(self.model, "api_base", None) or "http://127.0.0.1:8000/v1"
        self.sources = []  # "tool" or "text", one per history entry.
        self.muse = self.llm in MUSE_RAW_MODELS
        self.reasoning_field = "reasoning_content" if self.muse else "reasoning"

    @property
    def uid(self):
        return super().uid + "_tool"

    @property
    def params(self):
        return {**super().params, "agent_type": "react-tool", "tool": "game_action",
                "tool_choice": "none (unconstrained; template still renders the tool)" if self.muse else "auto"}

    def build_messages(self, observation):
        messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        for i, ((obs, action, thinking), source) in enumerate(zip(self.history, self.sources)):
            messages.append({"role": "user", "content": obs})
            if source == "tool":
                call_id = f"call_{i + 1}"
                message = {"role": "assistant", "content": "", "tool_calls": [{
                    "id": call_id, "type": "function",
                    "function": {"name": "game_action", "arguments": json.dumps({"action": action})}}]}
                if self.keep_reasoning and thinking:
                    message[self.reasoning_field] = thinking
                messages.append(message)
                messages.append({"role": "tool", "tool_call_id": call_id, "name": "game_action", "content": TOOL_RESULT})
            else:
                message = {"role": "assistant", "content": f"{action}\n"}
                if self.keep_reasoning and thinking:
                    message[self.reasoning_field] = thinking
                messages.append(message)
        messages.append({"role": "user", "content": observation})
        return merge_messages(messages)

    def _post(self, body):
        request = urllib.request.Request(self.api_base.rstrip("/") + "/chat/completions", json.dumps(body).encode(),
                                         {"Content-Type": "application/json"})
        for attempt in range(20):
            try:
                return json.load(urllib.request.urlopen(request, timeout=7200))
            except OSError as exc:  # Connection problems only; HTTP 4xx/5xx with a body still raise below.
                if attempt == 19:
                    raise
                logging.getLogger("tales").warning(f"request failed ({exc}); retrying")
                time.sleep(30)

    def _parse(self, message):
        """(action or None, problem or None) from one reply."""
        if self.muse:
            _, tools, _ = parse_muse(message.get("content") or "")
            if not tools:
                return None, None
            if len(tools) > 1:
                return None, f"{len(tools)} tool messages"
            recipient, body = tools[0]
            invokes = ATEM_INVOKE.findall(body)
            if recipient != "game_action" or len(invokes) != 1 or invokes[0][0] != "game_action":
                return None, f"call to {recipient!r} ({len(invokes)} invokes)"
            params = dict(ATEM_PARAM.findall(invokes[0][1]))
            if "action" not in params:
                return None, "no action parameter"
            return params["action"], None
        calls = message.get("tool_calls") or []
        if not calls:
            return None, None
        if len(calls) > 1:
            return None, f"{len(calls)} tool calls"
        fn = calls[0].get("function") or {}
        if fn.get("name") != "game_action":
            return None, f"call to {fn.get('name')!r}"
        try:
            args = json.loads(fn.get("arguments") or "")
        except json.JSONDecodeError:
            return None, "unparsable arguments"
        if not isinstance(args, dict) or not isinstance(args.get("action"), str):
            return None, "no string action"
        return args["action"], None

    def _split(self, message):
        """(thinking, text-fallback content, raw content) of one reply."""
        raw = message.get("content") or ""
        if self.muse:
            thinking, _, user = parse_muse(raw)
            return thinking, user, raw
        return message.get("reasoning") or message.get("reasoning_content"), raw, raw

    def act(self, obs, reward, done, infos):
        messages = self.build_messages(f"{obs}\n> ")
        body = {"model": self.llm, "messages": messages, "tools": [GAME_ACTION_TOOL],
                "tool_choice": "none" if self.muse else "auto",
                "temperature": self.cot_temp, "seed": self.seed, "reasoning_effort": self.reasoning_effort}
        if self.think_cap:
            body["max_tokens"] = self.think_cap
        malformed = []
        usage = {"prompt_tokens": 0, "completion_tokens": 0, "reasoning_tokens": 0}
        for attempt in range(TOOL_RETRIES + 1):
            body["seed"] = self.seed + attempt * RETRY_SEED_STEP
            response = self._post(body)
            u = response.get("usage") or {}
            usage["prompt_tokens"] = u.get("prompt_tokens", 0)
            usage["completion_tokens"] += u.get("completion_tokens", 0)
            usage["reasoning_tokens"] += ((u.get("completion_tokens_details") or {}).get("reasoning_tokens") or 0)
            choice = response["choices"][0]
            message = choice["message"]
            action, problem = self._parse(message)
            if problem is None:
                break
            malformed.append({"problem": problem, "content": message.get("content"), "tool_calls": message.get("tool_calls"),
                              "reasoning": message.get("reasoning") or message.get("reasoning_content")})
            logging.getLogger("tales").info(f"Malformed tool reply ({problem}); attempt {attempt + 1}.")
        thinking, content, raw = self._split(message)
        stats_thinking = thinking
        if problem is not None and self.muse:  # Retries exhausted: first game_action message's action, else the text.
            action = None
            for recipient, tool_body in parse_muse(raw)[1]:
                invokes = ATEM_INVOKE.findall(tool_body)
                if recipient == "game_action" and invokes and invokes[0][0] == "game_action":
                    action = dict(ATEM_PARAM.findall(invokes[0][1])).get("action")
                    break
        elif problem is not None:  # Retries exhausted: first game_action call, else the text.
            first = next((c for c in message.get("tool_calls") or []
                          if (c.get("function") or {}).get("name") == "game_action"), None)
            try:
                action = json.loads(first["function"]["arguments"])["action"] if first else None
                action = action if isinstance(action, str) else None
            except (json.JSONDecodeError, KeyError, TypeError):
                action = None

        if action is None and self.think_cap and not content.strip() and choice.get("finish_reason") == "length":
            # The thinking hit --think-cap with no action: as in plain-text mode, show the cut-off thinking and ask once.
            cut_thinking = thinking or ""
            body["messages"] = messages + [
                {"role": "assistant", "content": "(thinking cut off)", self.reasoning_field: cut_thinking},
                {"role": "user", "content": THINK_CAP_PROMPT},
            ]
            body["max_tokens"] = THINK_CAP_RETRY_TOKENS
            body["seed"] = self.seed
            response = self._post(body)
            message = response["choices"][0]["message"]
            action, problem = self._parse(message)
            thinking, content, raw = self._split(message)
            stats_thinking = f"{cut_thinking}\n[cut off at --think-cap {self.think_cap}]\n{thinking or ''}"
            messages = body["messages"]

        if action is not None:
            source = "tool"
            action = action.strip()
        else:
            source = "text"
            action = (cut_beam_junk(content) if self.llm in BEAM_MODELS else content).strip()
        if not action:
            action = "(empty)"

        self.history.append((f"{obs}\n> ", action, thinking))
        self.sources.append(source)

        record = {"source": source, "content": raw if self.muse else content, "tool_calls": message.get("tool_calls"),
                  "retries": attempt, "malformed": malformed}
        display = [dict(m, content=m.get("content") if not m.get("tool_calls") else
                        "[game_action] " + m["tool_calls"][0]["function"]["arguments"]) for m in messages]
        stats = {
            "prompt": format_messages_to_markdown(display),
            "thinking": stats_thinking,
            "response": json.dumps(record),
            "compaction": None,
            "nb_tokens_prompt": usage["prompt_tokens"],
            "nb_tokens_thinking": usage["reasoning_tokens"],
            "nb_tokens_response": max(0, usage["completion_tokens"] - usage["reasoning_tokens"]),
        }
        stats["nb_tokens"] = stats["nb_tokens_prompt"] + stats["nb_tokens_response"] + stats["nb_tokens_thinking"]
        return action, stats


register(
    name="reasoning-tool",
    desc="ReasoningAgent that submits each action through a terminal game_action tool (Astra-style), local vLLM models.",
    klass=ReasoningToolAgent,
    add_arguments=build_argparser,
)
