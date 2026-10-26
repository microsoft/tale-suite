#!/usr/bin/env python3
"""Sanity check: verify a vLLM-served model produces non-empty thinking traces.

Runs a single game-like prompt and checks that the reasoning agent's
thinking extraction logic yields non-empty thinking text. This catches
configuration issues (wrong parser, missing enable_thinking, token budget
problems) before committing to a full 5-seed benchmark run.

Usage (inside container after vLLM is ready):
    python scripts/sanity_check.py --model MODEL_NAME [--server-url URL]

Exit codes:
    0 = thinking traces extracted successfully
    1 = no thinking traces (configuration issue)
    2 = server error / model not responding
"""

import argparse
import os
import re
import sys
import time

from openai import OpenAI

_THINK_RE = re.compile(r"<think>(.*?)</think>", re.DOTALL)
_GEMMA_CHANNEL_RE = re.compile(r"<\|channel>thought\n(.*?)<channel\|>", re.DOTALL)

TEST_PROMPT = (
    "You are playing a text-based game and your goal is to finish it with "
    "the highest score. Upon reading the text observation, provide a *single* "
    "short phrase to interact with the game, e.g. `get lamp` (without the backticks)."
)

THINKING_INSTRUCTION = (
    "\n\nYour thinking process must follow the template below:\n"
    "<think>\n"
    "Your thoughts or draft.\n"
    "</think>\n\n"
    "After thinking, provide ONLY the game command."
)

# Models that need explicit thinking instructions in the prompt
NEEDS_THINK_INSTRUCTION = ["nemotron", "magistral", "mistral-small", "mistral-large"]

TEST_OBSERVATION = (
    "You are standing in an open field west of a white house, with a boarded "
    "front door. There is a small mailbox here."
)


def extract_thinking(msg, model_name: str) -> tuple[str, str]:
    """Extract thinking and action from a chat completion message.

    Mirrors the extraction logic in agents/reasoning.py _act_vllm().
    Returns (thinking, action).
    """
    thinking = (
        getattr(msg, "reasoning", None)
        or getattr(msg, "reasoning_content", None)
        or None
    )
    action = (msg.content or "").strip()

    # Fallback: parse <think> tags
    if thinking is None and ("<think>" in action or "</think>" in action):
        closed_blocks = _THINK_RE.findall(action)
        if closed_blocks:
            non_empty = [b.strip() for b in closed_blocks if b.strip()]
            thinking = "\n\n".join(non_empty) if non_empty else ""
            action = _THINK_RE.sub("", action).strip()
        elif "</think>" in action:
            parts = action.split("</think>", 1)
            thinking = parts[0].strip()
            action = parts[1].strip() if len(parts) > 1 else ""

    # Fallback: gemma4 channel tokens
    if thinking is None and "<|channel>" in action:
        m = _GEMMA_CHANNEL_RE.search(action)
        if m:
            thinking = m.group(1).strip()
            action = _GEMMA_CHANNEL_RE.sub("", action).strip()
            for tag in ("<|channel>", "<channel|>", "<|turn>", "<turn|>"):
                action = action.replace(tag, "")
            action = action.strip()

    return thinking or "", action


def run_check(model: str, server_url: str, reasoning_parser: str) -> bool:
    """Run a single inference and verify thinking is produced."""
    client = OpenAI(base_url=server_url, api_key="not-needed")

    # Build system prompt — add thinking instruction for models that need it
    system_prompt = TEST_PROMPT
    if any(m in model.lower() for m in NEEDS_THINK_INSTRUCTION):
        system_prompt += THINKING_INSTRUCTION

    # Build request matching reasoning agent behavior
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": TEST_OBSERVATION},
    ]

    extra_body = {}
    is_gptoss = "gpt-oss" in model.lower()
    is_gemma4 = "gemma-4" in model.lower() or "gemma4" in model.lower()
    needs_instruction = any(m in model.lower() for m in NEEDS_THINK_INSTRUCTION)

    if is_gptoss:
        extra_body["chat_template_kwargs"] = {"reasoning_effort": "high"}
    elif not needs_instruction:
        # Models with parser support: use enable_thinking + budget
        extra_body.setdefault("chat_template_kwargs", {})["enable_thinking"] = True
        if reasoning_parser:
            extra_body["thinking_token_budget"] = 1024

    if is_gemma4:
        extra_body["skip_special_tokens"] = False

    kwargs = {
        "temperature": 0.0,
        "seed": 42,
        "max_tokens": 1536,
    }
    if extra_body:
        kwargs["extra_body"] = extra_body

    print(f"  Model: {model}")
    print(f"  Server: {server_url}")
    print(f"  Parser: {reasoning_parser or '(none)'}")
    print(f"  extra_body: {extra_body}")
    print()

    try:
        t0 = time.time()
        response = client.chat.completions.create(
            model=model, messages=messages, **kwargs
        )
        elapsed = time.time() - t0
    except Exception as e:
        print(f"  ❌ Server error: {e}", file=sys.stderr)
        return False

    msg = response.choices[0].message
    thinking, action = extract_thinking(msg, model)

    # Display results
    print(f"  Response time: {elapsed:.1f}s")
    print(f"  Finish reason: {response.choices[0].finish_reason}")
    print(
        f"  API reasoning field: {repr((getattr(msg, 'reasoning', None) or getattr(msg, 'reasoning_content', None) or '')[:100])}"
    )
    print(f"  Raw content (first 200): {repr((msg.content or '')[:200])}")
    print()
    print(f"  Extracted thinking ({len(thinking)} chars): {repr(thinking[:200])}")
    print(f"  Extracted action: {repr(action[:100])}")
    print()

    if thinking and len(thinking) > 10:
        print(f"  ✅ PASS — thinking traces present ({len(thinking)} chars)")
        return True
    else:
        print(f"  ❌ FAIL — no thinking traces extracted")
        if not action:
            print(f"     (also no action — model may not be responding properly)")
        return False


def wait_for_server(server_url: str, timeout: int = 300) -> bool:
    """Wait for vLLM server to be ready."""
    import urllib.error
    import urllib.request

    health_url = server_url.rstrip("/v1").rstrip("/") + "/health"
    models_url = server_url + "/models"
    start = time.time()

    while time.time() - start < timeout:
        try:
            urllib.request.urlopen(models_url, timeout=5)
            return True
        except (urllib.error.URLError, OSError):
            time.sleep(2)

    return False


def main():
    parser = argparse.ArgumentParser(description="Sanity check thinking traces")
    parser.add_argument("--model", required=True, help="Model name (as served)")
    parser.add_argument(
        "--server-url",
        default=os.environ.get("SERVER_URL", "http://localhost:8000/v1"),
        help="vLLM server URL",
    )
    parser.add_argument(
        "--reasoning-parser",
        default=os.environ.get("REASONING_PARSER", ""),
        help="Reasoning parser configured on server",
    )
    parser.add_argument(
        "--wait", action="store_true", help="Wait for server to be ready"
    )
    parser.add_argument(
        "--timeout", type=int, default=300, help="Server wait timeout (seconds)"
    )
    args = parser.parse_args()

    print("=" * 60)
    print("TALES Thinking Trace Sanity Check")
    print("=" * 60)
    print()

    if args.wait:
        print(f"  Waiting for server at {args.server_url}...")
        if not wait_for_server(args.server_url, args.timeout):
            print(f"  ❌ Server not ready after {args.timeout}s", file=sys.stderr)
            sys.exit(2)
        print(f"  Server ready.")
        print()

    success = run_check(args.model, args.server_url, args.reasoning_parser)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
