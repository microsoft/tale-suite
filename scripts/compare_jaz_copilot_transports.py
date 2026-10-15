#!/usr/bin/env python3
"""Run a one-call conformance probe across both JAZ Copilot transports."""

from __future__ import annotations

import argparse
import ast
import json
import sys
import time
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from jaz.llm.copilot import _split_conversation as official_split_conversation

from scripts.copilot_evals import storage
from scripts.copilot_evals.bridge import CopilotBridge
from scripts.copilot_evals.config import RunConfig as BridgeConfig
from scripts.jaz_jericho_harness import (
    CopilotJAZLLM,
    OfficialCopilotJAZLLM,
    split_copilot_conversation,
)


SYSTEM_PROMPT = """\
Respond with only executable Python code. Do not use Markdown fences, notebook
prompts, XML, or prose outside Python comments."""
USER_PROMPT = """\
Return a one-line Python program that prints the exact text transport-ok.
Do not execute any other operation."""


def _valid_python(content: str | None) -> bool:
    try:
        ast.parse(content or "")
    except SyntaxError:
        return False
    return True


def _response_payload(content: str | None, stats: dict[str, Any]) -> dict[str, Any]:
    return {
        "content": content,
        "valid_python": _valid_python(content),
        "stats": stats,
    }


def compare_transports(
    *,
    model: str,
    reasoning_effort: str,
    output_dir: Path,
    call_timeout: float,
    sdk_path: str | None,
    cli_path: str,
    copilot_home: str,
) -> dict[str, Any]:
    output_dir = output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": USER_PROMPT},
    ]
    custom_rendering = split_copilot_conversation(messages)
    official_rendering = official_split_conversation(messages)
    if custom_rendering != official_rendering:
        raise RuntimeError("Custom and official Copilot prompt rendering diverged.")

    bridge = CopilotBridge(
        BridgeConfig(
            output_root=output_dir,
            call_timeout=call_timeout,
            reasoning_summary="detailed",
            sdk_path=sdk_path,
            cli_path=cli_path,
            copilot_home=copilot_home,
            bridge_log_level="error",
        ),
        log_path=output_dir / "custom-bridge.log",
        working_directory=output_dir / "custom-work",
    )
    official: OfficialCopilotJAZLLM | None = None
    try:
        raw_model = next(
            (item for item in bridge.list_models() if item.get("id") == model),
            None,
        )
        if raw_model is None:
            raise ValueError(f"Copilot model is unavailable: {model}")

        custom = CopilotJAZLLM(
            bridge=bridge,
            model=model,
            reasoning_effort=reasoning_effort,
            model_metadata=raw_model,
            telemetry_path=output_dir / "custom.jsonl",
            generated_code_dir=output_dir / "custom-code",
            max_retries=0,
        )
        custom_response = custom.complete(model, messages)

        official = OfficialCopilotJAZLLM(
            model=model,
            reasoning_effort=reasoning_effort,
            model_metadata=raw_model,
            telemetry_path=output_dir / "official.jsonl",
            generated_code_dir=output_dir / "official-code",
            base_directory=Path(copilot_home).expanduser().resolve(),
            cli_path=cli_path,
            call_timeout=call_timeout,
            max_retries=0,
        )
        official_response = official.complete(model, messages)
        result = {
            "model": model,
            "reasoning_effort": reasoning_effort,
            "timestamp": time.time(),
            "prompt_rendering_equivalent": True,
            "custom": _response_payload(custom_response.content, custom.stats()),
            "official": _response_payload(
                official_response.content,
                official.stats(),
            ),
        }
        result["both_valid_python"] = bool(
            result["custom"]["valid_python"] and result["official"]["valid_python"]
        )
        storage.atomic_write_json(output_dir / "comparison.json", result, indent=2)
        return result
    finally:
        if official is not None:
            official.close()
        bridge.stop()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compare custom and official JAZ Copilot transports."
    )
    parser.add_argument("--model", default="gpt-6-astra")
    parser.add_argument("--reasoning-effort", default="max")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--call-timeout", type=float, default=900.0)
    parser.add_argument("--sdk-path")
    parser.add_argument("--cli-path", default="/usr/local/bin/copilot")
    parser.add_argument(
        "--copilot-home",
        default=str(Path.home() / ".copilot"),
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    result = compare_transports(
        model=args.model,
        reasoning_effort=args.reasoning_effort,
        output_dir=args.output_dir,
        call_timeout=args.call_timeout,
        sdk_path=args.sdk_path,
        cli_path=args.cli_path,
        copilot_home=args.copilot_home,
    )
    print(json.dumps(result, indent=2))
    return 0 if result["both_valid_python"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
