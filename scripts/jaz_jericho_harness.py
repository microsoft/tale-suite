"""JAZ-style code-REPL harness for TALES/Jericho.

This module adapts the ``invoke`` primitive from *Harness as a Language* to a
single long-horizon Jericho episode. It intentionally lives beside the existing
Copilot evaluation tooling instead of changing ``benchmark.py``: canonical
ReAct trajectories and code-mode JAZ trajectories remain separate experiments.
"""

from __future__ import annotations

import ast
import hashlib
import io
import os
import re
import threading
import time
import tokenize
import traceback
from contextlib import ExitStack
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable

import jaz
from jaz.exceptions import BudgetExhaustedError
from jaz.hooks import (
    BudgetPool,
    ContextWindowWarning,
    RecursionLimit,
    TrajectoryRecorder,
    ValidateReturn,
)
from jaz.llm import BaseLLM, CopilotLLM, LLMResponse
from jaz.protocol import CodeOnlyProtocol
from jaz.repl import PythonREPL

import benchmark
from scripts.copilot_evals import episode, games, schema, storage
from scripts.copilot_evals.bridge import BridgeError, CopilotBridge
from scripts.copilot_evals.config import (
    RunConfig as BridgeConfig,
    default_sdk_path,
)
from scripts.copilot_evals.models import build_model_spec, provider_family, slugify

DEFAULT_MODELS = ("gpt-6-astra", "claude-opus-5.5")
DEFAULT_SEEDS = (202411061, 202411062, 202411063, 202411064, 202411065)
DEFAULT_GAMES = ("JerichoEnvZork1",)
DEFAULT_OUTPUT_ROOT = (
    Path("/root/tale-suite-clean-trajectories")
    / "jaz-jericho-zork1-500calls-rootguard-5seeds"
)
ALLOWED_IMPORTS = ("re", "collections", "ast", "datetime", "textwrap", "pprint")
NANO_AI_CREDITS = 1_000_000_000
CONTENT_FILTER_REPL_CODE = (
    'print("Provider content filtering blocked this model turn. No model code '
    'was executed; continue with a different safe game approach.")'
)
OUTPUT_LIMIT_REPL_CODE = (
    'print("The provider exhausted its output limit before completing usable '
    "model code. No model code was executed; continue with a shorter, concise "
    'next step.")'
)

_ROLE_MARKER_LINE = re.compile(
    r"^\[\[(?:user|assistant|system|developer|tool)\]\][ \t]*$", re.MULTILINE
)
_COMPOUND_COMMAND_RE = re.compile(r"[.]|\b(?:and|then)\b", re.IGNORECASE)
_REPO_ROOT = Path(__file__).resolve().parents[1]
_IMPLEMENTATION_FILES = (
    _REPO_ROOT / "requirements.txt",
    _REPO_ROOT / "benchmark.py",
    _REPO_ROOT / "scripts" / "jaz_jericho_harness.py",
    _REPO_ROOT / "scripts" / "run_jaz_jericho.py",
    _REPO_ROOT / "scripts" / "copilot_model_bridge.mjs",
    _REPO_ROOT / "scripts" / "copilot_evals" / "bridge.py",
    _REPO_ROOT / "scripts" / "copilot_evals" / "config.py",
    _REPO_ROOT / "scripts" / "copilot_evals" / "episode.py",
    _REPO_ROOT / "scripts" / "copilot_evals" / "games.py",
    _REPO_ROOT / "scripts" / "copilot_evals" / "models.py",
    _REPO_ROOT / "scripts" / "copilot_evals" / "schema.py",
    _REPO_ROOT / "scripts" / "copilot_evals" / "storage.py",
    _REPO_ROOT / "tales" / "jericho" / "game_catalog.jsonl",
    _REPO_ROOT / "tales" / "jericho" / "game_catalog.py",
    _REPO_ROOT / "tales" / "jericho" / "jericho_env.py",
)


class GameCompleteError(RuntimeError):
    """The model attempted another action after the run was complete."""


class ActionSafetyLimitExceededError(GameCompleteError):
    """The model attempted another action after reaching the runaway guard."""


class CompletionIntegrityError(BridgeError):
    """A completed Copilot turn violated provider provenance requirements."""


def implementation_sha256() -> str:
    digest = hashlib.sha256()
    for path in _IMPLEMENTATION_FILES:
        digest.update(path.relative_to(_REPO_ROOT).as_posix().encode())
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


@dataclass(frozen=True)
class JAZRunConfig:
    """Configuration for one model/game/seed episode."""

    model: str
    game: str
    seed: int
    output_root: Path = DEFAULT_OUTPUT_ROOT
    reasoning_effort: str = "max"
    max_model_calls: int = 500
    action_safety_limit: int | None = 5_000
    return_guard_scope: str = "root"
    transport: str = "custom"
    max_recursion_depth: int = 8
    context_warn_fraction: float = 0.70
    call_timeout: float = 600.0
    repl_timeout: float = 900.0
    repl_memory_limit_mb: int = 2048
    native_step_timeout: float = 30.0
    max_repl_output_length: int = 50_000
    max_invoke_input_length: int = 50_000
    max_retries: int = 3
    sdk_path: str | None = None
    cli_path: str = "/usr/local/bin/copilot"
    copilot_home: str = str(Path.home() / ".copilot")
    bridge_log_level: str = "error"
    force: bool = False
    implementation_digest: str = field(default_factory=implementation_sha256)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "output_root",
            Path(self.output_root).expanduser().resolve(),
        )
        object.__setattr__(
            self,
            "copilot_home",
            str(Path(self.copilot_home).expanduser().resolve()),
        )
        resolved_sdk_path = self.sdk_path or default_sdk_path()
        if resolved_sdk_path is not None:
            object.__setattr__(
                self,
                "sdk_path",
                str(Path(resolved_sdk_path).expanduser().resolve()),
            )
        for name in (
            "max_model_calls",
            "max_recursion_depth",
            "repl_memory_limit_mb",
            "max_repl_output_length",
            "max_invoke_input_length",
        ):
            if int(getattr(self, name)) <= 0:
                raise ValueError(f"{name} must be positive")
        if self.action_safety_limit is not None and int(self.action_safety_limit) <= 0:
            raise ValueError("action_safety_limit must be positive when provided")
        if self.return_guard_scope not in {"root", "tree"}:
            raise ValueError("return_guard_scope must be 'root' or 'tree'")
        if self.transport not in {"custom", "official"}:
            raise ValueError("transport must be 'custom' or 'official'")
        for name in ("call_timeout", "repl_timeout", "native_step_timeout"):
            if float(getattr(self, name)) <= 0:
                raise ValueError(f"{name} must be positive")
        if not 0.0 < self.context_warn_fraction < 1.0:
            raise ValueError("context_warn_fraction must be between 0 and 1")

    @property
    def run_id(self) -> str:
        return (
            f"{self.game}-{slugify(self.model)}-jaz-"
            f"{self.transport}-{self.return_guard_scope}-"
            f"{self.reasoning_effort}-s{self.seed}"
        )

    @property
    def run_dir(self) -> Path:
        return self.output_root / slugify(self.model) / self.game / f"s{self.seed}"

    @property
    def workspace_dir(self) -> Path:
        return self.run_dir / "workspace"

    @property
    def generated_code_dir(self) -> Path:
        return self.run_dir / "generated-code"

    @property
    def workspace_glob(self) -> str:
        return f"/{self.workspace_dir.as_posix()}/**"

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["output_root"] = str(self.output_root)
        payload["run_id"] = self.run_id
        payload["run_dir"] = str(self.run_dir)
        payload["workspace_dir"] = str(self.workspace_dir)
        payload["generated_code_dir"] = str(self.generated_code_dir)
        payload["allowed_imports"] = list(ALLOWED_IMPORTS)
        payload["allow_timeout_pragma"] = False
        payload["allowed_read_paths"] = [self.workspace_glob]
        payload["allowed_write_paths"] = [self.workspace_glob]
        payload["implementation_sha256"] = payload.pop("implementation_digest")
        return payload


@dataclass(frozen=True)
class GameState:
    """Visible result returned by ``observe()`` and ``step(action)``."""

    observation: str
    feedback: str
    submitted_action: str | None
    executed_action: str | None
    action_was_executed: bool
    invalid_reason: str | None
    blocked_command: str | None
    reward: int
    score: int
    highscore: int
    max_score: int
    moves: int
    episode_done: bool
    episode_won: bool
    episode_lost: bool
    restarted: bool
    actions_used: int
    actions_remaining: int | None
    action_safety_remaining: int | None
    run_complete: bool
    completion_reason: str | None

    def __str__(self) -> str:
        sections = []
        if self.feedback and self.feedback.strip() != self.observation.strip():
            sections.append(self.feedback)
        sections.append(self.observation)
        remaining = "unlimited"
        status = (
            f"[score={self.score}/{self.max_score}; "
            f"highscore={self.highscore}/{self.max_score}; moves={self.moves}; "
            f"actions={self.actions_used}, remaining={remaining}"
        )
        if self.action_safety_remaining is not None:
            status += f"; safety_remaining={self.action_safety_remaining}"
        if self.restarted:
            status += "; restarted=true"
        if self.run_complete:
            status += f"; complete={self.completion_reason}"
        status += "]"
        sections.append(status)
        return "\n\n".join(section for section in sections if section)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _message_text(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(
            str(part.get("text") or "")
            for part in content
            if isinstance(part, dict) and part.get("type") == "text"
        )
    return str(content)


def split_copilot_conversation(
    messages: list[Any],
) -> tuple[str, str, bool]:
    """Render JAZ messages for Copilot's one-prompt session interface."""
    system_parts: list[str] = []
    turns: list[tuple[str, str]] = []
    for message in messages:
        role = str(message.get("role") or "user")
        text = _message_text(message.get("content"))
        if role in {"system", "developer"}:
            system_parts.append(text)
        else:
            turns.append((role, text))

    system_prompt = "\n\n".join(part for part in system_parts if part)
    flattened = not (len(turns) == 1 and turns[0][0] == "user")
    if not flattened:
        prompt = turns[0][1]
    else:
        prompt = "\n\n".join(f"[[{role}]]\n{text}" for role, text in turns)
    return system_prompt, prompt, flattened


def trim_transcript_continuation(content: str) -> tuple[str, int]:
    """Remove a hallucinated continuation of the flattened chat transcript."""
    match = _ROLE_MARKER_LINE.search(content)
    if match is None:
        return content, 0
    marker_line = content.count("\n", 0, match.start()) + 1
    previous_newline = content.rfind("\n", 0, match.start())
    marker_position = (marker_line, match.start() - previous_newline - 1)
    try:
        tokens = tokenize.generate_tokens(io.StringIO(content).readline)
        if any(
            token.start <= marker_position < token.end
            and (
                token.type == tokenize.STRING
                or tokenize.tok_name[token.type].startswith("FSTRING")
            )
            for token in tokens
        ):
            return content, 0
    except (IndentationError, SyntaxError, tokenize.TokenError):
        pass
    trimmed = content[: match.start()].rstrip()
    return trimmed, len(content) - len(trimmed)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _python_symbols(path: Path) -> list[dict[str, Any]]:
    try:
        source = path.read_text(encoding="utf-8")
        tree = ast.parse(source)
    except (OSError, UnicodeDecodeError, SyntaxError):
        return []
    symbols = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            symbols.append(
                {
                    "kind": (
                        "class"
                        if isinstance(node, ast.ClassDef)
                        else (
                            "async_function"
                            if isinstance(node, ast.AsyncFunctionDef)
                            else "function"
                        )
                    ),
                    "name": node.name,
                    "line": node.lineno,
                }
            )
    return sorted(symbols, key=lambda item: (item["line"], item["name"]))


def _directory_manifest(root: Path, *, include_symbols: bool) -> dict[str, Any]:
    files: list[dict[str, Any]] = []
    if root.exists():
        for path in sorted(root.rglob("*")):
            relative = path.relative_to(root).as_posix()
            if path.is_symlink():
                files.append(
                    {
                        "path": relative,
                        "type": "symlink",
                        "target": str(path.readlink()),
                    }
                )
            elif path.is_file():
                entry: dict[str, Any] = {
                    "path": relative,
                    "type": "file",
                    "size_bytes": path.stat().st_size,
                    "sha256": _sha256_file(path),
                }
                if include_symbols and path.suffix == ".py":
                    entry["python_symbols"] = _python_symbols(path)
                files.append(entry)
    return {
        "root": str(root),
        "file_count": sum(entry["type"] == "file" for entry in files),
        "files": files,
    }


def _write_artifact_manifests(
    run_dir: Path,
    workspace_dir: Path,
    generated_code_dir: Path,
) -> tuple[Path, Path]:
    workspace_manifest = run_dir / "workspace-manifest.json"
    generated_code_manifest = run_dir / "generated-code-manifest.json"
    storage.atomic_write_json(
        workspace_manifest,
        _directory_manifest(workspace_dir, include_symbols=True),
        indent=2,
    )
    storage.atomic_write_json(
        generated_code_manifest,
        _directory_manifest(generated_code_dir, include_symbols=True),
        indent=2,
    )
    return workspace_manifest, generated_code_manifest


class CopilotJAZLLM(BaseLLM):
    """JAZ backend using the hardened TALES Copilot bridge in plain-text mode."""

    def __init__(
        self,
        *,
        bridge: CopilotBridge,
        model: str,
        reasoning_effort: str,
        model_metadata: dict[str, Any],
        telemetry_path: Path,
        generated_code_dir: Path | None = None,
        max_retries: int = 3,
    ) -> None:
        super().__init__(model=model, max_retries=max_retries)
        self.bridge = bridge
        self.reasoning_effort = reasoning_effort
        self.model_metadata = dict(model_metadata)
        self.telemetry_path = telemetry_path
        self.generated_code_dir = (
            Path(generated_code_dir) if generated_code_dir is not None else None
        )
        if self.generated_code_dir is not None:
            self.generated_code_dir.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._completion_attempts = 0
        self._completed_calls = 0
        self._provider_api_calls = 0
        self._prompt_tokens = 0
        self._completion_tokens = 0
        self._reasoning_tokens = 0
        self._cached_tokens = 0
        self._cache_write_tokens = 0
        self._aiu_nano = 0

    def complete(self, model: str, messages: list[Any], **kwargs: Any) -> LLMResponse:
        if model != self.model:
            raise ValueError(f"Configured model {self.model!r}, received {model!r}")
        if kwargs:
            raise ValueError(
                f"CopilotJAZLLM does not accept per-call parameters: {sorted(kwargs)}"
            )

        system_prompt, prompt, flattened = split_copilot_conversation(messages)
        with self._lock:
            self._completion_attempts += 1
            attempt = self._completion_attempts

        session_id: str | None = None
        call: TextModelCall | None = None
        totals: dict[str, int | float] | None = None
        served_model: str | None = None
        content_filter_triggered = False
        output_limit_exhausted = False
        started = time.time()
        try:
            session_id = self.bridge.create_text_session(
                model=model,
                reasoning_effort=self.reasoning_effort,
                system_prompt=system_prompt,
            )
            call = self.bridge.send_text(session_id=session_id, prompt=prompt)
            totals = call.totals()
            prompt_tokens = int(totals["input_tokens"])
            completion_tokens = int(totals["output_tokens"])
            reasoning_tokens = int(totals["reasoning_tokens"])
            cached_tokens = int(totals["cache_read_tokens"])
            cache_write_tokens = int(totals["cache_write_tokens"])
            provider_api_calls = int(totals["api_calls"])
            aiu_nano = int(totals["aiu_nano"])
            ai_credits = aiu_nano / NANO_AI_CREDITS

            # Account for every reported provider call before validating its
            # provenance so a rejected paid response is never recorded as free.
            with self._lock:
                self._provider_api_calls += provider_api_calls
                self._prompt_tokens += prompt_tokens
                self._completion_tokens += completion_tokens
                self._reasoning_tokens += reasoning_tokens
                self._cached_tokens += cached_tokens
                self._cache_write_tokens += cache_write_tokens
                self._aiu_nano += aiu_nano

            if not call.usage:
                raise CompletionIntegrityError(
                    "Expected at least one provider API usage event for a fresh "
                    "no-tools session."
                )
            content_filter_triggered = any(
                usage.get("content_filter_triggered") for usage in call.usage
            )
            output_limit_exhausted = call.usage[-1].get("finish_reason") == "length"
            if len(call.usage) > 1 and any(
                usage.get("finish_reason") != "length" for usage in call.usage[:-1]
            ):
                raise CompletionIntegrityError(
                    "Multiple provider API usage events are valid only as an "
                    "automatic continuation chain after output-length exhaustion."
                )
            served_models = {
                str(value)
                for value in (
                    call.reported_model,
                    *(usage.get("model") for usage in call.usage),
                )
                if value
            }
            if not served_models:
                raise CompletionIntegrityError(
                    "Copilot did not report a served model in the response or usage event."
                )
            if served_models - {model}:
                raise CompletionIntegrityError(
                    f"Requested model {model!r}, served {sorted(served_models)!r}."
                )
            served_model = next(iter(served_models))
        except Exception as exc:
            rejected_response_file: str | None = None
            if call is not None and call.text and self.generated_code_dir is not None:
                rejected_path = (
                    self.generated_code_dir
                    / f"attempt-{attempt:04d}-rejected.provider.txt"
                )
                storage.atomic_write_text(rejected_path, call.text)
                rejected_response_file = str(rejected_path)
            storage.append_jsonl(
                self.telemetry_path,
                {
                    "attempt": attempt,
                    "status": "error",
                    "transport": "custom",
                    "timestamp": started,
                    "duration": time.time() - started,
                    "model": model,
                    "reasoning_effort": self.reasoning_effort,
                    "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
                    "reported_model": call.reported_model if call else None,
                    "response": call.text if call else None,
                    "reasoning_summary": call.thinking if call else None,
                    "usage": call.usage if call else [],
                    "totals": totals,
                    "rejected_response_file": rejected_response_file,
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                },
            )
            raise
        finally:
            if session_id is not None:
                try:
                    self.bridge.close_session(session_id)
                except BridgeError:
                    pass

        assert call is not None
        assert totals is not None
        assert served_model is not None
        raw_content = call.text
        response_transform_reason: str | None = None
        if content_filter_triggered:
            content = CONTENT_FILTER_REPL_CODE
            response_transform_reason = "content_filter"
        elif output_limit_exhausted:
            content = OUTPUT_LIMIT_REPL_CODE
            response_transform_reason = "output_limit"
        else:
            content = raw_content
        dropped_chars = 0
        if flattened and response_transform_reason is None:
            content, dropped_chars = trim_transcript_continuation(content)

        code_file: str | None = None
        raw_response_file: str | None = None
        with self._lock:
            call_index = self._completed_calls + 1
            try:
                if self.generated_code_dir is not None:
                    code_path = self.generated_code_dir / f"call-{call_index:04d}.py"
                    storage.atomic_write_text(
                        code_path,
                        content if content.endswith("\n") else f"{content}\n",
                    )
                    code_file = str(code_path)
                    if response_transform_reason is not None:
                        raw_path = (
                            self.generated_code_dir
                            / f"call-{call_index:04d}.provider.txt"
                        )
                        storage.atomic_write_text(raw_path, raw_content)
                        raw_response_file = str(raw_path)

                storage.append_jsonl(
                    self.telemetry_path,
                    {
                        "attempt": attempt,
                        "call": call_index,
                        "status": "ok",
                        "transport": "custom",
                        "timestamp": started,
                        "duration": time.time() - started,
                        "model": model,
                        "served_model": served_model,
                        "reported_model": call.reported_model,
                        "reasoning_effort": self.reasoning_effort,
                        "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
                        "system_prompt_sha256": hashlib.sha256(
                            system_prompt.encode()
                        ).hexdigest(),
                        "flattened_conversation": flattened,
                        "response": content,
                        "raw_response": (
                            raw_content
                            if response_transform_reason is not None
                            else None
                        ),
                        "response_transformed": response_transform_reason is not None,
                        "response_transform_reason": response_transform_reason,
                        "generated_code_file": code_file,
                        "raw_response_file": raw_response_file,
                        "reasoning_summary": call.thinking,
                        "reasoning_is_summary": bool(call.thinking),
                        "usage": call.usage,
                        "totals": totals,
                        "cost_usd": None,
                        "ai_credits": ai_credits,
                        "aiu_nano": aiu_nano,
                        "model_multiplier_cost": float(totals["cost"]),
                        "dropped_continuation_chars": dropped_chars,
                    },
                )
            except Exception as exc:
                fallback_file: str | None = None
                fallback_error: str | None = None
                if self.generated_code_dir is not None:
                    fallback_path = (
                        self.generated_code_dir
                        / f"attempt-{attempt:04d}-artifact-error.provider.txt"
                    )
                    try:
                        storage.atomic_write_text(fallback_path, raw_content)
                        fallback_file = str(fallback_path)
                    except Exception as fallback_exc:
                        fallback_error = (
                            f"{type(fallback_exc).__name__}: {fallback_exc}"
                        )
                storage.append_jsonl(
                    self.telemetry_path,
                    {
                        "attempt": attempt,
                        "status": "artifact_error",
                        "transport": "custom",
                        "timestamp": started,
                        "duration": time.time() - started,
                        "model": model,
                        "served_model": served_model,
                        "reported_model": call.reported_model,
                        "reasoning_effort": self.reasoning_effort,
                        "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
                        "response": raw_content,
                        "reasoning_summary": call.thinking,
                        "usage": call.usage,
                        "totals": totals,
                        "fallback_response_file": fallback_file,
                        "fallback_error": fallback_error,
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                    },
                )
                raise
            self._completed_calls = call_index

        extra: dict[str, Any] = {
            "reasoning_tokens": reasoning_tokens,
            "copilot_ai_credits": ai_credits,
            "copilot_aiu_nano": aiu_nano,
            "copilot_model_multiplier_cost": float(totals["cost"]),
            "copilot_model": served_model,
            "copilot_model_calls": provider_api_calls,
            "copilot_content_filter_triggered": content_filter_triggered,
            "copilot_output_limit_exhausted": output_limit_exhausted,
            "copilot_generated_code_file": code_file,
        }
        if call.thinking:
            extra["reasoning_summary"] = call.thinking
            extra["reasoning_is_summary"] = True
        if cache_write_tokens:
            extra["cache_creation_input_tokens"] = cache_write_tokens
        if dropped_chars:
            extra["copilot_dropped_continuation_chars"] = dropped_chars

        return LLMResponse(
            content=content,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            cached_tokens=cached_tokens,
            cost_usd=None,
            extra=extra,
            raw_response={
                "text": call.text,
                "thinking": call.thinking,
                "model": call.model,
                "reported_model": call.reported_model,
                "usage": call.usage,
            },
        )

    def get_model_info(self) -> dict[str, Any]:
        max_input = self.model_metadata.get("max_prompt_tokens")
        if not max_input:
            max_input = self.model_metadata.get("context_window")
        return {"max_input_tokens": int(max_input)} if max_input else {}

    def can_report_cost(self) -> bool:
        # CAPI reports AI-unit consumption and a billing multiplier, not USD.
        return False

    @property
    def non_retryable_exceptions(self) -> tuple[type[BaseException], ...]:
        return (*super().non_retryable_exceptions, CompletionIntegrityError)

    def __deepcopy__(self, memo: dict[int, Any]) -> "CopilotJAZLLM":
        # ConfigOverride deep-copies its effective config. The live bridge and
        # telemetry counters are external shared state, just like JAZ's mock LLM.
        return self

    @property
    def completed_calls(self) -> int:
        with self._lock:
            return self._completed_calls

    def stats(self) -> dict[str, Any]:
        with self._lock:
            return {
                "completion_attempts": self._completion_attempts,
                "completed_calls": self._completed_calls,
                "provider_api_calls": self._provider_api_calls,
                "prompt_tokens": self._prompt_tokens,
                "completion_tokens": self._completion_tokens,
                "reasoning_tokens": self._reasoning_tokens,
                "cached_tokens": self._cached_tokens,
                "cache_write_tokens": self._cache_write_tokens,
                "aiu_nano": self._aiu_nano,
                "ai_credits": self._aiu_nano / NANO_AI_CREDITS,
            }

    def close(self) -> None:
        """The owning run closes the shared Node bridge."""


class OfficialCopilotJAZLLM(BaseLLM):
    """Telemetry wrapper around JAZ's official Copilot backend."""

    def __init__(
        self,
        *,
        model: str,
        reasoning_effort: str,
        model_metadata: dict[str, Any],
        telemetry_path: Path,
        generated_code_dir: Path | None,
        base_directory: Path,
        cli_path: str | None,
        call_timeout: float,
        max_retries: int,
        backend: Any | None = None,
    ) -> None:
        super().__init__(model=model, max_retries=max_retries)
        self.reasoning_effort = reasoning_effort
        self.model_metadata = dict(model_metadata)
        self.telemetry_path = telemetry_path
        self.generated_code_dir = (
            Path(generated_code_dir) if generated_code_dir is not None else None
        )
        if self.generated_code_dir is not None:
            self.generated_code_dir.mkdir(parents=True, exist_ok=True)
        self.cli_path = cli_path
        self.backend = backend or CopilotLLM(
            model=model,
            reasoning_effort=reasoning_effort,
            timeout=call_timeout,
            base_directory=str(base_directory),
            max_retries=0,
        )
        self._lock = threading.RLock()
        self._completion_attempts = 0
        self._completed_calls = 0
        self._provider_api_calls = 0
        self._prompt_tokens = 0
        self._completion_tokens = 0
        self._reasoning_tokens = 0
        self._cached_tokens = 0
        self._cache_write_tokens = 0
        self._aiu_nano = 0

    def complete(self, model: str, messages: list[Any], **kwargs: Any) -> LLMResponse:
        if model != self.model:
            raise ValueError(f"Configured model {self.model!r}, received {model!r}")
        if kwargs:
            raise ValueError(
                "OfficialCopilotJAZLLM does not accept per-call parameters: "
                f"{sorted(kwargs)}"
            )

        system_prompt, prompt, flattened = split_copilot_conversation(messages)
        with self._lock:
            self._completion_attempts += 1
            attempt = self._completion_attempts
        started = time.time()
        previous_cli_path = os.environ.get("COPILOT_CLI_PATH")
        if self.cli_path is not None:
            os.environ["COPILOT_CLI_PATH"] = self.cli_path
        try:
            response = self.backend.complete(model, messages)
        except Exception as exc:
            storage.append_jsonl(
                self.telemetry_path,
                {
                    "attempt": attempt,
                    "status": "error",
                    "transport": "official",
                    "timestamp": started,
                    "duration": time.time() - started,
                    "model": model,
                    "reasoning_effort": self.reasoning_effort,
                    "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                },
            )
            raise
        finally:
            if self.cli_path is not None:
                if previous_cli_path is None:
                    os.environ.pop("COPILOT_CLI_PATH", None)
                else:
                    os.environ["COPILOT_CLI_PATH"] = previous_cli_path

        extra = dict(response.extra or {})
        provider_api_calls = int(extra.get("copilot_model_calls") or 1)
        prompt_tokens = int(response.prompt_tokens or 0)
        completion_tokens = int(response.completion_tokens or 0)
        reasoning_tokens = int(extra.get("reasoning_tokens") or 0)
        cached_tokens = int(response.cached_tokens or 0)
        cache_write_tokens = int(extra.get("cache_creation_input_tokens") or 0)
        ai_credits = float(extra.get("copilot_ai_credits") or 0.0)
        aiu_nano = int(round(ai_credits * NANO_AI_CREDITS))
        served_model = extra.get("copilot_model")

        with self._lock:
            self._provider_api_calls += provider_api_calls
            self._prompt_tokens += prompt_tokens
            self._completion_tokens += completion_tokens
            self._reasoning_tokens += reasoning_tokens
            self._cached_tokens += cached_tokens
            self._cache_write_tokens += cache_write_tokens
            self._aiu_nano += aiu_nano

        if served_model is not None and served_model != model:
            error = CompletionIntegrityError(
                f"Requested model {model!r}, served {served_model!r}."
            )
            storage.append_jsonl(
                self.telemetry_path,
                {
                    "attempt": attempt,
                    "status": "error",
                    "transport": "official",
                    "timestamp": started,
                    "duration": time.time() - started,
                    "model": model,
                    "served_model": served_model,
                    "reasoning_effort": self.reasoning_effort,
                    "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
                    "response": response.content,
                    "error_type": type(error).__name__,
                    "error": str(error),
                },
            )
            raise error

        with self._lock:
            self._completed_calls += 1
            call_index = self._completed_calls

        content = response.content or ""
        code_file: str | None = None
        if self.generated_code_dir is not None:
            code_path = self.generated_code_dir / f"call-{call_index:04d}.py"
            storage.atomic_write_text(
                code_path,
                content if content.endswith("\n") else f"{content}\n",
            )
            code_file = str(code_path)

        storage.append_jsonl(
            self.telemetry_path,
            {
                "attempt": attempt,
                "call": call_index,
                "status": "ok",
                "transport": "official",
                "timestamp": started,
                "duration": time.time() - started,
                "model": model,
                "served_model": served_model,
                "reasoning_effort": self.reasoning_effort,
                "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
                "system_prompt_sha256": hashlib.sha256(
                    system_prompt.encode()
                ).hexdigest(),
                "flattened_conversation": flattened,
                "response": content,
                "generated_code_file": code_file,
                "reasoning_summary": None,
                "reasoning_is_summary": False,
                "usage": [],
                "totals": {
                    "input_tokens": prompt_tokens,
                    "output_tokens": completion_tokens,
                    "reasoning_tokens": reasoning_tokens,
                    "cache_read_tokens": cached_tokens,
                    "cache_write_tokens": cache_write_tokens,
                    "cost_usd": response.cost_usd,
                    "aiu_nano": aiu_nano,
                    "api_calls": provider_api_calls,
                },
                "cost_usd": response.cost_usd,
                "ai_credits": ai_credits,
                "aiu_nano": aiu_nano,
                "dropped_continuation_chars": int(
                    extra.get("copilot_dropped_continuation_chars") or 0
                ),
            },
        )
        response.extra["copilot_generated_code_file"] = code_file
        return response

    def get_model_info(self) -> dict[str, Any]:
        max_input = self.model_metadata.get("max_prompt_tokens")
        if not max_input:
            max_input = self.model_metadata.get("context_window")
        return {"max_input_tokens": int(max_input)} if max_input else {}

    def can_report_cost(self) -> bool:
        return True

    @property
    def non_retryable_exceptions(self) -> tuple[type[BaseException], ...]:
        return (*super().non_retryable_exceptions, CompletionIntegrityError)

    def __deepcopy__(self, memo: dict[int, Any]) -> "OfficialCopilotJAZLLM":
        return self

    @property
    def completed_calls(self) -> int:
        with self._lock:
            return self._completed_calls

    def stats(self) -> dict[str, Any]:
        with self._lock:
            return {
                "completion_attempts": self._completion_attempts,
                "completed_calls": self._completed_calls,
                "provider_api_calls": self._provider_api_calls,
                "prompt_tokens": self._prompt_tokens,
                "completion_tokens": self._completion_tokens,
                "reasoning_tokens": self._reasoning_tokens,
                "cached_tokens": self._cached_tokens,
                "cache_write_tokens": self._cache_write_tokens,
                "aiu_nano": self._aiu_nano,
                "ai_credits": self._aiu_nano / NANO_AI_CREDITS,
            }

    def close(self) -> None:
        close = getattr(self.backend, "close", None)
        if callable(close):
            close()


class JerichoGameController:
    """Safe, observable boundary between a JAZ REPL and one Jericho game."""

    def __init__(
        self,
        env: Any,
        *,
        game: str,
        seed: int,
        action_safety_limit: int | None,
        native_step_timeout: float,
        action_log_path: Path,
        model_calls_completed: Callable[[], int],
    ) -> None:
        self._env = env
        self._game = game
        self._seed = int(seed)
        self._action_safety_limit = (
            int(action_safety_limit) if action_safety_limit is not None else None
        )
        self._native_step_timeout = float(native_step_timeout)
        self._action_log_path = action_log_path
        self._model_calls_completed = model_calls_completed
        self._lock = threading.RLock()

        observation, info = env.reset(seed=self._seed)
        max_score = int(info.get("max_score") or 0)
        self._initial_observation = observation
        self._state: dict[str, Any] = {
            "obs": observation,
            "info": info,
            "done": False,
            "score": int(info.get("score") or 0),
            "moves": int(info.get("moves") or 0),
            "highscore": int(info.get("score") or 0),
            "max_score": max_score,
            "nb_wins": 0,
            "nb_losts": 0,
            "nb_resets": 0,
        }
        self._actions_used = 0
        self._actions_executed = 0
        self._blocked_actions = 0
        self._invalid_actions = 0
        self._environment_errors = 0
        self._completion_reason: str | None = None

    @property
    def initial_observation(self) -> str:
        return self._initial_observation

    def _refresh_completion_reason(self) -> None:
        if self._state["max_score"] > 0 and (
            self._state["highscore"] >= self._state["max_score"]
        ):
            self._completion_reason = "max_score"
        elif (
            self._action_safety_limit is not None
            and self._actions_used >= self._action_safety_limit
        ):
            self._completion_reason = "action_safety_limit"

    def is_complete(self) -> bool:
        with self._lock:
            self._refresh_completion_reason()
            return self._completion_reason is not None

    def _view(
        self,
        *,
        feedback: str | None = None,
        submitted_action: str | None = None,
        executed_action: str | None = None,
        action_was_executed: bool = False,
        invalid_reason: str | None = None,
        blocked_command: str | None = None,
        reward: int = 0,
        episode_done: bool = False,
        episode_won: bool = False,
        episode_lost: bool = False,
        restarted: bool = False,
    ) -> GameState:
        self._refresh_completion_reason()
        return GameState(
            observation=str(self._state["obs"]),
            feedback=str(
                feedback
                if feedback is not None
                else self._state["info"].get("feedback") or self._state["obs"]
            ),
            submitted_action=submitted_action,
            executed_action=executed_action,
            action_was_executed=action_was_executed,
            invalid_reason=invalid_reason,
            blocked_command=blocked_command,
            reward=int(reward),
            score=int(self._state["score"]),
            highscore=int(self._state["highscore"]),
            max_score=int(self._state["max_score"]),
            moves=int(self._state["moves"]),
            episode_done=bool(episode_done),
            episode_won=bool(episode_won),
            episode_lost=bool(episode_lost),
            restarted=bool(restarted),
            actions_used=self._actions_used,
            actions_remaining=None,
            action_safety_remaining=(
                max(self._action_safety_limit - self._actions_used, 0)
                if self._action_safety_limit is not None
                else None
            ),
            run_complete=self._completion_reason is not None,
            completion_reason=self._completion_reason,
        )

    def observe(self) -> GameState:
        """Return the current visible game observation and public run status."""
        with self._lock:
            return self._view()

    def step(self, action: str) -> GameState:
        """Execute one text-adventure command and return the resulting visible state.

        The return value exposes only ordinary TALES agent information: game text,
        score, moves, terminal flags, reset status, and action accounting.
        Admissible commands, walkthroughs, world objects, and interpreter state are
        never exposed. Invalid or forbidden actions are counted as attempts but do
        not call the environment.
        """
        with self._lock:
            self._refresh_completion_reason()
            if self._completion_reason == "action_safety_limit":
                raise ActionSafetyLimitExceededError(
                    "The operational game-action safety limit was reached. "
                    "Return immediately; this run is invalid and will not be scored."
                )
            if self._completion_reason is not None:
                raise GameCompleteError(
                    "The game is complete; return the final result without more actions."
                )

            self._actions_used += 1
            score_before = int(self._state["score"])
            observation_before = str(self._state["obs"])
            submitted_action = action if isinstance(action, str) else repr(action)[:500]
            normalized = (
                schema.normalize_game_action(action)
                if isinstance(action, str)
                else None
            )
            invalid_reason: str | None = None
            blocked_command: str | None = None
            executed_action: str | None = None
            action_was_executed = False
            reward = 0
            episode_done = False
            episode_won = False
            episode_lost = False
            restarted = False

            if not isinstance(action, str):
                self._invalid_actions += 1
                invalid_reason = "Action must be a string."
                feedback = (
                    "The rejected action was discarded; this action attempt was wasted."
                )
            elif normalized is None:
                self._invalid_actions += 1
                invalid_reason = (
                    "Action must be one printable command of at most "
                    f"{schema.MAX_GAME_ACTION_BYTES} UTF-8 bytes."
                )
                feedback = (
                    "The rejected action was discarded; this action attempt was wasted."
                )
            else:
                blocked_command = benchmark._blocked_state_command(normalized)
                if blocked_command is not None:
                    self._blocked_actions += 1
                    feedback = (
                        f"The disabled interpreter command {blocked_command!r} was "
                        "rejected; this action attempt was wasted."
                    )
                elif _COMPOUND_COMMAND_RE.search(normalized):
                    self._invalid_actions += 1
                    invalid_reason = (
                        "Compound parser commands are disabled. Call step() separately "
                        "for each command instead of using periods, AND, or THEN."
                    )
                    feedback = "The compound action was discarded; this action attempt was wasted."
                else:
                    action_was_executed = True
                    executed_action = schema.native_game_action(normalized)
                    self._actions_executed += 1
                    try:
                        _, feedback = episode.step_env(
                            self._env,
                            self._state,
                            normalized,
                            timeout=self._native_step_timeout,
                        )
                    except Exception as exc:
                        self._environment_errors += 1
                        self._refresh_completion_reason()
                        storage.append_jsonl(
                            self._action_log_path,
                            {
                                "action_index": self._actions_used,
                                "timestamp": time.time(),
                                "model_calls_completed": int(
                                    self._model_calls_completed()
                                ),
                                "game": self._game,
                                "seed": self._seed,
                                "observation_before": observation_before,
                                "submitted_action": submitted_action,
                                "executed_action": executed_action,
                                "action_was_executed": True,
                                "error_type": type(exc).__name__,
                                "error": str(exc),
                                "actions_used": self._actions_used,
                                "actions_remaining": None,
                                "action_safety_remaining": (
                                    max(
                                        self._action_safety_limit - self._actions_used,
                                        0,
                                    )
                                    if self._action_safety_limit is not None
                                    else None
                                ),
                                "completion_reason": self._completion_reason,
                            },
                        )
                        raise
                    reward = int(self._state["score"]) - score_before
                    episode_done = bool(self._state["done"])
                    episode_won = bool(self._state["info"].get("won"))
                    episode_lost = bool(self._state["info"].get("lost"))
                    done_result = episode.handle_done(
                        self._env,
                        self._state,
                        self._seed,
                    )
                    restarted = done_result == "reset"
                    if restarted:
                        self._state["score"] = int(
                            self._state["info"].get("score") or 0
                        )
                        self._state["moves"] = int(
                            self._state["info"].get("moves") or 0
                        )

            view = self._view(
                feedback=feedback,
                submitted_action=submitted_action,
                executed_action=executed_action,
                action_was_executed=action_was_executed,
                invalid_reason=invalid_reason,
                blocked_command=blocked_command,
                reward=reward,
                episode_done=episode_done,
                episode_won=episode_won,
                episode_lost=episode_lost,
                restarted=restarted,
            )
            storage.append_jsonl(
                self._action_log_path,
                {
                    "action_index": self._actions_used,
                    "timestamp": time.time(),
                    "model_calls_completed": int(self._model_calls_completed()),
                    "game": self._game,
                    "seed": self._seed,
                    "observation_before": observation_before,
                    **view.to_dict(),
                },
            )
            return view

    def summary(self) -> dict[str, Any]:
        with self._lock:
            self._refresh_completion_reason()
            max_score = int(self._state["max_score"])
            highscore = int(self._state["highscore"])
            return {
                "game": self._game,
                "seed": self._seed,
                "score": highscore,
                "max_score": max_score,
                "normalized_score": (
                    min(max(highscore / max_score, 0.0), 1.0) if max_score else 0.0
                ),
                "current_score": int(self._state["score"]),
                "moves": int(self._state["moves"]),
                "action_safety_limit": self._action_safety_limit,
                "action_safety_limit_hit": (
                    self._completion_reason == "action_safety_limit"
                ),
                "actions_used": self._actions_used,
                "actions_executed": self._actions_executed,
                "invalid_actions": self._invalid_actions,
                "blocked_actions": self._blocked_actions,
                "environment_errors": self._environment_errors,
                "wins": int(self._state["nb_wins"]),
                "losses": int(self._state["nb_losts"]),
                "resets": int(self._state["nb_resets"]),
                "completion_reason": self._completion_reason,
                "initial_observation": self._initial_observation,
                "final_observation": str(self._state["obs"]),
            }


def _instructions(config: JAZRunConfig, game_title: str) -> str:
    action_safety = (
        "There is no score-ending game-action budget."
        if config.action_safety_limit is None
        else (
            "There is no score-ending game-action budget. A non-scoring "
            f"operational guard stops runaway execution after "
            f"{config.action_safety_limit} calls to `step`; reaching it "
            "invalidates the run."
        )
    )
    return f"""\
Play {game_title} ({config.game}) through the TALES/Jericho environment and maximize
the numerical score. The shared budget is {config.max_model_calls} logical LLM
calls across the root invoke and all recursive invokes. {action_safety}

The scoped functions `step(action: str)` and `observe()` are the only game
interfaces. They return structured GameState values; print a value when you need
to inspect it on the next REPL iteration. You may call `step` more than once in
one code response when executing a sequence you are confident about. Every call
must contain exactly one game command: do not combine commands with periods,
AND, or THEN.

For single-key prompts, use `step(" ")` for SPACE or any-key screens,
`step("<ENTER>")` for Enter/Return, and `step("<BACKSPACE>")` for Backspace.
Literal control characters are rejected; use these printable markers instead.

Saving, loading, restoring, undoing, interpreter transcription, debugging,
teleportation, object manipulation, command recording, and replay are disabled.
Do not attempt to inspect Python objects for hidden environment state. The
harness automatically restarts after death or a sub-maximal ending while keeping
your REPL variables and history.

The scoped `workspace_dir` string names your persistent writable directory.
Ordinary `open(...)` may read or write only within that directory. Use it for
durable notes, helper code, or outputs when useful; every file remains attached
to the run. Every Python response is also archived automatically, so do not
duplicate code merely for logging.

Finish only after the returned GameState says the run is complete. If maximum
score is not reached first, the harness terminates and scores the run when the
shared LLM-call budget is exhausted."""


def _guidance(config: JAZRunConfig) -> str:
    if config.return_guard_scope == "tree":
        delegation = """\
Recursive `invoke(...)` is available and inherits the scoped environment
functions. Use it only to delegate continuation of the same unfinished workflow,
not for independent speculation. When delegating, pass the original
`instructions` and `guidance`, the current `observe()` result, a concise progress
record, next steps, and
`prev_history=globals().get("prev_history", []) + __history__`; then return the
delegate's result so the workflow continues tail-recursively."""
    else:
        delegation = """\
Recursive `invoke(...)` is available and inherits the scoped environment
functions. You may use bounded sub-invokes for planning, puzzle analysis, or
history search; they may return partial results to you. For a full context
handoff, tail-delegate the unfinished game by passing the original
`instructions` and `guidance`, the current `observe()` result, a concise progress
record, next steps, and
`prev_history=globals().get("prev_history", []) + __history__`, then return the
delegate's result."""
    return f"""\
Use the persistent Python REPL as your working memory. Keep concise structured
notes about established facts, attempted actions, unresolved goals, and the next
few actions. Work incrementally: call the available functions, print the
resulting evidence, stop that code block, and inspect the output on the next
iteration before making uncertain follow-up decisions. Use `__history__` as a
searchable variable instead of repeatedly restating the full interaction.

{delegation}"""


def _context_warning() -> str:
    return """\
Your context is nearing its limit. Tail-delegate the unfinished workflow now.
Call `invoke(...)` with the original instructions and guidance, the current
`observe()` result, concise progress and next steps, and
`prev_history=globals().get("prev_history", []) + __history__`. Return the
delegate's result. The scoped `step` and `observe` functions are inherited."""


def _budget_warning() -> str:
    return """\
The shared model-call budget is nearly exhausted. Stop speculative work, use
the remaining calls only for the highest-value game progress, and return as soon
as the GameState reports that the run is complete."""


def _return_validator(controller: JerichoGameController, _value: Any) -> None:
    if controller.is_complete():
        return
    state = controller.observe()
    raise ValueError(
        "The Jericho run is not complete. Continue playing instead of returning: "
        f"highscore={state.highscore}/{state.max_score}."
    )


def _is_budget_exhaustion(exc: BaseException) -> bool:
    if isinstance(exc, BudgetExhaustedError):
        return True
    if isinstance(exc, BaseExceptionGroup):
        return bool(exc.exceptions) and all(
            _is_budget_exhaustion(child) for child in exc.exceptions
        )
    return False


def _is_context_budget_exhaustion(exc: BaseException) -> bool:
    if isinstance(exc, BaseExceptionGroup):
        return bool(exc.exceptions) and all(
            _is_context_budget_exhaustion(child) for child in exc.exceptions
        )
    message = str(exc).lower()
    return any(
        marker in message
        for marker in (
            "request exceeds the model's prompt budget",
            "context window exceeded",
            "maximum context length",
            "prompt is too long",
        )
    )


def _originated_in_repl(exc: BaseException) -> bool:
    traceback_cursor = exc.__traceback__
    while traceback_cursor is not None:
        filename = traceback_cursor.tb_frame.f_code.co_filename
        if filename.startswith("<repl_"):
            return True
        traceback_cursor = traceback_cursor.tb_next
    return False


def _resolve_game(game_name: str):
    matches = [
        game
        for game in games.list_split_games("jericho-all")
        if game.env_name == game_name
    ]
    if not matches:
        raise ValueError(f"Unknown Jericho environment: {game_name!r}")
    return matches[0]


def _prepare_run_dir(config: JAZRunConfig) -> tuple[Path, bool]:
    run_dir = config.run_dir
    summary_path = run_dir / "summary.json"
    existing = storage.read_json(summary_path)
    if isinstance(existing, dict) and not config.force:
        existing_config = dict(existing.get("config") or {})
        requested_config = config.to_dict()
        existing_config.pop("force", None)
        requested_config.pop("force", None)
        reusable = bool(existing.get("score_valid")) and not str(
            existing.get("status") or ""
        ).startswith("failed")
        if reusable and existing_config == requested_config:
            return run_dir, True
        raise FileExistsError(
            f"An incompatible or failed run already exists at {run_dir}. "
            "Pass --force to replace this exact run directory."
        )
    if run_dir.exists():
        if not config.force:
            raise FileExistsError(
                f"Partial run exists without a reusable summary: {run_dir}. "
                "Pass --force to replace this exact run directory."
            )
        if run_dir == config.output_root or config.output_root not in run_dir.parents:
            raise ValueError(f"Refusing to remove unsafe run directory: {run_dir}")
        storage.remove_tree(run_dir)
    run_dir.mkdir(parents=True)
    return run_dir, False


def run_jaz_episode(config: JAZRunConfig) -> dict[str, Any]:
    """Run one JAZ Jericho episode and persist all score/trace artifacts."""
    game_spec = _resolve_game(config.game)
    run_dir, skipped = _prepare_run_dir(config)
    summary_path = run_dir / "summary.json"
    if skipped:
        summary = storage.read_json(summary_path)
        assert isinstance(summary, dict)
        return {**summary, "skipped": True}

    storage.atomic_write_json(run_dir / "config.json", config.to_dict(), indent=2)
    action_log = run_dir / "actions.jsonl"
    model_log = run_dir / "model_calls.jsonl"
    trace_path = run_dir / "trace.atif.json"
    bridge_log = run_dir / "copilot-bridge.log"
    work_dir = run_dir / "work"
    game_work_dir = work_dir / "game"
    copilot_work_dir = work_dir / "copilot"
    workspace_dir = config.workspace_dir
    generated_code_dir = config.generated_code_dir
    workspace_dir.mkdir(parents=True, exist_ok=True)
    generated_code_dir.mkdir(parents=True, exist_ok=True)
    workspace_manifest_path: Path | None = None
    generated_code_manifest_path: Path | None = None
    artifact_manifest_error: str | None = None

    bridge_config = BridgeConfig(
        output_root=run_dir,
        call_timeout=config.call_timeout,
        reasoning_summary="detailed",
        sdk_path=config.sdk_path,
        cli_path=config.cli_path,
        copilot_home=config.copilot_home,
        bridge_log_level=config.bridge_log_level,
    )
    started = time.time()
    status = "failed"
    score_valid = False
    result_repr: str | None = None
    error_type: str | None = None
    error_message: str | None = None
    error_traceback: str | None = None
    controller: JerichoGameController | None = None
    backend: CopilotJAZLLM | OfficialCopilotJAZLLM | None = None
    bridge: CopilotBridge | None = None
    env = None

    try:
        bridge = CopilotBridge(
            bridge_config,
            log_path=bridge_log,
            working_directory=copilot_work_dir,
        )
        raw_models = bridge.list_models()
        raw_model = next(
            (item for item in raw_models if item.get("id") == config.model),
            None,
        )
        if raw_model is None:
            raise ValueError(f"Copilot model is unavailable: {config.model}")
        model_spec = build_model_spec(raw_model)
        if config.reasoning_effort not in model_spec.supported_efforts:
            raise ValueError(
                f"{config.model} does not advertise reasoning effort "
                f"{config.reasoning_effort!r}: {model_spec.supported_efforts}"
            )
        storage.atomic_write_json(
            run_dir / "model.json",
            model_spec.to_dict(),
            indent=2,
        )

        env = games.make_env(
            config.game,
            admissible_commands=False,
            working_directory=game_work_dir,
        )
        if config.transport == "custom":
            backend = CopilotJAZLLM(
                bridge=bridge,
                model=config.model,
                reasoning_effort=config.reasoning_effort,
                model_metadata=raw_model,
                telemetry_path=model_log,
                generated_code_dir=generated_code_dir,
                max_retries=config.max_retries,
            )
        else:
            bridge.stop()
            bridge = None
            backend = OfficialCopilotJAZLLM(
                model=config.model,
                reasoning_effort=config.reasoning_effort,
                model_metadata=raw_model,
                telemetry_path=model_log,
                generated_code_dir=generated_code_dir,
                base_directory=Path(config.copilot_home),
                cli_path=config.cli_path,
                call_timeout=config.call_timeout,
                max_retries=config.max_retries,
            )
        controller = JerichoGameController(
            env,
            game=config.game,
            seed=config.seed,
            action_safety_limit=config.action_safety_limit,
            native_step_timeout=config.native_step_timeout,
            action_log_path=action_log,
            model_calls_completed=lambda: backend.completed_calls,
        )

        repl = PythonREPL(
            exec_timeout=config.repl_timeout,
            exec_memory_limit=config.repl_memory_limit_mb * 1024 * 1024,
            allow_timeout_pragma=False,
            allowed_imports=list(ALLOWED_IMPORTS),
            allowed_read_paths=[config.workspace_glob],
            allowed_write_paths=[config.workspace_glob],
            traceback_verbosity="repl_only",
        )
        protocol = CodeOnlyProtocol(
            max_invoke_input_length=config.max_invoke_input_length,
            max_repl_output_length=config.max_repl_output_length,
        )
        recorder = TrajectoryRecorder(trace_path)
        call_warning_threshold = min(
            10,
            max(config.max_model_calls // 20, 1),
        )
        budget = BudgetPool(
            calls_budget=config.max_model_calls,
            warn_calls_remaining=call_warning_threshold,
            warning_text=_budget_warning(),
        )
        recursion_limit = RecursionLimit(max_depth=config.max_recursion_depth)
        context_warning = ContextWindowWarning(
            warn_fraction=config.context_warn_fraction,
            warning_text=_context_warning(),
        )
        return_guard = ValidateReturn(
            lambda value: _return_validator(controller, value),
            max_failures=None,
        )

        with ExitStack() as stack:
            stack.enter_context(
                jaz.ConfigOverride(llm=backend, repl=repl, protocol=protocol)
            )
            stack.enter_context(
                jaz.scope(
                    step=controller.step,
                    observe=controller.observe,
                    workspace_dir=str(workspace_dir),
                )
            )
            stack.enter_context(recorder)
            stack.enter_context(budget)
            stack.enter_context(recursion_limit)
            stack.enter_context(context_warning)
            if config.return_guard_scope == "tree":
                stack.enter_context(return_guard)
                root_hooks: tuple[Any, ...] = ()
            else:
                root_hooks = (return_guard,)
            result = jaz.invoke(
                *root_hooks,
                instructions=_instructions(config, game_spec.title),
                guidance=_guidance(config),
                initial_state=controller.observe(),
                prev_history=[],
            )
            result_repr = repr(result)[:10_000]

        game_result = controller.summary()
        if game_result["environment_errors"]:
            status = "failed_environment_step"
            score_valid = False
        else:
            score_valid = True
            completion_reason = game_result["completion_reason"]
            if completion_reason == "action_safety_limit":
                status = "failed_action_safety_limit"
                score_valid = False
            elif completion_reason != "max_score":
                raise RuntimeError(
                    "Return validation accepted an incomplete Jericho run."
                )
            else:
                status = "completed_max_score"
    except Exception as exc:
        error_type = type(exc).__name__
        error_message = str(exc)
        if _is_budget_exhaustion(exc):
            game_result = controller.summary() if controller is not None else {}
            if game_result.get("environment_errors"):
                status = "failed_environment_step"
                score_valid = False
            elif game_result.get("completion_reason") == "max_score":
                status = "completed_max_score"
                score_valid = True
            elif game_result.get("completion_reason") == "action_safety_limit":
                status = "failed_action_safety_limit"
                score_valid = False
            else:
                status = "model_call_budget_exhausted"
                score_valid = controller is not None
        elif _is_context_budget_exhaustion(exc):
            game_result = controller.summary() if controller is not None else {}
            if game_result.get("environment_errors"):
                status = "failed_environment_step"
                score_valid = False
            elif game_result.get("completion_reason") == "max_score":
                status = "completed_max_score"
                score_valid = True
            elif game_result.get("completion_reason") == "action_safety_limit":
                status = "failed_action_safety_limit"
                score_valid = False
            else:
                status = "model_context_budget_exhausted"
                score_valid = controller is not None
        elif controller is not None and _originated_in_repl(exc):
            game_result = controller.summary()
            if game_result.get("environment_errors"):
                status = "failed_environment_step"
                score_valid = False
            elif game_result.get("completion_reason") == "max_score":
                status = "completed_max_score"
                score_valid = True
            elif game_result.get("completion_reason") == "action_safety_limit":
                status = "failed_action_safety_limit"
                score_valid = False
            else:
                status = "model_declared_impossible"
                score_valid = True
            error_traceback = traceback.format_exc()
            storage.atomic_write_text(run_dir / "error.txt", error_traceback)
        else:
            status = "failed"
            error_traceback = traceback.format_exc()
            storage.atomic_write_text(run_dir / "error.txt", error_traceback)
    finally:
        if env is not None:
            try:
                env.close()
            except Exception:
                pass
        if backend is not None:
            backend.close()
        if bridge is not None:
            bridge.stop()
        try:
            (
                workspace_manifest_path,
                generated_code_manifest_path,
            ) = _write_artifact_manifests(
                run_dir,
                workspace_dir,
                generated_code_dir,
            )
        except Exception as exc:
            artifact_manifest_error = f"{type(exc).__name__}: {exc}"
            if not status.startswith("failed"):
                status = "failed_artifact_capture"
                score_valid = False

    game_summary = (
        controller.summary()
        if controller is not None
        else {
            "game": config.game,
            "seed": config.seed,
            "score": 0,
            "max_score": 0,
            "normalized_score": 0.0,
            "completion_reason": None,
        }
    )
    model_usage = backend.stats() if backend is not None else {}
    resource_usage = {
        "logical_model_calls": int(model_usage.get("completed_calls") or 0),
        "provider_api_calls": int(model_usage.get("provider_api_calls") or 0),
        "game_action_attempts": int(game_summary.get("actions_used") or 0),
        "game_actions_executed": int(game_summary.get("actions_executed") or 0),
        "prompt_tokens": int(model_usage.get("prompt_tokens") or 0),
        "completion_tokens": int(model_usage.get("completion_tokens") or 0),
        "reasoning_tokens": int(model_usage.get("reasoning_tokens") or 0),
        "ai_credits": float(model_usage.get("ai_credits") or 0.0),
    }
    summary = {
        "run_id": config.run_id,
        "status": status,
        "score_valid": score_valid,
        "skipped": False,
        "model": config.model,
        "model_family": provider_family(config.model),
        "reasoning_effort": config.reasoning_effort,
        "transport": config.transport,
        "return_guard_scope": config.return_guard_scope,
        "harness": "jaz-invoke",
        "paper": "arXiv:2609.26891",
        "duration_seconds": time.time() - started,
        "result": result_repr,
        "error_type": error_type,
        "error": error_message,
        "error_traceback_file": str(run_dir / "error.txt") if error_traceback else None,
        "artifact_manifest_error": artifact_manifest_error,
        "config": config.to_dict(),
        "game_result": game_summary,
        "model_usage": model_usage,
        "resource_usage": resource_usage,
        "artifacts": {
            "run_dir": str(run_dir),
            "actions": str(action_log),
            "model_calls": str(model_log),
            "atif_trace": str(trace_path),
            "bridge_log": str(bridge_log),
            "workspace": str(workspace_dir),
            "workspace_manifest": (
                str(workspace_manifest_path)
                if workspace_manifest_path is not None
                else None
            ),
            "generated_code": str(generated_code_dir),
            "generated_code_manifest": (
                str(generated_code_manifest_path)
                if generated_code_manifest_path is not None
                else None
            ),
        },
        "versions": {
            "jaz": getattr(jaz, "__version__", "unknown"),
        },
    }
    storage.atomic_write_json(summary_path, summary, indent=2)
    return summary


def aggregate_results(
    configs: Iterable[JAZRunConfig],
    summaries: Iterable[dict[str, Any]],
) -> dict[str, Any]:
    configs = list(configs)
    summaries = list(summaries)
    by_model: dict[str, list[dict[str, Any]]] = {}
    for summary in summaries:
        by_model.setdefault(summary["model"], []).append(summary)

    aggregates: dict[str, Any] = {}
    for model, rows in sorted(by_model.items()):
        valid = [row for row in rows if row.get("score_valid")]
        normalized = [float(row["game_result"]["normalized_score"]) for row in valid]
        aggregates[model] = {
            "runs": len(rows),
            "valid_runs": len(valid),
            "failed_runs": sum(
                str(row.get("status") or "").startswith("failed") for row in rows
            ),
            "mean_normalized_score": (
                sum(normalized) / len(normalized) if normalized else None
            ),
            "max_normalized_score": max(normalized) if normalized else None,
            "scores": [
                {
                    "game": row["game_result"]["game"],
                    "seed": row["game_result"]["seed"],
                    "score": row["game_result"]["score"],
                    "max_score": row["game_result"]["max_score"],
                    "normalized_score": row["game_result"]["normalized_score"],
                    "status": row["status"],
                    "logical_model_calls": row.get("resource_usage", {}).get(
                        "logical_model_calls", 0
                    ),
                    "provider_api_calls": row.get("resource_usage", {}).get(
                        "provider_api_calls", 0
                    ),
                    "game_action_attempts": row.get("resource_usage", {}).get(
                        "game_action_attempts", 0
                    ),
                    "ai_credits": row.get("resource_usage", {}).get("ai_credits", 0.0),
                }
                for row in sorted(
                    rows,
                    key=lambda item: (
                        item["game_result"]["game"],
                        item["game_result"]["seed"],
                    ),
                )
            ],
            "total_model_calls": sum(
                int(row.get("model_usage", {}).get("completed_calls") or 0)
                for row in rows
            ),
            "total_provider_api_calls": sum(
                int(row.get("model_usage", {}).get("provider_api_calls") or 0)
                for row in rows
            ),
            "total_game_action_attempts": sum(
                int(row.get("game_result", {}).get("actions_used") or 0) for row in rows
            ),
            "total_game_actions_executed": sum(
                int(row.get("game_result", {}).get("actions_executed") or 0)
                for row in rows
            ),
            "total_aiu_nano": sum(
                int(row.get("model_usage", {}).get("aiu_nano") or 0) for row in rows
            ),
            "total_ai_credits": sum(
                float(row.get("model_usage", {}).get("ai_credits") or 0.0)
                for row in rows
            ),
        }

    return {
        "generated_at": time.time(),
        "paper": "arXiv:2609.26891",
        "harness": "jaz-invoke",
        "jobs": len(configs),
        "models": sorted({config.model for config in configs}),
        "games": sorted({config.game for config in configs}),
        "seeds": sorted({config.seed for config in configs}),
        "settings": {
            "max_model_calls": sorted({config.max_model_calls for config in configs}),
            "action_safety_limits": sorted(
                {config.action_safety_limit for config in configs},
                key=lambda value: (value is None, value or 0),
            ),
            "return_guard_scopes": sorted(
                {config.return_guard_scope for config in configs}
            ),
            "transports": sorted({config.transport for config in configs}),
            "reasoning_efforts": sorted(
                {config.reasoning_effort for config in configs}
            ),
            "max_recursion_depth": sorted(
                {config.max_recursion_depth for config in configs}
            ),
            "context_warn_fraction": sorted(
                {config.context_warn_fraction for config in configs}
            ),
        },
        "aggregates": aggregates,
        "summaries": summaries,
    }
