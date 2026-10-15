"""Python client for ``scripts/copilot_model_bridge.mjs``.

The bridge keeps one Copilot CLI runtime alive per worker and exposes a JSONL
RPC surface. One environment step costs exactly one model API call: the model
calls the ``game_action`` tool, the bridge captures the action and aborts the
turn before the runtime can issue the usual post-tool follow-up request.
"""

from __future__ import annotations

import json
import os
import subprocess
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .config import BRIDGE_SCRIPT, RunConfig


class BridgeError(RuntimeError):
    """A bridge call failed; the caller may retry."""


class BridgeTimeout(BridgeError):
    """A bridge call exceeded its deadline; the process must be restarted."""


def dedupe_repeated_thinking(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    text = value.strip()
    for separator in ("\n\n", "\n"):
        remainder = len(text) - len(separator)
        if remainder < 0 or remainder % 2:
            continue
        midpoint = remainder // 2
        if (
            text[midpoint : midpoint + len(separator)] == separator
            and text[:midpoint] == text[midpoint + len(separator) :]
        ):
            return text[:midpoint] or None
    return text or None


@dataclass
class ModelCall:
    """One model turn: the chosen action plus telemetry about the call."""

    status: str
    action: str = ""
    raw_output: str = ""
    thinking: str | None = None
    model: str | None = None
    usage: list[dict[str, Any]] = field(default_factory=list)
    assistant_text: str = ""
    extra_tool_calls: int = 0
    duration: float = 0.0
    call_id: str | None = None

    @property
    def ok(self) -> bool:
        return self.status == "ok"

    def totals(self) -> dict[str, int | float]:
        keys = (
            "input_tokens",
            "output_tokens",
            "reasoning_tokens",
            "cache_read_tokens",
            "cache_write_tokens",
            "duration_ms",
            "cost",
            "aiu_nano",
        )
        totals = {key: 0 for key in keys}
        for usage in self.usage:
            for key in keys:
                totals[key] += usage.get(key) or 0
        totals["api_calls"] = len(self.usage)
        return totals


@dataclass
class TextModelCall:
    """One plain-text completion used by code-mode harnesses."""

    text: str
    thinking: str | None = None
    model: str | None = None
    reported_model: str | None = None
    usage: list[dict[str, Any]] = field(default_factory=list)
    duration: float = 0.0

    def totals(self) -> dict[str, int | float]:
        keys = (
            "input_tokens",
            "output_tokens",
            "reasoning_tokens",
            "cache_read_tokens",
            "cache_write_tokens",
            "duration_ms",
            "cost",
            "aiu_nano",
        )
        totals = {key: 0 for key in keys}
        for usage in self.usage:
            for key in keys:
                totals[key] += usage.get(key) or 0
        totals["api_calls"] = len(self.usage)
        return totals


class CopilotBridge:
    """Serial JSONL RPC client around one bridge subprocess."""

    def __init__(
        self,
        config: RunConfig,
        *,
        log_path: Path | None = None,
        working_directory: Path | None = None,
    ) -> None:
        if not config.sdk_path:
            raise BridgeError(
                "Copilot SDK not found; set COPILOT_SDK_PATH or pass --sdk-path."
            )
        self.config = config
        self.log_path = log_path
        self.working_directory = (
            Path(working_directory).expanduser().resolve()
            if working_directory is not None
            else Path(BRIDGE_SCRIPT).resolve().parents[1]
        )
        self._process: subprocess.Popen | None = None
        self._lock = threading.Lock()
        self._next_id = 0
        self._broken = False
        self._stderr_handle = None

    # -- Process lifecycle ------------------------------------------------

    def start(self, *, timeout: float = 120.0) -> None:
        if self._process is not None:
            return

        env = dict(os.environ)
        env["COPILOT_SDK_PATH"] = str(self.config.sdk_path)
        env["COPILOT_CLI_PATH"] = str(self.config.cli_path)
        env["COPILOT_HOME"] = str(self.config.copilot_home)
        self.working_directory.mkdir(parents=True, exist_ok=True)
        env["TALES_REPO_ROOT"] = str(self.working_directory)
        env["COPILOT_BRIDGE_LOG_LEVEL"] = self.config.bridge_log_level

        if self.log_path is not None:
            self.log_path.parent.mkdir(parents=True, exist_ok=True)
            self._stderr_handle = open(self.log_path, "a", encoding="utf-8")
            stderr = self._stderr_handle
        else:
            stderr = subprocess.DEVNULL

        self._process = subprocess.Popen(
            [self.config.node_bin, str(BRIDGE_SCRIPT)],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=stderr,
            env=env,
            text=True,
            bufsize=1,
        )

        try:
            message = self._read_message(timeout=timeout)
            if message.get("type") != "ready":
                raise BridgeError(f"Unexpected bridge handshake: {message!r}")
        except BaseException:
            self._broken = True
            self.stop(timeout=1.0)
            raise
        self._broken = False

    def stop(self, *, timeout: float = 30.0) -> None:
        process, self._process = self._process, None
        if process is None:
            return
        try:
            if process.poll() is None:
                try:
                    self._write({"id": -1, "command": "shutdown"}, process)
                    process.stdin.close()
                except (BrokenPipeError, ValueError, OSError):
                    pass
                try:
                    process.wait(timeout=timeout)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=10)
        finally:
            for stream in (process.stdin, process.stdout):
                try:
                    if stream is not None and not stream.closed:
                        stream.close()
                except OSError:
                    pass
            if self._stderr_handle is not None:
                self._stderr_handle.close()
                self._stderr_handle = None

    def restart(self) -> None:
        self.stop()
        self.start()

    # -- Low level RPC ----------------------------------------------------

    def _write(self, payload: dict[str, Any], process: subprocess.Popen) -> None:
        assert process.stdin is not None
        process.stdin.write(json.dumps(payload) + "\n")
        process.stdin.flush()

    def _read_message(self, *, timeout: float) -> dict[str, Any]:
        process = self._process
        if process is None or process.stdout is None:
            raise BridgeError("Bridge is not running.")

        result: dict[str, Any] = {}
        error: list[BaseException] = []

        def _reader() -> None:
            try:
                line = process.stdout.readline()
                if not line:
                    error.append(BridgeError("Bridge closed its output stream."))
                    return
                result.update(json.loads(line))
            except BaseException as exc:  # pragma: no cover - defensive
                error.append(exc)

        thread = threading.Thread(target=_reader, daemon=True)
        thread.start()
        thread.join(timeout)
        if thread.is_alive():
            raise BridgeTimeout(f"Bridge did not answer within {timeout:.0f}s.")
        if error:
            raise BridgeError(str(error[0]))
        return result

    def call(
        self, command: str, payload: dict[str, Any] | None = None, *, timeout: float
    ) -> dict[str, Any]:
        with self._lock:
            if self._broken:
                # A previous call timed out and left an orphan reader; the only
                # safe recovery is a fresh runtime.
                self.stop()
            if self._process is None:
                self.start()
            process = self._process
            assert process is not None
            self._next_id += 1
            request = {"id": self._next_id, "command": command, **(payload or {})}
            try:
                self._write(request, process)
            except (BrokenPipeError, ValueError, OSError) as exc:
                self._broken = True
                raise BridgeError(f"Bridge stdin unusable: {exc}") from exc

            deadline = time.monotonic() + timeout
            while True:
                remaining = max(deadline - time.monotonic(), 0.1)
                try:
                    message = self._read_message(timeout=remaining)
                except BridgeError:
                    self._broken = True
                    raise
                if message.get("type") == "ready":
                    continue
                if message.get("id") != request["id"]:
                    continue
                if not message.get("ok"):
                    error = message.get("error") or {}
                    raise BridgeError(
                        f"{error.get('name', 'Error')}: {error.get('message', message)}"
                    )
                return message.get("result") or {}

    # -- Commands ---------------------------------------------------------

    def ping(self, *, timeout: float = 30.0) -> bool:
        return bool(self.call("ping", timeout=timeout).get("pong"))

    def list_models(self, *, timeout: float = 120.0) -> list[dict[str, Any]]:
        return self.call("list_models", timeout=timeout).get("models", [])

    def create_session(
        self, *, model: str, reasoning_effort: str | None, session_id: str | None = None
    ) -> str:
        payload = {
            "model": model,
            "reasoning_effort": reasoning_effort,
            "reasoning_summary": self.config.reasoning_summary,
            "system_prompt": self._system_prompt(),
        }
        if session_id:
            payload["session_id"] = session_id
        result = self.call("create_session", payload, timeout=self.config.call_timeout)
        return result["session_id"]

    def create_text_session(
        self,
        *,
        model: str,
        reasoning_effort: str | None,
        system_prompt: str,
    ) -> str:
        """Create a no-tools, plain-text session for a code-mode agent."""
        result = self.call(
            "create_session",
            {
                "model": model,
                "reasoning_effort": reasoning_effort,
                "reasoning_summary": self.config.reasoning_summary,
                "system_prompt": system_prompt,
                "response_mode": "text",
            },
            timeout=self.config.call_timeout,
        )
        return result["session_id"]

    def resume_session(
        self, *, session_id: str, model: str, reasoning_effort: str | None
    ) -> str:
        payload = {
            "session_id": session_id,
            "model": model,
            "reasoning_effort": reasoning_effort,
            "reasoning_summary": self.config.reasoning_summary,
            "system_prompt": self._system_prompt(),
        }
        result = self.call("resume_session", payload, timeout=self.config.call_timeout)
        return result["session_id"]

    def close_session(self, session_id: str, *, delete: bool = True) -> None:
        self.call(
            "close_session",
            {"session_id": session_id, "delete": delete},
            timeout=60.0,
        )

    def send(self, *, session_id: str, prompt: str) -> ModelCall:
        started = time.monotonic()
        timeout_ms = int(self.config.call_timeout * 1000)
        result = self.call(
            "send",
            {
                "session_id": session_id,
                "prompt": prompt,
                "timeout_ms": timeout_ms,
            },
            timeout=self.config.call_timeout + 60.0,
        )
        return ModelCall(
            status=result.get("status", "ok"),
            action=result.get("action") or "",
            raw_output=result.get("raw_output") or "",
            thinking=dedupe_repeated_thinking(result.get("thinking")),
            model=result.get("model"),
            reported_model=result.get("reported_model"),
            usage=result.get("usage") or [],
            assistant_text=result.get("assistant_text") or "",
            extra_tool_calls=int(result.get("extra_tool_calls") or 0),
            duration=time.monotonic() - started,
        )

    def send_text(self, *, session_id: str, prompt: str) -> TextModelCall:
        """Send one prompt and return the model's exact plain-text response."""
        started = time.monotonic()
        timeout_ms = int(self.config.call_timeout * 1000)
        result = self.call(
            "send",
            {
                "session_id": session_id,
                "prompt": prompt,
                "timeout_ms": timeout_ms,
            },
            timeout=self.config.call_timeout + 60.0,
        )
        if result.get("status", "ok") != "ok":
            raise BridgeError(f"Text completion failed: {result!r}")
        return TextModelCall(
            text=result.get("text") or "",
            thinking=dedupe_repeated_thinking(result.get("thinking")),
            model=result.get("model"),
            usage=result.get("usage") or [],
            duration=time.monotonic() - started,
        )

    @staticmethod
    def _system_prompt() -> str:
        from .prompts import SYSTEM_PROMPT

        return SYSTEM_PROMPT

    def __enter__(self) -> "CopilotBridge":
        self.start()
        return self

    def __exit__(self, *exc: Any) -> None:
        self.stop()


def list_models(config: RunConfig) -> list[dict[str, Any]]:
    """One-shot helper used by ``list-models`` and ``plan``."""
    bridge = CopilotBridge(config)
    try:
        bridge.start()
        return bridge.list_models()
    finally:
        bridge.stop()
