"""Offline tests for the JAZ Jericho code-REPL harness."""

from __future__ import annotations

import copy
import io
import json
import os
import signal
import subprocess
import sys
from pathlib import Path

import jaz
import pytest
from jaz.exceptions import BudgetExhaustedError
from jaz.hooks import BudgetPool, RecursionLimit, TrajectoryRecorder, ValidateReturn
from jaz.llm import LLMResponse
from jaz.llm.copilot import _split_conversation as official_split_conversation
from jaz.protocol import CodeOnlyProtocol
from jaz.repl import PythonREPL

from copilot_evals import episode, storage
from copilot_evals.bridge import CopilotBridge, TextModelCall
from copilot_evals.config import RunConfig

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.jaz_jericho_harness import (
    ALLOWED_IMPORTS,
    ActionSafetyLimitExceededError,
    CompletionIntegrityError,
    CopilotJAZLLM,
    JAZRunConfig,
    JerichoGameController,
    OfficialCopilotJAZLLM,
    _write_artifact_manifests,
    aggregate_results,
    _is_budget_exhaustion,
    _is_context_budget_exhaustion,
    _originated_in_repl,
    _return_validator,
    split_copilot_conversation,
    trim_transcript_continuation,
)
from run_jaz_jericho import build_parser


class FakeGame:
    def __init__(self) -> None:
        self.history: list[str] = []
        self.reset_count = 0
        self.score = 0
        self.moves = 0

    def reset(self, *, seed: int):
        self.reset_count += 1
        self.score = 0
        self.moves = 0
        observation = f"Start {self.reset_count} (seed {seed})."
        return observation, self._info(observation)

    def step(self, action: str):
        self.history.append(action)
        self.moves += 1
        done = False
        won = False
        lost = False
        if action == "coin":
            self.score += 3
            observation = "Taken."
        elif action == "win":
            self.score = 10
            observation = "You have won."
            done = won = True
        elif action == "die":
            observation = "You have died."
            done = lost = True
        else:
            observation = f"Executed {action}."
        info = self._info(observation, won=won, lost=lost)
        return observation, 0, done, info

    def close(self) -> None:
        pass

    def _info(self, observation: str, *, won: bool = False, lost: bool = False) -> dict:
        return {
            "score": self.score,
            "moves": self.moves,
            "max_score": 10,
            "feedback": observation,
            "won": won,
            "lost": lost,
        }


class FakeTextBridge:
    def __init__(self, response: TextModelCall) -> None:
        self.response = response
        self.created: list[dict] = []
        self.sent: list[dict] = []
        self.closed: list[str] = []

    def create_text_session(self, **kwargs) -> str:
        self.created.append(kwargs)
        return f"session-{len(self.created)}"

    def send_text(self, **kwargs) -> TextModelCall:
        self.sent.append(kwargs)
        return self.response

    def close_session(self, session_id: str) -> None:
        self.closed.append(session_id)


class FakeOfficialBackend:
    def __init__(self, response: LLMResponse) -> None:
        self.response = response
        self.calls: list[tuple[str, list[dict]]] = []
        self.cli_paths: list[str | None] = []
        self.closed = False

    def complete(self, model: str, messages: list[dict]) -> LLMResponse:
        self.calls.append((model, messages))
        self.cli_paths.append(os.environ.get("COPILOT_CLI_PATH"))
        return self.response

    def close(self) -> None:
        self.closed = True


def make_controller(
    tmp_path: Path,
    env: FakeGame,
    *,
    action_safety_limit: int | None = 10,
):
    return JerichoGameController(
        env,
        game="FakeGame",
        seed=123,
        action_safety_limit=action_safety_limit,
        native_step_timeout=0,
        action_log_path=tmp_path / "actions.jsonl",
        model_calls_completed=lambda: 7,
    )


def test_conversation_flattening_and_continuation_trim():
    system, prompt, flattened = split_copilot_conversation(
        [
            {"role": "system", "content": "system rules"},
            {"role": "user", "content": "first"},
        ]
    )
    assert system == "system rules"
    assert prompt == "first"
    assert flattened is False

    system, prompt, flattened = split_copilot_conversation(
        [
            {"role": "system", "content": "system rules"},
            {"role": "user", "content": "first"},
            {"role": "assistant", "content": "print(1)"},
            {"role": "user", "content": "1"},
        ]
    )
    assert system == "system rules"
    assert prompt == ("[[user]]\nfirst\n\n[[assistant]]\nprint(1)\n\n[[user]]\n1")
    assert flattened is True
    assert trim_transcript_continuation(
        "print(step('north'))\n[[user]]\nfabricated output"
    ) == ("print(step('north'))", 27)
    valid_code = 'text = """\\n[[user]]\\n"""\nreturn text'
    assert trim_transcript_continuation(valid_code) == (valid_code, 0)
    assert split_copilot_conversation(
        [
            {"role": "system", "content": "system rules"},
            {"role": "user", "content": "first"},
            {"role": "assistant", "content": "print(1)"},
            {"role": "user", "content": "1"},
        ]
    ) == official_split_conversation(
        [
            {"role": "system", "content": "system rules"},
            {"role": "user", "content": "first"},
            {"role": "assistant", "content": "print(1)"},
            {"role": "user", "content": "1"},
        ]
    )


def test_run_config_defaults_to_llm_call_budget_only(tmp_path):
    config = JAZRunConfig(
        model="requested-model",
        game="FakeGame",
        seed=123,
        output_root=tmp_path,
        sdk_path="/tmp/sdk.js",
    )

    assert config.max_model_calls == 500
    assert config.action_safety_limit == 5000
    assert config.return_guard_scope == "root"
    assert config.transport == "custom"
    assert config.to_dict()["action_safety_limit"] == 5000
    assert config.to_dict()["allowed_read_paths"] == [config.workspace_glob]
    assert config.to_dict()["allowed_write_paths"] == [config.workspace_glob]


def test_cli_and_aggregate_report_safety_and_harness_modes(tmp_path):
    args = build_parser().parse_args([])
    assert args.max_model_calls == 500
    assert args.action_safety_limit == 5000
    assert args.return_guard_scope == "root"
    assert args.transport == "custom"

    configs = [
        JAZRunConfig(
            model="requested-model",
            game="FakeGame",
            seed=1,
            output_root=tmp_path,
            sdk_path="/tmp/sdk.js",
        ),
        JAZRunConfig(
            model="requested-model",
            game="FakeGame",
            seed=2,
            output_root=tmp_path,
            action_safety_limit=None,
            return_guard_scope="tree",
            transport="official",
            sdk_path="/tmp/sdk.js",
        ),
    ]
    aggregate = aggregate_results(configs, [])
    assert aggregate["settings"]["action_safety_limits"] == [5000, None]
    assert aggregate["settings"]["return_guard_scopes"] == ["root", "tree"]
    assert aggregate["settings"]["transports"] == ["custom", "official"]


def test_copilot_backend_preserves_text_usage_and_summary(tmp_path):
    bridge = FakeTextBridge(
        TextModelCall(
            text="print(step('north'))\n[[user]]\nfabricated",
            thinking="reasoning summary",
            model="requested-model",
            reported_model="requested-model",
            usage=[
                {
                    "input_tokens": 100,
                    "output_tokens": 20,
                    "reasoning_tokens": 12,
                    "cache_read_tokens": 30,
                    "cache_write_tokens": 4,
                    "duration_ms": 900,
                    "cost": 0,
                    "aiu_nano": 2_000_000_000,
                }
            ],
        )
    )
    telemetry_path = tmp_path / "model_calls.jsonl"
    backend = CopilotJAZLLM(
        bridge=bridge,
        model="requested-model",
        reasoning_effort="max",
        model_metadata={"max_prompt_tokens": 123_456},
        telemetry_path=telemetry_path,
        generated_code_dir=tmp_path / "generated-code",
        max_retries=0,
    )

    response = backend.complete(
        "requested-model",
        [
            {"role": "system", "content": "code only"},
            {"role": "user", "content": "start"},
            {"role": "assistant", "content": "print(1)"},
            {"role": "user", "content": "1"},
        ],
    )

    assert response.content == "print(step('north'))"
    assert response.prompt_tokens == 100
    assert response.completion_tokens == 20
    assert response.cached_tokens == 30
    assert response.cost_usd is None
    assert response.extra["reasoning_tokens"] == 12
    assert response.extra["reasoning_summary"] == "reasoning summary"
    assert response.extra["copilot_model"] == "requested-model"
    assert response.extra["copilot_ai_credits"] == 2
    assert backend.get_model_info() == {"max_input_tokens": 123_456}
    assert copy.deepcopy(backend) is backend
    assert bridge.created == [
        {
            "model": "requested-model",
            "reasoning_effort": "max",
            "system_prompt": "code only",
        }
    ]
    assert bridge.closed == ["session-1"]

    rows, damaged = storage.read_jsonl_prefix(telemetry_path)
    assert damaged is False
    assert len(rows) == 1
    assert rows[0]["response"] == "print(step('north'))"
    assert rows[0]["reasoning_is_summary"] is True
    assert rows[0]["dropped_continuation_chars"] > 0
    assert (
        tmp_path / "generated-code" / "call-0001.py"
    ).read_text() == "print(step('north'))\n"


def test_copilot_backend_retains_response_when_code_archival_fails(
    tmp_path, monkeypatch
):
    bridge = FakeTextBridge(
        TextModelCall(
            text="print(step('north'))",
            thinking="reasoning summary",
            model="requested-model",
            reported_model="requested-model",
            usage=[
                {
                    "input_tokens": 100,
                    "output_tokens": 20,
                    "reasoning_tokens": 12,
                    "cache_read_tokens": 30,
                    "cache_write_tokens": 4,
                    "duration_ms": 900,
                    "cost": 0,
                    "aiu_nano": 2_000_000_000,
                }
            ],
        )
    )
    generated_code_dir = tmp_path / "generated-code"
    telemetry_path = tmp_path / "model_calls.jsonl"
    backend = CopilotJAZLLM(
        bridge=bridge,
        model="requested-model",
        reasoning_effort="max",
        model_metadata={},
        telemetry_path=telemetry_path,
        generated_code_dir=generated_code_dir,
        max_retries=0,
    )
    harness_storage = sys.modules[CopilotJAZLLM.__module__].storage
    real_atomic_write_text = harness_storage.atomic_write_text
    failed = False

    def fail_first_code_archive(path, text):
        nonlocal failed
        if Path(path).name == "call-0001.py" and not failed:
            failed = True
            raise OSError("simulated code archive failure")
        return real_atomic_write_text(path, text)

    monkeypatch.setattr(
        harness_storage,
        "atomic_write_text",
        fail_first_code_archive,
    )

    with pytest.raises(OSError, match="simulated code archive failure"):
        backend.complete(
            "requested-model",
            [{"role": "user", "content": "start"}],
        )

    rows, damaged = storage.read_jsonl_prefix(telemetry_path)
    assert damaged is False
    assert len(rows) == 1
    assert rows[0]["status"] == "artifact_error"
    assert rows[0]["response"] == "print(step('north'))"
    fallback = Path(rows[0]["fallback_response_file"])
    assert fallback.read_text() == "print(step('north'))"
    assert backend.stats()["completion_attempts"] == 1
    assert backend.stats()["completed_calls"] == 0
    assert backend.stats()["provider_api_calls"] == 1


def test_official_copilot_backend_records_comparable_usage(tmp_path, monkeypatch):
    monkeypatch.setenv("COPILOT_CLI_PATH", "/previous/copilot")
    inner = FakeOfficialBackend(
        LLMResponse(
            content='print("ok")',
            prompt_tokens=100,
            completion_tokens=25,
            cached_tokens=40,
            cost_usd=0.06,
            extra={
                "reasoning_tokens": 20,
                "cache_creation_input_tokens": 5,
                "copilot_ai_credits": 6.0,
                "copilot_model_calls": 2,
                "copilot_model": "requested-model",
            },
        )
    )
    telemetry_path = tmp_path / "model_calls.jsonl"
    backend = OfficialCopilotJAZLLM(
        model="requested-model",
        reasoning_effort="max",
        model_metadata={"max_prompt_tokens": 123_456},
        telemetry_path=telemetry_path,
        generated_code_dir=tmp_path / "generated-code",
        base_directory=tmp_path / "official",
        cli_path="/usr/local/bin/copilot",
        call_timeout=30,
        max_retries=0,
        backend=inner,
    )

    response = backend.complete(
        "requested-model",
        [{"role": "user", "content": "test"}],
    )
    backend.close()

    assert response.content == 'print("ok")'
    assert inner.cli_paths == ["/usr/local/bin/copilot"]
    assert os.environ["COPILOT_CLI_PATH"] == "/previous/copilot"
    assert backend.get_model_info() == {"max_input_tokens": 123_456}
    assert backend.stats() == {
        "completion_attempts": 1,
        "completed_calls": 1,
        "provider_api_calls": 2,
        "prompt_tokens": 100,
        "completion_tokens": 25,
        "reasoning_tokens": 20,
        "cached_tokens": 40,
        "cache_write_tokens": 5,
        "aiu_nano": 6_000_000_000,
        "ai_credits": 6.0,
    }
    assert inner.closed is True
    rows, damaged = storage.read_jsonl_prefix(telemetry_path)
    assert damaged is False
    assert rows[0]["transport"] == "official"
    assert rows[0]["totals"]["api_calls"] == 2
    assert (tmp_path / "generated-code" / "call-0001.py").read_text() == (
        'print("ok")\n'
    )


def test_copilot_backend_accepts_usage_model_when_response_model_is_absent(
    tmp_path,
):
    bridge = FakeTextBridge(
        TextModelCall(
            text="return 1",
            model="requested-model",
            reported_model=None,
            usage=[
                {
                    "model": "requested-model",
                    "input_tokens": 10,
                    "output_tokens": 2,
                    "aiu_nano": 500_000_000,
                }
            ],
        )
    )
    telemetry_path = tmp_path / "model_calls.jsonl"
    backend = CopilotJAZLLM(
        bridge=bridge,
        model="requested-model",
        reasoning_effort="max",
        model_metadata={},
        telemetry_path=telemetry_path,
        max_retries=0,
    )

    response = backend.complete(
        "requested-model",
        [{"role": "user", "content": "test"}],
    )

    assert response.content == "return 1"
    assert response.extra["copilot_model"] == "requested-model"
    assert backend.stats()["completed_calls"] == 1
    rows, damaged = storage.read_jsonl_prefix(telemetry_path)
    assert damaged is False
    assert rows[0]["reported_model"] is None
    assert rows[0]["served_model"] == "requested-model"


def test_copilot_backend_accepts_provider_length_continuation(tmp_path):
    bridge = FakeTextBridge(
        TextModelCall(
            text="return 1",
            model="requested-model",
            reported_model=None,
            usage=[
                {
                    "model": "requested-model",
                    "input_tokens": 100,
                    "output_tokens": 32_000,
                    "reasoning_tokens": 32_000,
                    "finish_reason": "length",
                    "aiu_nano": 10_000_000_000,
                },
                {
                    "model": "requested-model",
                    "input_tokens": 32_100,
                    "output_tokens": 20,
                    "reasoning_tokens": 10,
                    "finish_reason": "stop",
                    "aiu_nano": 1_000_000_000,
                },
            ],
        )
    )
    backend = CopilotJAZLLM(
        bridge=bridge,
        model="requested-model",
        reasoning_effort="max",
        model_metadata={},
        telemetry_path=tmp_path / "model_calls.jsonl",
        max_retries=0,
    )

    response = backend.complete(
        "requested-model",
        [{"role": "user", "content": "test"}],
    )

    assert response.content == "return 1"
    assert response.extra["copilot_model_calls"] == 2
    assert backend.stats()["completed_calls"] == 1
    assert backend.stats()["provider_api_calls"] == 2


def test_copilot_backend_turns_content_filter_into_repl_feedback(tmp_path):
    bridge = FakeTextBridge(
        TextModelCall(
            text=(
                "The model returned no content because the response was blocked "
                "by content filtering."
            ),
            model="requested-model",
            reported_model=None,
            usage=[
                {
                    "model": "requested-model",
                    "input_tokens": 100,
                    "output_tokens": 32_000,
                    "reasoning_tokens": 32_000,
                    "finish_reason": "length",
                    "aiu_nano": 10_000_000_000,
                },
                {
                    "model": "requested-model",
                    "input_tokens": 32_100,
                    "output_tokens": 0,
                    "reasoning_tokens": 0,
                    "finish_reason": "content_filter",
                    "content_filter_triggered": True,
                    "aiu_nano": 1_000_000_000,
                },
            ],
        )
    )
    telemetry_path = tmp_path / "model_calls.jsonl"
    backend = CopilotJAZLLM(
        bridge=bridge,
        model="requested-model",
        reasoning_effort="max",
        model_metadata={},
        telemetry_path=telemetry_path,
        max_retries=0,
    )

    response = backend.complete(
        "requested-model",
        [{"role": "user", "content": "test"}],
    )

    assert response.content.startswith('print("Provider content filtering blocked')
    assert response.extra["copilot_content_filter_triggered"] is True
    assert response.extra["copilot_model_calls"] == 2
    assert backend.stats()["completed_calls"] == 1
    rows, damaged = storage.read_jsonl_prefix(telemetry_path)
    assert damaged is False
    assert rows[0]["response_transformed"] is True
    assert rows[0]["response_transform_reason"] == "content_filter"
    assert rows[0]["raw_response"].startswith("The model returned no content")


def test_copilot_backend_turns_output_limit_into_repl_feedback(tmp_path):
    bridge = FakeTextBridge(
        TextModelCall(
            text="In [ ]:\n\nIn [ ]:",
            model="requested-model",
            reported_model=None,
            usage=[
                {
                    "model": "requested-model",
                    "input_tokens": 100,
                    "output_tokens": 32_000,
                    "reasoning_tokens": 32_000,
                    "finish_reason": "length",
                    "aiu_nano": 10_000_000_000,
                },
                {
                    "model": "requested-model",
                    "input_tokens": 32_100,
                    "output_tokens": 32_000,
                    "reasoning_tokens": 100,
                    "finish_reason": "length",
                    "aiu_nano": 2_000_000_000,
                },
            ],
        )
    )
    telemetry_path = tmp_path / "model_calls.jsonl"
    backend = CopilotJAZLLM(
        bridge=bridge,
        model="requested-model",
        reasoning_effort="max",
        model_metadata={},
        telemetry_path=telemetry_path,
        max_retries=0,
    )

    response = backend.complete(
        "requested-model",
        [{"role": "user", "content": "test"}],
    )

    assert response.content.startswith('print("The provider exhausted its output')
    assert response.extra["copilot_output_limit_exhausted"] is True
    assert response.extra["copilot_model_calls"] == 2
    assert backend.stats()["completed_calls"] == 1
    rows, damaged = storage.read_jsonl_prefix(telemetry_path)
    assert damaged is False
    assert rows[0]["response_transform_reason"] == "output_limit"
    assert rows[0]["raw_response"] == "In [ ]:\n\nIn [ ]:"


@pytest.mark.parametrize(
    "call",
    (
        TextModelCall(
            text="return 1",
            model="other-model",
            reported_model="other-model",
            usage=[{"model": "other-model"}],
        ),
        TextModelCall(
            text="return 1",
            model="requested-model",
            reported_model="requested-model",
            usage=[],
        ),
        TextModelCall(
            text="return 1",
            model="requested-model",
            reported_model="requested-model",
            usage=[
                {
                    "model": "requested-model",
                    "finish_reason": "stop",
                },
                {
                    "model": "requested-model",
                    "finish_reason": "stop",
                },
            ],
        ),
    ),
)
def test_copilot_backend_rejects_integrity_mismatches(tmp_path, call):
    bridge = FakeTextBridge(call)
    backend = CopilotJAZLLM(
        bridge=bridge,
        model="requested-model",
        reasoning_effort="max",
        model_metadata={},
        telemetry_path=tmp_path / "model_calls.jsonl",
        max_retries=0,
    )

    with pytest.raises(CompletionIntegrityError):
        backend.complete(
            "requested-model",
            [{"role": "user", "content": "test"}],
        )

    assert bridge.closed == ["session-1"]
    stats = backend.stats()
    assert stats["completed_calls"] == 0
    assert stats["provider_api_calls"] == len(call.usage)


def test_controller_counts_rejected_actions_without_stepping(tmp_path):
    env = FakeGame()
    controller = make_controller(tmp_path, env, action_safety_limit=2)

    invalid = controller.step("north\nsouth")
    assert invalid.action_was_executed is False
    assert invalid.invalid_reason
    assert env.history == []
    assert invalid.actions_remaining is None
    assert invalid.action_safety_remaining == 1

    blocked = controller.step("save")
    assert blocked.action_was_executed is False
    assert blocked.blocked_command == "save"
    assert blocked.run_complete is True
    assert blocked.completion_reason == "action_safety_limit"
    assert blocked.action_safety_remaining == 0
    assert env.history == []

    with pytest.raises(ActionSafetyLimitExceededError):
        controller.step("north")

    rows, damaged = storage.read_jsonl_prefix(tmp_path / "actions.jsonl")
    assert damaged is False
    assert [row["action_index"] for row in rows] == [1, 2]
    assert all(row["model_calls_completed"] == 7 for row in rows)


def test_controller_can_disable_operational_action_guard(tmp_path):
    env = FakeGame()
    controller = make_controller(tmp_path, env, action_safety_limit=None)

    for _ in range(20):
        state = controller.step("look")

    assert state.actions_used == 20
    assert state.actions_remaining is None
    assert state.action_safety_remaining is None
    assert state.run_complete is False
    assert controller.summary()["action_safety_limit"] is None
    assert controller.summary()["action_safety_limit_hit"] is False


def test_controller_translates_explicit_backspace_marker(tmp_path):
    env = FakeGame()
    controller = make_controller(tmp_path, env, action_safety_limit=2)

    state = controller.step("<BACKSPACE>")

    assert state.submitted_action == "<BACKSPACE>"
    assert state.executed_action == "\x08"
    assert state.action_was_executed is True
    assert env.history == ["\x08"]


def test_blocked_state_command_takes_precedence_over_compound_rejection(tmp_path):
    env = FakeGame()
    controller = make_controller(tmp_path, env, action_safety_limit=1)

    blocked = controller.step("north then save")

    assert blocked.blocked_command == "save"
    assert blocked.invalid_reason is None
    assert env.history == []


def test_controller_counts_non_string_step_calls(tmp_path):
    env = FakeGame()
    controller = make_controller(tmp_path, env, action_safety_limit=1)

    rejected = controller.step(123)  # type: ignore[arg-type]

    assert rejected.submitted_action == "123"
    assert rejected.invalid_reason == "Action must be a string."
    assert rejected.completion_reason == "action_safety_limit"
    assert env.history == []


@pytest.mark.parametrize("action", ("north. east", "north then east", "north AND east"))
def test_controller_rejects_compound_parser_commands(tmp_path, action):
    env = FakeGame()
    controller = make_controller(tmp_path, env, action_safety_limit=1)

    rejected = controller.step(action)

    assert rejected.action_was_executed is False
    assert "Compound parser commands" in rejected.invalid_reason
    assert rejected.completion_reason == "action_safety_limit"
    assert env.history == []


def test_budget_exception_groups_are_recognized():
    grouped = ExceptionGroup(
        "budget hooks",
        [
            BudgetExhaustedError("pool"),
            ExceptionGroup("nested", [BudgetExhaustedError("iterations")]),
        ],
    )
    mixed = ExceptionGroup(
        "mixed",
        [BudgetExhaustedError("pool"), ValueError("real failure")],
    )

    assert _is_budget_exhaustion(grouped) is True
    assert _is_budget_exhaustion(mixed) is False


def test_context_budget_exception_groups_are_recognized():
    grouped = ExceptionGroup(
        "context hooks",
        [
            RuntimeError(
                "Request exceeds the model's prompt budget after reserving output"
            ),
            ExceptionGroup(
                "nested",
                [RuntimeError("Maximum context length exceeded")],
            ),
        ],
    )
    mixed = ExceptionGroup(
        "mixed",
        [
            RuntimeError("Context window exceeded"),
            ValueError("real failure"),
        ],
    )

    assert _is_context_budget_exhaustion(grouped) is True
    assert _is_context_budget_exhaustion(mixed) is False
    assert _is_context_budget_exhaustion(RuntimeError("Prompt is too long")) is True


def test_repl_originated_exceptions_are_distinguished_from_harness_failures():
    try:
        exec(compile("raise RuntimeError('impossible')", "<repl_test_1>", "exec"))
    except RuntimeError as exc:
        repl_exception = exc

    assert _originated_in_repl(repl_exception) is True
    assert _originated_in_repl(RuntimeError("infrastructure")) is False


def test_controller_resets_after_loss_and_stops_at_max_score(tmp_path):
    env = FakeGame()
    controller = make_controller(tmp_path, env, action_safety_limit=5)

    coin = controller.step("coin")
    assert coin.score == coin.highscore == 3
    death = controller.step("die")
    assert death.episode_lost is True
    assert death.restarted is True
    assert env.reset_count == 2
    assert death.score == 0
    assert death.moves == 0
    assert death.highscore == 3
    assert "-= Restarting =-" in death.observation

    win = controller.step("win")
    assert win.episode_won is True
    assert win.restarted is False
    assert win.run_complete is True
    assert win.completion_reason == "max_score"
    assert win.highscore == 10
    assert controller.summary()["resets"] == 1


def test_controller_logs_environment_step_errors(tmp_path):
    class BrokenGame(FakeGame):
        def step(self, action: str):
            raise RuntimeError(f"native failure for {action}")

    controller = make_controller(
        tmp_path,
        BrokenGame(),
        action_safety_limit=2,
    )

    with pytest.raises(RuntimeError, match="native failure"):
        controller.step("north")

    summary = controller.summary()
    assert summary["actions_used"] == 1
    assert summary["actions_executed"] == 1
    assert summary["environment_errors"] == 1
    rows, damaged = storage.read_jsonl_prefix(tmp_path / "actions.jsonl")
    assert damaged is False
    assert rows[0]["error_type"] == "RuntimeError"
    assert rows[0]["executed_action"] == "north"


def test_jaz_repl_is_persistent_and_denies_hidden_or_external_access(tmp_path):
    env = FakeGame()
    controller = make_controller(tmp_path, env)
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    responses = iter(
        [
            "import socket",
            "print(open('/etc/passwd').read())",
            (
                "with open(workspace_dir + '/notes.py', 'w') as handle:\n"
                "    handle.write('def helper():\\n    return 42\\n')\n"
                "print(open(workspace_dir + '/notes.py').read())"
            ),
            "print(step.__self__)",
            "counter = 41\nprint(step('coin'))",
            "counter += 1\nprint(step('win'))",
            "return counter",
        ]
    )
    llm = jaz.MockLLMClient(fn=lambda *_args, **_kwargs: next(responses))
    repl = PythonREPL(
        exec_timeout=5,
        allow_timeout_pragma=False,
        allowed_imports=list(ALLOWED_IMPORTS),
        allowed_read_paths=[f"/{workspace.as_posix()}/**"],
        allowed_write_paths=[f"/{workspace.as_posix()}/**"],
        traceback_verbosity="repl_only",
    )
    trace_path = tmp_path / "trace.json"

    with (
        jaz.ConfigOverride(
            llm=llm,
            repl=repl,
            protocol=CodeOnlyProtocol(),
        ),
        jaz.scope(
            step=controller.step,
            observe=controller.observe,
            workspace_dir=str(workspace),
        ),
        TrajectoryRecorder(trace_path),
        BudgetPool(calls_budget=10),
        ValidateReturn(lambda value: _return_validator(controller, value)),
    ):
        result = jaz.invoke(instructions="Win the fake game.")

    assert result == 42
    assert env.history == ["coin", "win"]
    assert controller.summary()["completion_reason"] == "max_score"
    trace = trace_path.read_text()
    assert "socket" in trace
    assert "/etc/passwd" in trace
    assert "__self__" in trace
    assert "SandboxPermissionError" in trace or "not allowed" in trace
    assert (workspace / "notes.py").read_text() == ("def helper():\n    return 42\n")


def test_artifact_manifests_hash_files_and_index_python_symbols(tmp_path):
    workspace = tmp_path / "workspace"
    generated = tmp_path / "generated-code"
    workspace.mkdir()
    generated.mkdir()
    (workspace / "notes.py").write_text("def route():\n    return ['north']\n")
    (generated / "call-0001.py").write_text(
        "class Plan:\n    pass\n\ndef act():\n    return 1\n"
    )

    workspace_manifest, generated_manifest = _write_artifact_manifests(
        tmp_path,
        workspace,
        generated,
    )

    workspace_payload = json.loads(workspace_manifest.read_text())
    generated_payload = json.loads(generated_manifest.read_text())
    assert workspace_payload["files"][0]["python_symbols"] == [
        {"kind": "function", "name": "route", "line": 1}
    ]
    assert generated_payload["files"][0]["python_symbols"] == [
        {"kind": "class", "name": "Plan", "line": 1},
        {"kind": "function", "name": "act", "line": 4},
    ]


def test_root_return_guard_allows_bounded_subinvoke_to_return(tmp_path):
    env = FakeGame()
    controller = make_controller(tmp_path, env)
    responses = iter(
        [
            (
                'plan = invoke(task="analyze the puzzle")\n'
                "print(plan)\n"
                "print(step('win'))"
            ),
            'return "bounded plan"',
            "return plan",
        ]
    )
    llm = jaz.MockLLMClient(fn=lambda *_args, **_kwargs: next(responses))
    guard = ValidateReturn(lambda value: _return_validator(controller, value))

    with (
        jaz.ConfigOverride(
            llm=llm,
            repl=PythonREPL(exec_timeout=5, allow_timeout_pragma=False),
            protocol=CodeOnlyProtocol(),
        ),
        jaz.scope(step=controller.step, observe=controller.observe),
        BudgetPool(calls_budget=10),
        RecursionLimit(max_depth=3),
    ):
        result = jaz.invoke(guard, instructions="Win the fake game.")

    assert result == "bounded plan"
    assert env.history == ["win"]
    assert controller.is_complete()


def test_tree_return_guard_propagates_through_recursive_invoke(tmp_path):
    env = FakeGame()
    controller = make_controller(tmp_path, env)
    responses = iter(
        [
            'return invoke(task="finish the game")',
            'return "premature"',
            "print(step('win'))",
            'return "done"',
        ]
    )
    llm = jaz.MockLLMClient(fn=lambda *_args, **_kwargs: next(responses))

    with (
        jaz.ConfigOverride(
            llm=llm,
            repl=PythonREPL(exec_timeout=5, allow_timeout_pragma=False),
            protocol=CodeOnlyProtocol(),
        ),
        jaz.scope(step=controller.step, observe=controller.observe),
        BudgetPool(calls_budget=10),
        RecursionLimit(max_depth=3),
        ValidateReturn(lambda value: _return_validator(controller, value)),
    ):
        result = jaz.invoke(instructions="Win the fake game.")

    assert result == "done"
    assert env.history == ["win"]
    assert controller.is_complete()


def test_native_step_deadline_restores_enclosing_alarm(monkeypatch):
    handlers: list[object] = []
    timers: list[tuple[float, float]] = []
    monotonic = iter([100.0, 102.0])

    monkeypatch.setattr(signal, "getsignal", lambda _sig: "outer-handler")
    monkeypatch.setattr(signal, "getitimer", lambda _which: (100.0, 0.0))
    monkeypatch.setattr(
        signal,
        "signal",
        lambda _sig, handler: handlers.append(handler),
    )
    monkeypatch.setattr(
        signal,
        "setitimer",
        lambda _which, delay, interval=0.0: timers.append((delay, interval)),
    )
    monkeypatch.setattr(episode.time, "monotonic", lambda: next(monotonic))

    with episode._native_step_deadline(30):
        pass

    assert handlers == [signal.SIG_DFL, "outer-handler"]
    assert timers == [(30.0, 0.0), (0, 0.0), (98.0, 0.0)]


def test_javascript_bridge_text_mode_has_no_tools_or_turn_abort():
    bridge_path = Path(__file__).resolve().parents[1] / "copilot_model_bridge.mjs"
    source = bridge_path.read_text()
    config_start = source.index("function sessionConfig(")
    config_end = source.index("\n\nfunction registerSession(", config_start)
    register_start = config_end + 2
    register_end = source.index("\n\nasync function createSession(", register_start)
    script = f"""
const sessions = new Map();
function gameActionTool() {{ return {{ name: "game_action" }}; }}
function sanitizeUsage(data) {{ return data; }}
{source[config_start:config_end]}
{source[register_start:register_end]}

const toolConfig = sessionConfig({{
  model: "m",
  system_prompt: "s",
}}, false, {{}});
const textConfig = sessionConfig({{
  model: "m",
  system_prompt: "s",
  response_mode: "text",
}}, false, {{}});

function fakeSession(id) {{
  return {{
    sessionId: id,
    callbacks: [],
    aborts: 0,
    on(callback) {{ this.callbacks.push(callback); }},
    abort() {{ this.aborts += 1; return Promise.resolve(); }},
  }};
}}
const toolSession = fakeSession("tool");
const textSession = fakeSession("text");
registerSession({{ events: [], toolCalls: [], responseMode: "tool" }}, toolSession);
registerSession({{ events: [], toolCalls: [], responseMode: "text" }}, textSession);
toolSession.callbacks[0]({{ type: "assistant.turn_end", data: {{ turnId: "1" }} }});
textSession.callbacks[0]({{ type: "assistant.turn_end", data: {{ turnId: "2" }} }});

console.log(JSON.stringify({{
  toolAllowedModels: toolConfig.allowedModels,
  textAllowedModels: textConfig.allowedModels,
  toolNames: toolConfig.availableTools,
  toolCount: toolConfig.tools.length,
  textNames: textConfig.availableTools,
  textCount: textConfig.tools.length,
  memory: textConfig.memory,
  infiniteSessions: textConfig.infiniteSessions,
  toolAborts: toolSession.aborts,
  textAborts: textSession.aborts,
}}));
"""
    result = subprocess.run(
        ["node", "--input-type=module", "--eval", script],
        check=True,
        capture_output=True,
        text=True,
    )
    payload = json.loads(result.stdout)

    assert payload == {
        "toolAllowedModels": ["m"],
        "textAllowedModels": ["m"],
        "toolNames": ["game_action"],
        "toolCount": 1,
        "textNames": [],
        "textCount": 0,
        "memory": {"enabled": False},
        "infiniteSessions": {"enabled": False},
        "toolAborts": 1,
        "textAborts": 0,
    }


def test_javascript_bridge_enforces_exact_model_policy():
    bridge_path = Path(__file__).resolve().parents[1] / "copilot_model_bridge.mjs"
    source = bridge_path.read_text()
    start = source.index("async function enforceRequestedModel(")
    end = source.index("\n\nasync function createSession(", start)
    enforce_function = source[start:end]
    script = f"""
{enforce_function}
const calls = [];
const session = {{
  rpc: {{
    model: {{
      async setAllowedModels(params) {{
        calls.push(params);
        return {{
          allowedModels: params.allowedModels,
          effectiveAllowedModels: params.allowedModels,
          fallbackModel: "claude-opus-5.5",
          modelId: "claude-opus-5.5",
        }};
      }},
    }},
  }},
}};
const policy = await enforceRequestedModel(session, "claude-opus-5.5");
let mismatch;
try {{
  await enforceRequestedModel({{
    rpc: {{
      model: {{
        async setAllowedModels() {{
          return {{
            fallbackModel: "claude-opus-4.8",
            modelId: "claude-opus-5.5",
          }};
        }},
      }},
    }},
  }}, "claude-opus-5.5");
}} catch (error) {{
  mismatch = error.message;
}}
console.log(JSON.stringify({{ calls, policy, mismatch }}));
"""
    result = subprocess.run(
        ["node", "--input-type=module", "--eval", script],
        check=True,
        capture_output=True,
        text=True,
    )
    payload = json.loads(result.stdout)

    assert payload["calls"] == [{"allowedModels": ["claude-opus-5.5"]}]
    assert payload["policy"]["fallbackModel"] == "claude-opus-5.5"
    assert "fallbackModel=claude-opus-4.8" in payload["mismatch"]


def test_bridge_keeps_auth_home_separate_from_working_directory(tmp_path, monkeypatch):
    captured = {}

    class FakeProcess:
        def __init__(self):
            self.stdin = io.StringIO()
            self.stdout = io.StringIO('{"type":"ready"}\n')

        def poll(self):
            return 0

    def fake_popen(*args, **kwargs):
        captured["args"] = args
        captured["kwargs"] = kwargs
        return FakeProcess()

    monkeypatch.setattr("copilot_evals.bridge.subprocess.Popen", fake_popen)
    auth_home = tmp_path / "auth"
    work_dir = tmp_path / "isolated-work"
    bridge = CopilotBridge(
        RunConfig(
            output_root=tmp_path / "output",
            sdk_path="/fake/sdk/index.js",
            cli_path="/fake/copilot",
            copilot_home=str(auth_home),
        ),
        working_directory=work_dir,
    )

    bridge.start()
    bridge.stop()

    env = captured["kwargs"]["env"]
    assert env["COPILOT_HOME"] == str(auth_home)
    assert env["TALES_REPO_ROOT"] == str(work_dir.resolve())
    assert work_dir.is_dir()
