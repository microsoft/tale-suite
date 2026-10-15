"""Hugging Face trajectory schema (``PEARLS-Lab/TALES-Trajectories`` compatible).

The exported rows carry exactly :data:`HF_COLUMNS`, in that order. ``turns`` and
``score_progression`` are JSON strings, ``turns`` holding ``2 * total_steps + 1``
entries: the initial observation followed by alternating agent/environment
turns.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field
from typing import Any, Iterable, Sequence

HF_COLUMNS: tuple[str, ...] = (
    "transcript_id",
    "framework",
    "game",
    "model",
    "agent_type",
    "total_steps",
    "score",
    "max_score",
    "normalized_score",
    "turns",
    "score_progression",
    "thinking_available",
    "total_thinking_tokens",
    "llm",
    "seed",
    "reasoning_effort",
    "max_steps",
    "tags",
    "episode_token_usage",
    "model_family",
)

_COLUMN_TYPES: dict[str, type | tuple[type, ...]] = {
    "transcript_id": str,
    "framework": str,
    "game": str,
    "model": str,
    "agent_type": str,
    "total_steps": int,
    "score": int,
    "max_score": int,
    "normalized_score": float,
    "turns": str,
    "score_progression": str,
    "thinking_available": bool,
    "total_thinking_tokens": int,
    "llm": str,
    "seed": int,
    "reasoning_effort": str,
    "max_steps": int,
    "tags": list,
    "episode_token_usage": float,
    "model_family": str,
}


class SchemaError(ValueError):
    """Raised when a trajectory record does not match the HF schema."""


MAX_GAME_ACTION_BYTES = 198
ENTER_GAME_ACTION = "<ENTER>"
BACKSPACE_GAME_ACTION = "<BACKSPACE>"


def normalize_game_action(action: Any) -> str | None:
    if not isinstance(action, str):
        return None
    if not action.isprintable():
        return None
    if action and all(character == " " for character in action):
        return " "
    normalized = action.strip()
    if not normalized:
        return None
    if normalized.casefold() == "<space>":
        return " "
    if normalized.casefold() in {"<enter>", "<return>"}:
        return ENTER_GAME_ACTION
    if normalized.casefold() == "<backspace>":
        return BACKSPACE_GAME_ACTION
    if normalized.startswith("\\"):
        return None
    try:
        encoded = normalized.encode("utf-8")
    except UnicodeEncodeError:
        return None
    if len(encoded) > MAX_GAME_ACTION_BYTES:
        return None
    return normalized


def native_game_action(normalized_action: str) -> str:
    if normalized_action == ENTER_GAME_ACTION:
        return ""
    if normalized_action == BACKSPACE_GAME_ACTION:
        return "\x08"
    return normalized_action


def valid_game_action(action: Any) -> bool:
    return normalize_game_action(action) is not None


def normalized(score: int, max_score: int) -> float:
    return float(score) / float(max_score) if max_score else 0.0


def compat_token_usage(usage: dict[str, Any] | None) -> int:
    """Per-turn token usage, matching the existing TALES agent accounting.

    ``nb_tokens = prompt + response + thinking`` (reasoning tokens are added
    even though some providers already fold them into the output count).
    """
    if not usage:
        return 0
    return (
        int(usage.get("input_tokens") or 0)
        + int(usage.get("output_tokens") or 0)
        + int(usage.get("reasoning_tokens") or 0)
    )


def transcript_id(
    game: str, model: str, agent_type: str, effort_label: str, seed: int
) -> str:
    """Deterministic transcript id, e.g. ``JerichoEnv905-gpt-5.5_react-xhigh_s202411061``."""
    from .models import slugify

    return f"{game}-{slugify(model)}_{agent_type}-{effort_label}_s{int(seed)}"


@dataclass
class StepRecord:
    """One agent/environment exchange, persisted to the resumable steps log."""

    step: int
    action: str
    raw_output: str
    thinking: str | None
    token_usage: int
    thinking_tokens: int
    score_before: int
    score_after: int
    max_score: int
    feedback: str
    observation: str = ""
    moves: int = 0
    done: bool = False
    won: bool = False
    lost: bool = False
    did_reset: bool = False
    session_id: str | None = None
    usage: dict[str, Any] = field(default_factory=dict)
    call: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "StepRecord":
        known = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in data.items() if k in known})


def initial_turn(observation: str, score: int, max_score: int) -> dict[str, Any]:
    return {
        "role": "environment",
        "content": observation,
        "turn_index": 0,
        "step": 1,
        "score_at_step": int(score),
        "max_score": int(max_score),
        "normalized_score_at_step": normalized(score, max_score),
        "is_initial_observation": True,
    }


def agent_turn(
    step: StepRecord,
    turn_index: int,
    max_score: int | None = None,
) -> dict[str, Any]:
    effective_max_score = int(step.max_score if max_score is None else max_score)
    return {
        "role": "agent",
        "content": step.action,
        "raw_output": step.raw_output,
        "thinking": step.thinking,
        "token_usage": int(step.token_usage),
        "turn_index": turn_index,
        "step": int(step.step),
        "score_at_step": int(step.score_after),
        "max_score": effective_max_score,
        "normalized_score_at_step": normalized(
            step.score_after,
            effective_max_score,
        ),
    }


def feedback_turn(step: StepRecord, turn_index: int) -> dict[str, Any]:
    return {
        "role": "environment",
        "content": step.feedback,
        "turn_index": turn_index,
        "step": int(step.step),
        "score_at_step": int(step.score_after),
        "is_feedback": True,
    }


def build_turns(
    initial_observation: str, steps: Sequence[StepRecord], max_score: int
) -> list[dict[str, Any]]:
    initial_score = steps[0].score_before if steps else 0
    turns = [initial_turn(initial_observation, initial_score, max_score)]
    for index, step in enumerate(steps):
        turns.append(agent_turn(step, 2 * index + 1, max_score))
        turns.append(feedback_turn(step, 2 * index + 2))
    return turns


def build_score_progression(steps: Sequence[StepRecord]) -> list[dict[str, int]]:
    """``[{"step": 1, "score": 0}, ...]`` with an entry per score change."""
    if not steps:
        return [{"step": 1, "score": 0}]

    progression: list[dict[str, int]] = []
    for step in steps:
        if not progression or int(step.score_after) != progression[-1]["score"]:
            progression.append({"step": int(step.step), "score": int(step.score_after)})
    return progression


def build_trajectory(
    *,
    game: str,
    model: str,
    seed: int,
    max_steps: int,
    max_score: int,
    initial_observation: str,
    steps: Sequence[StepRecord],
    reasoning_effort: str,
    model_family: str,
    agent_type: str = "react",
    framework: str = "jericho",
    llm: str | None = None,
    tags: Iterable[Any] | None = None,
) -> dict[str, Any]:
    """Assemble one HF-compatible trajectory row."""
    steps = list(steps)
    turns = build_turns(initial_observation, steps, max_score)
    progression = build_score_progression(steps)
    trajectory_score = (
        max(
            int(steps[0].score_before),
            *(int(step.score_after) for step in steps),
        )
        if steps
        else 0
    )
    thinking_tokens = sum(int(step.thinking_tokens or 0) for step in steps)
    thinking_available = thinking_tokens > 0 or any(step.thinking for step in steps)
    episode_token_usage = float(steps[-1].token_usage) if steps else 0.0

    record = {
        "transcript_id": transcript_id(game, model, agent_type, reasoning_effort, seed),
        "framework": framework,
        "game": game,
        "model": model,
        "agent_type": agent_type,
        "total_steps": len(steps),
        "score": trajectory_score,
        "max_score": int(max_score),
        "normalized_score": normalized(trajectory_score, max_score),
        "turns": json.dumps(turns),
        "score_progression": json.dumps(progression),
        "thinking_available": bool(thinking_available),
        "total_thinking_tokens": int(thinking_tokens),
        "llm": llm or model,
        "seed": int(seed),
        "reasoning_effort": reasoning_effort,
        "max_steps": int(max_steps),
        "tags": list(tags or []),
        "episode_token_usage": episode_token_usage,
        "model_family": model_family,
    }
    validate_trajectory(record)
    return record


def validate_trajectory(record: dict[str, Any]) -> dict[str, Any]:
    """Raise :class:`SchemaError` unless ``record`` matches the HF schema."""
    if tuple(record) != HF_COLUMNS:
        missing = [c for c in HF_COLUMNS if c not in record]
        extra = [c for c in record if c not in HF_COLUMNS]
        raise SchemaError(
            f"Column mismatch (missing={missing}, extra={extra}, "
            f"order_ok={list(record) == list(HF_COLUMNS)})"
        )

    for column, expected in _COLUMN_TYPES.items():
        value = record[column]
        if expected is float and isinstance(value, int) and not isinstance(value, bool):
            continue
        if expected is int and isinstance(value, bool):
            raise SchemaError(f"Column {column!r} must be int, got bool")
        if not isinstance(value, expected):
            raise SchemaError(
                f"Column {column!r} must be {expected}, got {type(value)}"
            )

    turns = json.loads(record["turns"])
    expected_turns = 2 * record["total_steps"] + 1
    if len(turns) != expected_turns:
        raise SchemaError(
            f"Expected {expected_turns} turns for {record['total_steps']} steps, "
            f"got {len(turns)}"
        )
    if turns:
        if not turns[0].get("is_initial_observation"):
            raise SchemaError("First turn must be the initial observation")
        for index, turn in enumerate(turns):
            if turn.get("turn_index") != index:
                raise SchemaError(
                    f"Turn {index} has turn_index={turn.get('turn_index')!r}"
                )
            expected_role = "environment" if index % 2 == 0 else "agent"
            if turn.get("role") != expected_role:
                raise SchemaError(
                    f"Turn {index} must have role {expected_role!r}, "
                    f"got {turn.get('role')!r}"
                )
            if turn.get("is_initial_observation") or expected_role == "agent":
                if turn.get("max_score") != record["max_score"]:
                    raise SchemaError(
                        f"Turn {index} max_score does not match trajectory "
                        f"max_score {record['max_score']}"
                    )
                expected_turn_normalized = normalized(
                    turn.get("score_at_step", 0),
                    record["max_score"],
                )
                if not math.isclose(
                    float(turn.get("normalized_score_at_step", 0)),
                    expected_turn_normalized,
                    rel_tol=0.0,
                    abs_tol=1e-12,
                ):
                    raise SchemaError(
                        f"Turn {index} normalized_score_at_step does not match "
                        "score_at_step/max_score"
                    )
            if expected_role == "agent" and not valid_game_action(turn.get("content")):
                raise SchemaError(f"Agent turn {index} contains an invalid game action")

    progression = json.loads(record["score_progression"])
    if not progression or progression[0].get("step") != 1:
        raise SchemaError("score_progression must start at step 1")

    turn_scores = [
        int(turn["score_at_step"])
        for turn in turns
        if isinstance(turn.get("score_at_step"), int)
    ]
    peak_score = max(turn_scores, default=0)
    if record["score"] != peak_score:
        raise SchemaError(
            f"score must equal peak trajectory score {peak_score}, "
            f"got {record['score']}"
        )
    expected_normalized = normalized(peak_score, record["max_score"])
    if not math.isclose(
        float(record["normalized_score"]),
        expected_normalized,
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise SchemaError(
            "normalized_score does not match score/max_score: "
            f"expected {expected_normalized}, got {record['normalized_score']}"
        )

    if record["max_score"] < 0:
        raise SchemaError("max_score must be non-negative")
    if record["score"] > record["max_score"]:
        raise SchemaError("score cannot exceed max_score")
    if record["normalized_score"] > 1.0:
        raise SchemaError("normalized_score cannot exceed 1")
    if record["total_steps"] > record["max_steps"]:
        raise SchemaError("total_steps cannot exceed max_steps")
    return record
