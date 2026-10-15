"""Deterministic fakes: dry-run the whole pipeline without any API call."""

from __future__ import annotations

import hashlib
import itertools
import json
from typing import Any

from .bridge import ModelCall

FAKE_ACTIONS = (
    "look",
    "inventory",
    "north",
    "south",
    "east",
    "west",
    "take all",
    "open door",
    "examine me",
    "wait",
)


def _deterministic_action(session_id: str, step: int) -> str:
    digest = hashlib.blake2b(
        f"{session_id}:{step}".encode("utf-8"), digest_size=4
    ).digest()
    return FAKE_ACTIONS[int.from_bytes(digest, "big") % len(FAKE_ACTIONS)]


class FakeBridge:
    """Drop-in :class:`~copilot_evals.bridge.CopilotBridge` replacement."""

    def __init__(self, config=None, *, log_path=None, models=None) -> None:
        self.config = config
        self.log_path = log_path
        self._counter = itertools.count(1)
        self.sessions: dict[str, int] = {}
        self.closed: list[str] = []
        self.started = False
        self.calls = 0
        self.models = models or [
            {
                "id": "fake-model",
                "name": "Fake Model",
                "family": "fake-model",
                "supported_reasoning_efforts": ["low", "medium", "high", "max"],
            }
        ]
        # Optional {step: "no_tool_call" | Exception} injection used by tests.
        self.script: dict[int, Any] = {}

    def start(self, **_: Any) -> None:
        self.started = True

    def stop(self, **_: Any) -> None:
        self.started = False

    def restart(self) -> None:
        self.stop()
        self.start()

    def ping(self, **_: Any) -> bool:
        return True

    def list_models(self, **_: Any) -> list[dict[str, Any]]:
        return list(self.models)

    def create_session(self, *, model: str, reasoning_effort=None, session_id=None):
        new_id = session_id or f"fake-session-{next(self._counter)}"
        self.sessions[new_id] = 0
        return new_id

    def resume_session(self, *, session_id: str, model: str, reasoning_effort=None):
        if session_id not in self.sessions:
            self.sessions[session_id] = 0
        return session_id

    def close_session(self, session_id: str, *, delete: bool = True) -> None:
        self.sessions.pop(session_id, None)
        self.closed.append(session_id)

    def send(self, *, session_id: str, prompt: str) -> ModelCall:
        self.calls += 1
        step = self.sessions.get(session_id, 0) + 1
        self.sessions[session_id] = step

        scripted = self.script.get(self.calls)
        if isinstance(scripted, BaseException):
            raise scripted
        if scripted == "no_tool_call":
            return ModelCall(
                status="no_tool_call",
                assistant_text="I think I should go north.",
                usage=[self._usage(12, 3, 0)],
                duration=0.01,
            )

        action = _deterministic_action(session_id, step)
        return ModelCall(
            status="ok",
            action=action,
            raw_output=json.dumps(
                {"name": "game_action", "arguments": {"action": action}}
            ),
            thinking=f"(fake reasoning for step {step})",
            model="fake-model",
            usage=[self._usage(100 + step, 20, 5)],
            duration=0.01,
        )

    @staticmethod
    def _usage(input_tokens: int, output_tokens: int, reasoning: int):
        return {
            "model": "fake-model",
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "reasoning_tokens": reasoning,
            "cache_read_tokens": 0,
            "cache_write_tokens": 0,
            "duration_ms": 10,
            "cost": 0.0,
            "finish_reason": "tool_calls",
        }

    def __enter__(self) -> "FakeBridge":
        self.start()
        return self

    def __exit__(self, *exc: Any) -> None:
        self.stop()


class FakeJerichoEnv:
    """Tiny deterministic stand-in for ``tales/JerichoEnv*-v0``.

    Scores a point every ``score_every`` steps and can be configured to finish
    (won/lost) at a given step, which exercises the game-reset path without
    pulling in the Jericho dependency.
    """

    def __init__(
        self,
        *,
        max_score: int = 3,
        score_every: int = 2,
        done_at: int | None = None,
        won: bool = False,
    ) -> None:
        self.max_score = max_score
        self.score_every = score_every
        self.done_at = done_at
        self.won = won
        self.moves = 0
        self.score = 0
        self.resets = 0
        self.reset_seeds: list[int | None] = []
        self.history: list[str] = []

    def _info(self, feedback: str, done: bool = False) -> dict[str, Any]:
        return {
            "score": self.score,
            "max_score": self.max_score,
            "moves": self.moves,
            "feedback": feedback,
            "won": done and self.won,
            "lost": done and not self.won,
            "admissible_commands": [],
        }

    def reset(self, *, seed: int | None = None, options=None):
        self.moves = 0
        self.score = 0
        self.resets += 1
        self.reset_seeds.append(seed)
        observation = f"Welcome to the fake game (reset #{self.resets})."
        return observation, self._info(observation)

    def step(self, action: str):
        self.moves += 1
        self.history.append(action)
        if self.score < self.max_score and self.moves % self.score_every == 0:
            self.score += 1
        done = self.done_at is not None and self.moves == self.done_at
        if done and self.won:
            self.score = self.max_score
        feedback = f"You {action}. (move {self.moves}, score {self.score})"
        info = self._info(feedback, done)
        return feedback, 0, done, info

    def close(self) -> None:
        return None
