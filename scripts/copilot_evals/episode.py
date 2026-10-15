"""Resumable single-episode runner (one model, one game, one seed)."""

from __future__ import annotations

import os
import signal
import time
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from . import availability, prompts, schema, storage
from .bridge import BridgeError, ModelCall
from .config import RunConfig
from .manifest import Job
from .models import ModelSpec
from .schema import StepRecord

META_VERSION = 4


class EpisodeError(RuntimeError):
    """The episode could not continue; partial progress stays on disk."""


class ModelActionUnavailable(EpisodeError):
    """The model repeatedly declined or failed to invoke the required tool."""

    def __init__(self, step: int) -> None:
        self.step = int(step)
        super().__init__(
            f"Model did not provide a native game_action at step {self.step}."
        )


class StopRequested(EpisodeError):
    """A STOP file requested a resumable pause."""


def trajectory_path(config: RunConfig, job: Job) -> Path:
    return config.trajectories_dir / job.model_slug / job.game / f"{job.job_id}.json.gz"


def partial_dir(config: RunConfig, job: Job) -> Path:
    return config.partial_dir / job.model_slug / job.game / job.job_id


def telemetry_path(config: RunConfig, job: Job) -> Path:
    return config.telemetry_dir / "calls" / job.model_slug / f"{job.job_id}.jsonl.gz"


def telemetry_work_path(config: RunConfig, job: Job) -> Path:
    return telemetry_path(config, job).with_suffix("")


def read_call_telemetry(config: RunConfig, job: Job) -> list[dict[str, Any]]:
    work_path = telemetry_work_path(config, job)
    path = work_path if work_path.exists() else telemetry_path(config, job)
    records, damaged = storage.read_jsonl_prefix(path)
    if damaged:
        records = list(storage.iter_jsonl(path))
    return records


def _damaged_call_telemetry_path(
    config: RunConfig,
    job: Job,
    source: Path,
) -> Path:
    suffix = ".jsonl.gz" if source.name.endswith(".jsonl.gz") else source.suffix
    basename = source.name[: -len(suffix)] if suffix else source.name
    return (
        config.telemetry_dir
        / "damaged-call-logs"
        / job.model_slug
        / job.job_id
        / f"{basename}-{time.time_ns()}{suffix}"
    )


def _recover_damaged_call_telemetry(
    config: RunConfig,
    job: Job,
    source: Path,
) -> list[dict[str, Any]]:
    records = list(storage.iter_jsonl(source))
    quarantine_path = _damaged_call_telemetry_path(config, job, source)
    storage.atomic_write_bytes(quarantine_path, source.read_bytes())

    records.append(
        {
            "step": 0,
            "status": "telemetry_repaired",
            "session_id": None,
            "model": job.model,
            "reasoning_effort": job.reasoning_effort,
            "timestamp": time.time(),
            "api_calls": 0,
            "usage_unavailable": True,
            "recovered_records": len(records),
            "quarantine_path": str(quarantine_path),
        }
    )
    storage.rewrite_jsonl(telemetry_work_path(config, job), records)
    if source != telemetry_work_path(config, job):
        source.unlink(missing_ok=True)
    return records


def _migrate_final_call_telemetry(config: RunConfig, job: Job) -> None:
    final_path = telemetry_path(config, job)
    records, damaged = storage.read_jsonl_prefix(final_path)
    if damaged:
        _recover_damaged_call_telemetry(config, job, final_path)
        return
    storage.rewrite_jsonl(telemetry_work_path(config, job), records)
    final_path.unlink(missing_ok=True)


def append_call_telemetry(
    config: RunConfig,
    job: Job,
    entry: dict[str, Any],
) -> None:
    final_path = telemetry_path(config, job)
    work_path = telemetry_work_path(config, job)
    work_path.parent.mkdir(parents=True, exist_ok=True)

    if not work_path.exists() and final_path.exists():
        _migrate_final_call_telemetry(config, job)
    elif work_path.exists():
        final_path.unlink(missing_ok=True)

    storage.append_jsonl(work_path, entry)


def finalize_call_telemetry(
    config: RunConfig,
    job: Job,
) -> list[dict[str, Any]]:
    final_path = telemetry_path(config, job)
    work_path = telemetry_work_path(config, job)
    if work_path.exists():
        records, damaged = storage.read_jsonl_prefix(work_path)
        if damaged:
            records = _recover_damaged_call_telemetry(config, job, work_path)
        storage.rewrite_jsonl(final_path, records)
        work_path.unlink(missing_ok=True)
        return records

    records, damaged = storage.read_jsonl_prefix(final_path)
    if damaged:
        records = _recover_damaged_call_telemetry(config, job, final_path)
        storage.rewrite_jsonl(final_path, records)
        work_path.unlink(missing_ok=True)
    return records


def remove_call_telemetry(config: RunConfig, job: Job) -> None:
    telemetry_path(config, job).unlink(missing_ok=True)
    telemetry_work_path(config, job).unlink(missing_ok=True)


def native_crash_dir(config: RunConfig, job: Job) -> Path:
    return config.telemetry_dir / "native-crashes" / job.model_slug / job.job_id


def native_crash_limit_path(config: RunConfig, job: Job) -> Path:
    return partial_dir(config, job) / "native-crash-limit.json"


def default_env_factory(config: RunConfig, job: Job) -> Callable[[], Any]:
    from .games import make_env

    return lambda: make_env(
        job.game,
        admissible_commands=config.admissible_commands,
        working_directory=partial_dir(config, job) / "game-files",
    )


def known_unsafe_actions(game: str, observation: str) -> dict[str, str]:
    if game in {"JerichoEnvJack", "JerichoEnvSpeedif_2C9F6Ff3"}:
        return {"undo": "UNDO"}
    normalized = " ".join(observation.casefold().split())
    if game == "JerichoEnvArthur" and (
        "do you want to undo, restore, restart, quit, or get a hint?" in normalized
        or "please type undo, restore, restart, quit, or hint." in normalized
    ):
        return {"undo": "UNDO"}
    return {}


@dataclass
class EpisodeResult:
    record: dict[str, Any]
    summary: dict[str, Any]
    skipped: bool = False
    resumed: bool = False
    divergences: list[str] = field(default_factory=list)


@contextmanager
def _native_step_deadline(seconds: float | None):
    if not seconds or not hasattr(signal, "SIGALRM"):
        yield
        return

    previous_handler = signal.getsignal(signal.SIGALRM)
    previous_delay, previous_interval = signal.getitimer(signal.ITIMER_REAL)
    started = time.monotonic()
    deadline = float(seconds)
    handler = signal.SIG_DFL
    if previous_delay > 0 and previous_delay <= deadline:
        # Preserve a sooner enclosing Python deadline (for example JAZ's REPL
        # timeout) instead of replacing it with the hard native-step kill.
        deadline = previous_delay
        handler = previous_handler
    signal.signal(signal.SIGALRM, handler)
    signal.setitimer(signal.ITIMER_REAL, deadline)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous_handler)
        if previous_delay > 0:
            elapsed = time.monotonic() - started
            remaining = max(previous_delay - elapsed, 1e-6)
            signal.setitimer(signal.ITIMER_REAL, remaining, previous_interval)


def step_env(
    env,
    state: dict[str, Any],
    action: str,
    *,
    timeout: float | None = None,
) -> tuple[str, str]:
    """Execute one action, mirroring ``benchmark.py``'s ``_step_env``."""
    prev_obs = state["obs"]
    normalized_action = schema.normalize_game_action(action)

    if normalized_action is None:
        state["obs"] = "The game only allows one action per step."
    else:
        native_action = schema.native_game_action(normalized_action)
        with _native_step_deadline(timeout):
            obs, _, done, info = env.step(native_action)
        state["obs"] = obs
        state["done"] = done
        state["info"] = info

    state["score"] = state["info"]["score"]
    state["moves"] = state["info"]["moves"]
    state["highscore"] = max(state["score"], state["highscore"])
    reported_max_score = state["info"].get("max_score")
    if isinstance(reported_max_score, (int, float)) and not isinstance(
        reported_max_score, bool
    ):
        state["max_score"] = max(
            int(state.get("max_score", 0)),
            int(reported_max_score),
        )
    return prev_obs, state["info"]["feedback"]


def handle_done(env, state: dict[str, Any], reset_seed: int) -> str:
    """Mirror ``benchmark.py``'s ``_handle_done``: ``"none" | "break" | "reset"``."""
    if not state["done"]:
        return "none"

    if state["info"].get("won"):
        state["nb_wins"] += 1
        if state["highscore"] == state["max_score"]:
            return "break"
    elif state["info"].get("lost"):
        state["nb_losts"] += 1

    last_obs = state["obs"]
    obs, info = env.reset(seed=reset_seed)
    state["obs"] = last_obs + "\n\n-= Restarting =-\n" + obs
    state["info"] = info
    state["done"] = False
    state["nb_resets"] += 1
    return "reset"


class EpisodeRunner:
    """Plays one game with one model, checkpointing after every step."""

    def __init__(
        self,
        config: RunConfig,
        bridge,
        model: ModelSpec,
        job: Job,
        *,
        env_factory: Callable[[], Any] | None = None,
        log: Callable[[str], None] | None = None,
        heartbeat: Callable[[int], None] | None = None,
    ) -> None:
        self.config = config
        self.bridge = bridge
        self.model = model
        self.job = job
        self.env_factory = env_factory or default_env_factory(config, job)
        self.log = log or (lambda message: None)
        self.heartbeat = heartbeat or (lambda step: None)

        self.dir = partial_dir(config, job)
        self.meta_path = self.dir / "meta.json"
        self.steps_path = self.dir / "steps.jsonl"
        self.pending_path = self.dir / "pending.json"
        self.final_path = trajectory_path(config, job)

        self.divergences: list[str] = []
        self.session_id: str | None = None
        self.pending_recap = False
        self.termination_reason: str | None = None
        self.termination_step: int | None = None
        self._last_heartbeat = 0.0
        self._deleted_sessions: set[str] = set()

    @property
    def effective_seed(self) -> int:
        return (
            int(self.config.game_seed)
            if self.config.game_seed is not None
            else int(self.job.seed)
        )

    # -- Entry point -----------------------------------------------------

    def run(self) -> EpisodeResult:
        existing = self._load_final()
        if existing is not None:
            self._cleanup_recorded_sessions()
            storage.remove_tree(self.dir)
            return EpisodeResult(
                record=existing,
                summary=self._summary(existing, duration=0.0),
                skipped=True,
            )

        started = time.time()
        meta, steps = self._load_partial()
        # Replay must rebuild saves in action order. Reusing files left by a
        # later recorded `save` could make an earlier `restore` succeed.
        storage.remove_tree(self.dir / "game-files")
        env = self.env_factory()
        try:
            record = self._play(env, meta, steps)
        except BaseException as exc:
            self._flush_call_telemetry()
            # Keep the Copilot session on disk (resume needs it), but let the
            # bridge forget about it so long-lived workers do not leak sessions.
            if isinstance(exc, StopRequested):
                self.session_id = None
            else:
                self._release_session(delete=False)
            raise
        finally:
            try:
                env.close()
            except Exception:  # pragma: no cover - env teardown is best effort
                pass

        self._release_session(delete=not self.config.keep_sessions)
        self._cleanup_recorded_sessions()
        duration = time.time() - started
        summary = self._summary(record, duration=duration)
        self._finalize(record, summary)
        return EpisodeResult(
            record=record,
            summary=summary,
            resumed=bool(steps),
            divergences=self.divergences,
        )

    # -- Persistence -----------------------------------------------------

    def _load_final(self) -> dict[str, Any] | None:
        record = storage.read_json(self.final_path)
        if record is None:
            return None
        try:
            record = schema.validate_trajectory(record)
        except schema.SchemaError as exc:
            self.log(f"Discarding invalid trajectory {self.final_path}: {exc}")
            self.final_path.unlink(missing_ok=True)
            return None
        if self._has_legacy_ambiguous_session_reuse():
            self.log(
                "Discarding trajectory produced after an ambiguous bridge error "
                "reused the same model session."
            )
            self._invalidate_legacy_ambiguous_session_reuse()
            from .telemetry import remove_episode_telemetry

            remove_episode_telemetry(
                self.config,
                model=self.job.model,
                job_id=self.job.job_id,
            )
            self.final_path.unlink(missing_ok=True)
            return None
        self._ensure_episode_telemetry(record)
        return record

    def _ensure_episode_telemetry(self, record: dict[str, Any]) -> None:
        from .telemetry import (
            episode_record_path,
            episode_telemetry,
            find_shared_episode,
            invalidated_episode_path,
            write_episode_telemetry,
            write_episode_telemetry_shard,
        )

        shard_path = episode_record_path(
            self.config,
            model=self.job.model,
            job_id=self.job.job_id,
        )
        if shard_path.exists():
            invalidated_episode_path(
                self.config,
                model=self.job.model,
                job_id=self.job.job_id,
            ).unlink(missing_ok=True)
            return

        payload = find_shared_episode(self.config, self.job.job_id)
        if payload is not None:
            write_episode_telemetry_shard(self.config, payload)
            return

        calls = finalize_call_telemetry(self.config, self.job)
        summary = self._summary(record, duration=0.0)
        payload = episode_telemetry(
            job=self.job,
            model=self.model,
            record=record,
            calls=calls,
            summary=summary,
            divergences=self.divergences,
        )
        payload["recovered_after_final"] = True
        write_episode_telemetry(self.config, payload)

    def _load_partial(self) -> tuple[dict[str, Any] | None, list[StepRecord]]:
        meta = storage.read_json(self.meta_path)
        if not isinstance(meta, dict) or meta.get("version") != META_VERSION:
            if meta is not None:
                self.log("Partial state has an incompatible layout; restarting.")
                self._cleanup_sessions(meta, reason="incompatible-partial")
            storage.remove_tree(self.dir)
            return None, []

        incompatible = (
            meta.get("job_id") != self.job.job_id
            or meta.get("model") != self.job.model
            or int(meta.get("max_steps", -1)) != int(self.job.max_steps)
            or meta.get("reasoning_effort") != self.job.reasoning_effort
            or int(meta.get("game_seed", -1)) != self.effective_seed
        )
        if incompatible:
            self.log("Partial state does not match this job; restarting from scratch.")
            self._cleanup_sessions(meta, reason="mismatched-partial")
            storage.remove_tree(self.dir)
            return None, []
        if self._has_legacy_ambiguous_session_reuse():
            self.log(
                "Partial state reused a model session after an ambiguous bridge "
                "error; restarting from scratch."
            )
            self._cleanup_sessions(meta, reason="legacy-ambiguous-session-reuse")
            self._invalidate_legacy_ambiguous_session_reuse()
            from .telemetry import remove_episode_telemetry

            remove_episode_telemetry(
                self.config,
                model=self.job.model,
                job_id=self.job.job_id,
            )
            storage.remove_tree(self.dir)
            return None, []

        payloads, damaged = storage.read_jsonl_prefix(self.steps_path)
        steps: list[StepRecord] = []
        for index, payload in enumerate(payloads, start=1):
            if not isinstance(payload, dict):
                damaged = True
                break
            try:
                record = StepRecord.from_dict(payload)
            except TypeError:
                damaged = True
                break
            if record.step != index:
                self.log(f"Step log breaks at index {index}; truncating there.")
                damaged = True
                break
            steps.append(record)

        if damaged or len(steps) != len(payloads):
            storage.rewrite_jsonl(self.steps_path, [s.to_dict() for s in steps])
        self._recover_uncommitted_model_call(meta, steps)
        self._recover_legacy_game_reset_rotation(meta)
        return meta, steps

    def _write_meta(self, meta: dict[str, Any]) -> None:
        meta["updated_at"] = time.time()
        storage.atomic_write_json(self.meta_path, meta)

    def _has_legacy_ambiguous_session_reuse(self) -> bool:
        calls = read_call_telemetry(self.config, self.job)
        marker_indexes = [
            index
            for index, call in enumerate(calls)
            if call.get("status") == "legacy_bridge_reuse_invalidated"
        ]
        if marker_indexes:
            marker_index = marker_indexes[-1]
            if any(
                call.get("call_id") or int(call.get("api_calls") or 0) > 0
                for call in calls[marker_index + 1 :]
            ):
                return False
            calls = calls[:marker_index]

        ambiguous_sessions: set[str] = set()
        for call in calls:
            session_id = call.get("session_id")
            if call.get("status") == "bridge_error":
                if isinstance(session_id, str) and session_id:
                    ambiguous_sessions.add(session_id)
                continue
            if session_id in ambiguous_sessions and (
                bool(call.get("call_id")) or int(call.get("api_calls") or 0) > 0
            ):
                return True
        return False

    def _invalidate_legacy_ambiguous_session_reuse(self) -> None:
        append_call_telemetry(
            self.config,
            self.job,
            {
                "step": 0,
                "status": "legacy_bridge_reuse_invalidated",
                "session_id": None,
                "model": self.model.id,
                "reasoning_effort": self.model.effort_label,
                "timestamp": time.time(),
                "api_calls": 0,
            },
        )

    def _append_step(self, record: StepRecord) -> None:
        storage.append_jsonl(self.steps_path, record.to_dict())

    # -- Model plumbing --------------------------------------------------

    def _ensure_session(self, meta: dict[str, Any], history) -> str:
        """Resume the recorded session, else start a fresh one with a recap."""
        sessions = meta.setdefault("sessions", [])
        if sessions and sessions[-1].get("resumable", True):
            session_id = sessions[-1]["session_id"]
            try:
                resumed = self.bridge.resume_session(
                    session_id=session_id,
                    model=self.model.id,
                    reasoning_effort=self.model.effort,
                )
                availability.clear_model_pause(self.config, self.model.id)
                self.session_id = resumed
                if resumed != session_id:
                    sessions[-1]["session_id"] = resumed
                    self._write_meta(meta)
                return resumed
            except BridgeError as exc:
                if availability.is_model_unavailable_error(
                    exc,
                    model=self.model.id,
                ):
                    raise availability.ModelTemporarilyUnavailable(
                        self.model.id, str(exc)
                    ) from exc
                self.log(f"Could not resume session {session_id}: {exc}")
                self.divergences.append(f"session-resume-failed:{session_id}")
                sessions[-1]["resumable"] = False
                sessions[-1]["ended_at"] = time.time()
                sessions[-1]["end_reason"] = "resume-failed"
                self.pending_recap = bool(history)
                if not self.config.keep_sessions:
                    self._delete_session(session_id, reason="resume-failed")
        elif sessions:
            self.pending_recap = bool(history)
            session_id = sessions[-1].get("session_id")
            if session_id and not self.config.keep_sessions:
                self._delete_session(
                    session_id,
                    reason=sessions[-1].get("end_reason") or "non-resumable",
                )

        try:
            session_id = self.bridge.create_session(
                model=self.model.id, reasoning_effort=self.model.effort
            )
        except BridgeError as exc:
            if availability.is_model_unavailable_error(
                exc,
                model=self.model.id,
            ):
                raise availability.ModelTemporarilyUnavailable(
                    self.model.id, str(exc)
                ) from exc
            raise
        availability.clear_model_pause(self.config, self.model.id)
        sessions.append(
            {
                "session_id": session_id,
                "started_step": len(history) + 1,
                "created_at": time.time(),
            }
        )
        self.session_id = session_id
        meta.pop("pending_session_rotation_step", None)
        self._write_meta(meta)
        return session_id

    def _release_session(self, *, delete: bool) -> None:
        """Disconnect (and optionally delete) the episode's Copilot session."""
        session_id, self.session_id = self.session_id, None
        if not session_id:
            return
        if delete:
            self._delete_session(session_id, reason="episode-finished")
            return
        try:
            self.bridge.close_session(session_id, delete=False)
        except Exception as exc:
            self.log(f"Could not disconnect session {session_id}: {exc!r}")

    def _abandon_session(self, meta: dict[str, Any], reason: str) -> None:
        """Discard a session that repeatedly refused the required native tool."""
        session_id, self.session_id = self.session_id, None
        sessions = meta.setdefault("sessions", [])
        if session_id and sessions and sessions[-1].get("session_id") == session_id:
            sessions[-1]["resumable"] = False
            sessions[-1]["ended_at"] = time.time()
            sessions[-1]["end_reason"] = reason
        self.pending_recap = True
        self._write_meta(meta)
        if session_id:
            if self.config.keep_sessions:
                try:
                    self.bridge.close_session(session_id, delete=False)
                except Exception as exc:
                    self.log(
                        f"Could not disconnect abandoned session {session_id}: {exc!r}"
                    )
            else:
                self._delete_session(session_id, reason=reason)

    def _delete_session(self, session_id: str, *, reason: str) -> bool:
        if session_id in self._deleted_sessions:
            return True
        session_path = Path(self.config.copilot_home) / "session-state" / session_id
        last_error: Exception | None = None
        for attempt in range(1, 4):
            try:
                self.bridge.close_session(session_id, delete=True)
            except Exception as exc:
                message = str(exc).lower()
                if (
                    "session file not found" in message
                    or "session not found" in message
                ):
                    storage.remove_tree(session_path)
                    if not session_path.exists():
                        self._deleted_sessions.add(session_id)
                        return True
                last_error = exc
                self.log(
                    f"Could not delete session {session_id} ({reason}), "
                    f"attempt {attempt}/3: {exc!r}"
                )
            else:
                storage.remove_tree(session_path)
                if not session_path.exists():
                    self._deleted_sessions.add(session_id)
                    return True
                last_error = RuntimeError(
                    "SDK deletion returned successfully but the session "
                    "directory remained"
                )
                self.log(
                    f"Session directory {session_id} remained after SDK "
                    f"deletion ({reason}), attempt {attempt}/3."
                )
            if attempt < 3:
                time.sleep(attempt)
        storage.remove_tree(session_path)
        if not session_path.exists():
            self._deleted_sessions.add(session_id)
            self.log(
                f"Removed stale session directory {session_id} locally "
                f"after SDK deletion failed ({reason})."
            )
            return True
        self.divergences.append(f"session-delete-failed:{session_id}:{last_error!r}")
        return False

    @property
    def _rate_limit_path(self) -> Path:
        return self.config.output_root / "rate-limit.json"

    @staticmethod
    def _is_rate_limit_error(error: BaseException) -> bool:
        return "rate limit" in str(error).lower()

    def _extend_rate_limit_gate(self, *, seconds: float = 70.0) -> None:
        current = storage.read_json(self._rate_limit_path)
        current_until = (
            float(current.get("until") or 0) if isinstance(current, dict) else 0.0
        )
        storage.atomic_write_json(
            self._rate_limit_path,
            {
                "until": max(current_until, time.time() + seconds),
                "model": self.model.id,
                "updated_at": time.time(),
            },
        )

    def _wait_for_rate_limit_gate(self) -> None:
        while True:
            payload = storage.read_json(self._rate_limit_path)
            deadline = (
                float(payload.get("until") or 0) if isinstance(payload, dict) else 0.0
            )
            if deadline:
                deadline += (os.getpid() % 1000) / 100.0
            remaining = deadline - time.time()
            if remaining <= 0:
                return
            if self.config.stop_file.exists():
                raise StopRequested("STOP file requested a resumable pause.")
            time.sleep(min(5.0, remaining))

    def _cleanup_recorded_sessions(self) -> None:
        meta = storage.read_json(self.meta_path)
        if not isinstance(meta, dict):
            return
        self._cleanup_sessions(meta, reason="episode-cleanup")

    def _cleanup_sessions(self, meta: dict[str, Any], *, reason: str) -> None:
        if self.config.keep_sessions:
            return
        for session in meta.get("sessions") or []:
            session_id = session.get("session_id")
            if session_id:
                self._delete_session(session_id, reason=reason)

    def _recover_uncommitted_model_call(
        self,
        meta: dict[str, Any],
        steps: list[StepRecord],
    ) -> None:
        step = meta.get("inflight_model_step")
        session_id = meta.get("inflight_session_id")
        pending = storage.read_json(self.pending_path)

        if isinstance(pending, dict):
            pending_step = int(pending.get("step") or -1)
            pending_session = pending.get("session_id")
            matches_inflight = (
                step is not None
                and pending_step == int(step)
                and isinstance(pending_session, str)
                and pending_session == session_id
            )

            if pending_step <= len(steps):
                self.log(
                    f"Discarding pending response for committed step {pending_step}."
                )
                self.pending_path.unlink(missing_ok=True)
                pending = None
                if matches_inflight:
                    meta.pop("inflight_model_step", None)
                    meta.pop("inflight_session_id", None)
                    self._write_meta(meta)
                    return
            elif matches_inflight:
                return
            elif step is not None:
                self._quarantine_stale_pending(
                    pending,
                    reason=(
                        f"pending step/session {pending_step}/{pending_session!r} "
                        f"did not match in-flight call {step}/{session_id!r}"
                    ),
                )
                pending = None

        if step is None:
            return
        for session in reversed(meta.get("sessions") or []):
            if session_id is None or session.get("session_id") == session_id:
                session["resumable"] = False
                session["ended_at"] = time.time()
                session["end_reason"] = "uncommitted-model-call"
                break
        entry = {
            "step": int(step),
            "status": "uncommitted_model_call",
            "session_id": session_id,
            "model": self.model.id,
            "reasoning_effort": self.model.effort_label,
            "timestamp": time.time(),
            "api_calls": 0,
            "usage_unavailable": True,
        }
        append_call_telemetry(self.config, self.job, entry)
        meta.pop("inflight_model_step", None)
        meta.pop("inflight_session_id", None)
        self._write_meta(meta)

    def _quarantine_stale_pending(
        self,
        pending: dict[str, Any],
        *,
        reason: str,
    ) -> None:
        step = int(pending.get("step") or -1)
        directory = (
            self.config.telemetry_dir
            / "orphaned-pending"
            / self.job.model_slug
            / self.job.job_id
        )
        directory.mkdir(parents=True, exist_ok=True)
        call_id = pending.get("call_id")
        suffix = call_id if isinstance(call_id, str) and call_id else uuid.uuid4().hex
        path = directory / f"pending-step-{step:03d}-{suffix}.json"
        payload = dict(pending)
        payload["orphaned_pending"] = {
            "reason": reason,
            "quarantined_at": time.time(),
        }
        storage.atomic_write_json(path, payload)

        usage = pending.get("usage") or []
        call_recorded = self._call_id_recorded(call_id)
        entry: dict[str, Any] = {
            "step": step,
            "status": "stale_pending_action",
            "session_id": pending.get("session_id"),
            "model": self.model.id,
            "reasoning_effort": self.model.effort_label,
            "timestamp": time.time(),
            "action": pending.get("action", ""),
            "error": reason[:500],
        }
        if call_recorded:
            entry["source_call_id"] = call_id
            entry["api_calls"] = 0
        else:
            entry["call_id"] = call_id or uuid.uuid4().hex
            entry["api_calls"] = len(usage)
            for key in (
                "input_tokens",
                "output_tokens",
                "reasoning_tokens",
                "cache_read_tokens",
                "cache_write_tokens",
                "duration_ms",
                "cost",
                "aiu_nano",
            ):
                entry[key] = sum(item.get(key) or 0 for item in usage)
        append_call_telemetry(self.config, self.job, entry)
        self.pending_path.unlink(missing_ok=True)

    def _recover_legacy_game_reset_rotation(self, meta: dict[str, Any]) -> None:
        if meta.pop("pending_session_rotation_step", None) is None:
            return

        sessions = meta.setdefault("sessions", [])
        if sessions:
            latest = sessions[-1]
            if latest.get("end_reason") == "game-reset":
                latest.pop("resumable", None)
                latest.pop("ended_at", None)
                latest.pop("end_reason", None)
        self._write_meta(meta)

    def _mark_current_session_non_resumable(
        self,
        meta: dict[str, Any],
        *,
        reason: str,
    ) -> str | None:
        session_id, self.session_id = self.session_id, None
        sessions = meta.setdefault("sessions", [])
        if session_id and sessions and sessions[-1].get("session_id") == session_id:
            sessions[-1]["resumable"] = False
            sessions[-1]["ended_at"] = time.time()
            sessions[-1]["end_reason"] = reason
        self.pending_recap = True
        self._write_meta(meta)
        return session_id

    def _mark_model_call_inflight(self, meta: dict[str, Any], step: int) -> None:
        if (
            meta.get("inflight_model_step") == step
            and meta.get("inflight_session_id") == self.session_id
        ):
            return
        meta["inflight_model_step"] = step
        meta["inflight_session_id"] = self.session_id
        self._write_meta(meta)

    def _call_id_recorded(self, call_id: str | None) -> bool:
        if not call_id:
            return False
        return any(
            entry.get("call_id") == call_id
            for entry in read_call_telemetry(self.config, self.job)
        )

    def _model_action(
        self, meta: dict[str, Any], observation: str, step: int, history
    ) -> ModelCall:
        crashed_actions = known_unsafe_actions(self.job.game, observation)
        crashed_actions.update(self._crashed_actions(step))

        def current_prompt() -> str:
            base = (
                prompts.recap_prompt(history, observation)
                if self.pending_recap
                else prompts.observation_prompt(observation)
            )
            if crashed_actions:
                return prompts.avoid_crashed_actions_prompt(
                    base, sorted(crashed_actions.values())
                )
            return base

        prompt = current_prompt()
        self.pending_recap = False

        tool_retries = 0
        call_retries = 0
        session_recoveries = 0
        while True:
            self._wait_for_rate_limit_gate()
            self._mark_model_call_inflight(meta, step)
            try:
                call = self.bridge.send(session_id=self.session_id, prompt=prompt)
            except BridgeError as exc:
                if availability.is_model_unavailable_error(
                    exc,
                    model=self.model.id,
                ):
                    self._record_call(
                        step,
                        "model_unavailable",
                        None,
                        error=str(exc),
                    )
                    meta.pop("inflight_model_step", None)
                    meta.pop("inflight_session_id", None)
                    self._write_meta(meta)
                    raise availability.ModelTemporarilyUnavailable(
                        self.model.id, str(exc)
                    ) from exc
                rate_limited = self._is_rate_limit_error(exc)
                if rate_limited:
                    self._extend_rate_limit_gate()
                else:
                    call_retries += 1
                self._record_call(step, "bridge_error", None, error=str(exc))
                ambiguous_session = self._mark_current_session_non_resumable(
                    meta,
                    reason="ambiguous-send-failure",
                )
                meta.pop("inflight_model_step", None)
                meta.pop("inflight_session_id", None)
                self._write_meta(meta)
                self.log(f"Bridge error at step {step} ({exc}); restarting bridge.")
                if self.config.stop_file.exists():
                    raise StopRequested("STOP file requested a resumable pause.")
                if not rate_limited and call_retries <= self.config.max_call_retries:
                    time.sleep(min(2**call_retries, 30))
                restart_error: Exception | None = None
                try:
                    self.bridge.restart()
                except Exception as caught:
                    restart_error = caught
                if ambiguous_session and not self.config.keep_sessions:
                    self._delete_session(
                        ambiguous_session,
                        reason="ambiguous-send-failure",
                    )
                if not rate_limited and call_retries > self.config.max_call_retries:
                    raise EpisodeError(f"Model call failed at step {step}: {exc}")
                if restart_error is not None:
                    raise EpisodeError(
                        f"Bridge restart failed at step {step}: {restart_error}"
                    )
                self._ensure_session(meta, history)
                prompt = current_prompt()
                self.pending_recap = False
                continue

            normalized_action = schema.normalize_game_action(call.action)
            if call.ok and normalized_action is not None:
                call.action = normalized_action
                normalized = normalized_action.casefold()
                if normalized not in crashed_actions:
                    if call.call_id is None:
                        call.call_id = uuid.uuid4().hex
                    return call
                call.status = "previously_crashed_action"
            if call.ok:
                call.status = "invalid_tool_action"

            self._record_call(step, call.status, call)
            meta.pop("inflight_model_step", None)
            meta.pop("inflight_session_id", None)
            self._write_meta(meta)
            if self.config.stop_file.exists():
                raise StopRequested("STOP file requested a resumable pause.")
            tool_retries += 1
            if tool_retries > self.config.max_tool_retries:
                self._abandon_session(meta, "native-tool-retries-exhausted")
                if session_recoveries == 0:
                    session_recoveries += 1
                    tool_retries = 0
                    self._ensure_session(meta, history)
                    prompt = prompts.recap_prompt(history, observation)
                    if crashed_actions:
                        prompt = prompts.avoid_crashed_actions_prompt(
                            prompt, sorted(crashed_actions.values())
                        )
                    self.pending_recap = False
                    continue
                raise ModelActionUnavailable(step)
            prompt = (
                prompts.crashed_action_nudge(call.action)
                if call.status == "previously_crashed_action"
                else prompts.NUDGE_PROMPT
            )

    def _crashed_actions(self, step: int) -> dict[str, str]:
        actions: dict[str, str] = {}
        for path in native_crash_dir(self.config, self.job).glob("pending-*.json"):
            pending = storage.read_json(path)
            if (
                isinstance(pending, dict)
                and int(pending.get("step") or -1) == step
                and isinstance(pending.get("action"), str)
            ):
                action = schema.normalize_game_action(pending["action"])
                if action is not None:
                    actions[action.casefold()] = action
        return actions

    def _record_call(
        self,
        step: int,
        status: str,
        call: ModelCall | None,
        *,
        error: str | None = None,
        include_usage: bool = True,
        source_call_id: str | None = None,
    ) -> None:
        entry: dict[str, Any] = {
            "step": step,
            "status": status,
            "session_id": self.session_id,
            "model": self.model.id,
            "reasoning_effort": self.model.effort_label,
            "timestamp": time.time(),
        }
        if call is not None:
            if include_usage:
                if call.call_id is None:
                    call.call_id = uuid.uuid4().hex
                entry["call_id"] = call.call_id
                entry.update(call.totals())
                entry["wall_time"] = round(call.duration, 3)
                entry["extra_tool_calls"] = call.extra_tool_calls
            else:
                entry["source_call_id"] = source_call_id or call.call_id
                entry["api_calls"] = 0
            if call.status in {"ok", "previously_crashed_action"}:
                entry["action"] = call.action
            else:
                if call.action:
                    entry["attempted_action"] = call.action[:500]
                if call.assistant_text:
                    entry["assistant_text"] = call.assistant_text[:500]
                if call.raw_output:
                    entry["raw_output"] = call.raw_output[:1000]
        if error:
            entry["error"] = error[:500]
        append_call_telemetry(self.config, self.job, entry)

    # -- Play ------------------------------------------------------------

    def _play(
        self, env, meta: dict[str, Any] | None, steps: list[StepRecord]
    ) -> dict[str, Any]:
        observation, info = env.reset(seed=self.effective_seed)
        initial_max_score = int(info["max_score"])
        state = {
            "obs": observation,
            "info": info,
            "done": False,
            "score": int(info.get("score", 0)),
            "highscore": int(info.get("score", 0)),
            "moves": int(info.get("moves", 0)),
            "max_score": initial_max_score,
            "nb_wins": 0,
            "nb_losts": 0,
            "nb_resets": 0,
        }

        if meta is None:
            meta = {
                "version": META_VERSION,
                "job_id": self.job.job_id,
                "model": self.job.model,
                "model_slug": self.job.model_slug,
                "game": self.job.game,
                "seed": self.job.seed,
                "reasoning_effort": self.job.reasoning_effort,
                "max_steps": self.job.max_steps,
                "game_seed": self.effective_seed,
                "initial_observation": observation,
                "max_score": initial_max_score,
                "sessions": [],
                "created_at": time.time(),
            }
            self._write_meta(meta)
        else:
            recorded = meta.get("initial_observation")
            if recorded is not None and recorded != observation:
                self._diverged("initial-observation", 0, recorded, observation)
            meta["initial_observation"] = meta.get("initial_observation", observation)

        if steps:
            steps = self._replay(env, state, steps)
            if not steps:
                # Replay diverged badly and the episode was reset from scratch.
                self._cleanup_sessions(meta, reason="replay-divergence")
                storage.remove_tree(self.dir)
                observation, info = env.reset(seed=self.effective_seed)
                state.update(
                    obs=observation,
                    info=info,
                    done=False,
                    score=int(info.get("score", 0)),
                    highscore=int(info.get("score", 0)),
                    moves=int(info.get("moves", 0)),
                    max_score=int(info["max_score"]),
                    nb_resets=0,
                    nb_wins=0,
                    nb_losts=0,
                )
                meta["sessions"] = []
                meta["initial_observation"] = observation
                self._write_meta(meta)

        meta["max_score"] = int(state["max_score"])
        history = [(step.observation, step.action) for step in steps]
        initial_observation = meta["initial_observation"]
        finished = (
            bool(steps) and steps[-1].won and state["highscore"] == state["max_score"]
        )
        step_number = len(steps) + 1

        native_crash_limit = storage.read_json(
            native_crash_limit_path(self.config, self.job)
        )
        if (
            isinstance(native_crash_limit, dict)
            and not finished
            and step_number <= self.job.max_steps
        ):
            self.termination_reason = "native_crash_limit"
            self.termination_step = step_number
            meta["termination_reason"] = self.termination_reason
            meta["termination_step"] = self.termination_step
            meta.pop("inflight_model_step", None)
            meta.pop("inflight_session_id", None)
            self._write_meta(meta)
            self.log(
                f"{self.job.job_id} ended early at step {step_number} after "
                f"{native_crash_limit.get('total_faults', 'multiple')} "
                "quarantined native interpreter faults."
            )
            finished = True

        if not finished and len(steps) < self.job.max_steps:
            self._ensure_session(meta, history)

        pending = storage.read_json(self.pending_path)

        while not finished and step_number <= self.job.max_steps:
            if self.config.stop_file.exists():
                raise StopRequested("STOP file requested a resumable pause.")

            observation = state["obs"]
            score_before = int(state["score"])

            call = None
            if isinstance(pending, dict) and pending.get("step") == step_number:
                pending_action = schema.normalize_game_action(pending.get("action", ""))
                pending_call = ModelCall(
                    status="ok" if pending_action else "invalid_pending_action",
                    action=pending_action or pending.get("action", ""),
                    raw_output=pending.get("raw_output", ""),
                    thinking=pending.get("thinking"),
                    model=pending.get("model"),
                    usage=pending.get("usage") or [],
                    call_id=pending.get("call_id"),
                )
                if pending_call.ok:
                    call = pending_call
                    call_recorded = self._call_id_recorded(call.call_id)
                    self._record_call(
                        step_number,
                        "recovered_pending",
                        call,
                        include_usage=not call_recorded,
                        source_call_id=call.call_id if call_recorded else None,
                    )
                    if not pending.get("call_id"):
                        pending["call_id"] = call.call_id
                        storage.atomic_write_json(self.pending_path, pending)
                else:
                    call_recorded = self._call_id_recorded(pending_call.call_id)
                    self._record_call(
                        step_number,
                        "invalid_pending_action",
                        pending_call,
                        include_usage=not call_recorded,
                        source_call_id=(
                            pending_call.call_id if call_recorded else None
                        ),
                    )
                    self.log(
                        f"Discarding invalid pending action at step {step_number}."
                    )
                    self.pending_path.unlink(missing_ok=True)
            pending = None

            if call is None:
                try:
                    call = self._model_action(meta, observation, step_number, history)
                except ModelActionUnavailable:
                    self.termination_reason = "model_action_unavailable"
                    self.termination_step = step_number
                    meta["termination_reason"] = self.termination_reason
                    meta["termination_step"] = self.termination_step
                    meta.pop("inflight_model_step", None)
                    meta.pop("inflight_session_id", None)
                    self._write_meta(meta)
                    self.log(
                        f"{self.job.job_id} ended early at step {step_number}: "
                        "the model did not provide a native game_action after "
                        "two fresh sessions."
                    )
                    break
                storage.atomic_write_json(
                    self.pending_path,
                    {
                        "step": step_number,
                        "action": call.action,
                        "raw_output": call.raw_output,
                        "thinking": call.thinking,
                        "model": call.model,
                        "usage": call.usage,
                        "call_id": call.call_id,
                        "session_id": self.session_id,
                        "written_at": time.time(),
                    },
                )
                self._record_call(step_number, "ok", call)

            totals = call.totals()
            _, feedback = step_env(
                env,
                state,
                call.action,
                timeout=self.config.native_step_timeout,
            )
            terminal_done = bool(state["done"])
            terminal_won = bool(state["info"].get("won"))
            terminal_lost = bool(state["info"].get("lost"))
            outcome = handle_done(env, state, self.effective_seed)

            record = StepRecord(
                step=step_number,
                action=call.action,
                raw_output=call.raw_output,
                thinking=call.thinking,
                token_usage=schema.compat_token_usage(
                    {
                        "input_tokens": totals["input_tokens"],
                        "output_tokens": totals["output_tokens"],
                        "reasoning_tokens": totals["reasoning_tokens"],
                    }
                ),
                thinking_tokens=int(totals["reasoning_tokens"]),
                score_before=score_before,
                score_after=int(state["score"]),
                max_score=int(state["max_score"]),
                feedback=feedback,
                observation=observation,
                moves=int(state["moves"]),
                done=terminal_done,
                won=terminal_won,
                lost=terminal_lost,
                did_reset=outcome == "reset",
                session_id=self.session_id,
                usage={
                    "input_tokens": int(totals["input_tokens"]),
                    "output_tokens": int(totals["output_tokens"]),
                    "reasoning_tokens": int(totals["reasoning_tokens"]),
                    "cache_read_tokens": int(totals["cache_read_tokens"]),
                    "cache_write_tokens": int(totals["cache_write_tokens"]),
                    "api_calls": int(totals["api_calls"]),
                    "cost": float(totals["cost"]),
                },
                call={"wall_time": round(call.duration, 3)},
            )
            self._append_step(record)
            steps.append(record)
            history.append((observation, call.action))

            meta["next_step"] = step_number + 1
            meta["score"] = int(state["score"])
            meta["highscore"] = int(state["highscore"])
            meta["max_score"] = int(state["max_score"])
            meta.pop("inflight_model_step", None)
            meta.pop("inflight_session_id", None)
            self._write_meta(meta)
            self.pending_path.unlink(missing_ok=True)
            self._maybe_heartbeat(step_number)

            finished = outcome == "break"
            step_number += 1

        return schema.build_trajectory(
            game=self.job.game,
            model=self.job.model,
            seed=self.job.seed,
            max_steps=self.job.max_steps,
            max_score=int(state["max_score"]),
            initial_observation=initial_observation,
            steps=steps,
            reasoning_effort=self.job.reasoning_effort,
            model_family=self.model.provider_family,
            agent_type=self.config.agent_type,
            framework=self.config.framework,
        )

    def _replay(self, env, state: dict[str, Any], steps: list[StepRecord]):
        """Re-apply recorded actions so the emulator matches the step log."""
        self.log(f"Replaying {len(steps)} recorded steps.")
        for record in steps:
            _, feedback = step_env(env, state, record.action)
            terminal_done = bool(state["done"])
            terminal_won = bool(state["info"].get("won"))
            terminal_lost = bool(state["info"].get("lost"))
            outcome = handle_done(env, state, self.effective_seed)

            if feedback != record.feedback:
                self._diverged("feedback", record.step, record.feedback, feedback)
            if int(state["score"]) != int(record.score_after):
                self._diverged("score", record.step, record.score_after, state["score"])
            if (outcome == "reset") != bool(record.did_reset):
                self._diverged("reset", record.step, record.did_reset, outcome)
            if terminal_done != bool(record.done):
                self._diverged("done", record.step, record.done, terminal_done)
            if terminal_won != bool(record.won):
                self._diverged("won", record.step, record.won, terminal_won)
            if terminal_lost != bool(record.lost):
                self._diverged("lost", record.step, record.lost, terminal_lost)

            if self.divergences and self.config.on_divergence == "restart":
                self.log("Divergence detected; discarding partial progress.")
                return []
        return steps

    def _diverged(self, kind: str, step: int, expected: Any, actual: Any) -> None:
        message = (
            f"{kind}@{step}: expected {str(expected)[:120]!r} got {str(actual)[:120]!r}"
        )
        self.divergences.append(message)
        self.log(f"Replay divergence {message}")

    def _maybe_heartbeat(self, step: int) -> None:
        now = time.monotonic()
        if now - self._last_heartbeat >= 30.0:
            self._last_heartbeat = now
            self.heartbeat(step)

    # -- Finalization ----------------------------------------------------

    def _summary(self, record: dict[str, Any], *, duration: float) -> dict[str, Any]:
        summary = {
            "job_id": self.job.job_id,
            "model": self.job.model,
            "game": self.job.game,
            "seed": self.job.seed,
            "reasoning_effort": self.job.reasoning_effort,
            "total_steps": record["total_steps"],
            "score": record["score"],
            "max_score": record["max_score"],
            "normalized_score": record["normalized_score"],
            "duration": duration,
            "trajectory_path": str(self.final_path),
        }
        if self.termination_reason:
            summary["termination_reason"] = self.termination_reason
            summary["termination_step"] = self.termination_step
        return summary

    def _flush_call_telemetry(self) -> None:
        """Calls are persisted incrementally before each environment step."""

    def _finalize(self, record: dict[str, Any], summary: dict[str, Any]) -> None:
        self._flush_call_telemetry()
        calls = finalize_call_telemetry(self.config, self.job)

        from .telemetry import episode_telemetry, write_episode_telemetry

        write_episode_telemetry(
            self.config,
            episode_telemetry(
                job=self.job,
                model=self.model,
                record=record,
                calls=calls,
                summary=summary,
                divergences=self.divergences,
            ),
        )
        storage.atomic_write_json(self.final_path, record)
        storage.remove_tree(self.dir)
