"""Worker loop: lease jobs from the manifest and run them to completion."""

from __future__ import annotations

import multiprocessing as mp
import os
import shutil
import signal
import threading
import time
import traceback
import uuid
from pathlib import Path
from typing import Any, Callable, Sequence

from . import availability, episode, storage
from .config import RunConfig
from .manifest import Job, Manifest, worker_id
from .models import ModelSpec

_STOP = {"requested": False}


def _request_stop(*_: Any) -> None:
    _STOP["requested"] = True


def _is_native_fault_exit(exitcode: int | None) -> bool:
    return exitcode in {
        -signal.SIGABRT,
        -signal.SIGALRM,
        -signal.SIGBUS,
        -signal.SIGFPE,
        -signal.SIGILL,
        -signal.SIGSEGV,
    }


def _model_pause_sleep_seconds(retry_at: float | None) -> float:
    """Stagger paused workers without repeatedly contending on the manifest."""
    now = time.time()
    jitter = (os.getpid() % 3000) / 100.0
    if retry_at is None or retry_at <= now:
        return 30.0 + jitter
    return min(60.0, max(retry_at - now, 0.1) + jitter)


def _sleep_for_model_pause(config: RunConfig, retry_at: float | None) -> None:
    """Sleep with low manifest churn while still observing STOP promptly."""
    deadline = time.monotonic() + _model_pause_sleep_seconds(retry_at)
    while not _STOP["requested"] and not config.stop_file.exists():
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            return
        time.sleep(min(2.0, remaining))


class WorkerLogger:
    def __init__(self, path: Path | None, *, echo: bool = False) -> None:
        self.path = path
        self.echo = echo
        if path is not None:
            path.parent.mkdir(parents=True, exist_ok=True)

    def __call__(self, message: str) -> None:
        line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {message}"
        if self.path is not None:
            with open(self.path, "a", encoding="utf-8") as handle:
                handle.write(line + "\n")
        if self.echo:
            print(line, flush=True)


class LeaseLost(RuntimeError):
    """Another worker took ownership of the current job."""


class LeaseHeartbeat:
    """Extend a job lease independently of long model calls."""

    def __init__(
        self,
        config: RunConfig,
        job: Job,
        identity: str,
        log: WorkerLogger,
    ) -> None:
        self.config = config
        self.job = job
        self.identity = identity
        self.log = log
        self.steps = max(job.steps_completed, _steps_on_disk(config, job))
        self.lost = False
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def __enter__(self) -> "LeaseHeartbeat":
        self._thread.start()
        return self

    def __exit__(self, *exc: Any) -> None:
        self._stop.set()
        self._thread.join(timeout=10)

    def __call__(self, step: int) -> None:
        self.steps = max(self.steps, int(step))
        self.ensure_owned()

    def ensure_owned(self) -> None:
        if self.lost:
            raise LeaseLost(f"Lease ownership lost for {self.job.job_id}")

    def _run(self) -> None:
        interval = min(60.0, max(5.0, self.config.lease_seconds / 4.0))
        failure_limit = max(
            interval,
            min(interval * 3.0, self.config.lease_seconds / 2.0),
        )
        last_success = time.monotonic()
        manifest = Manifest(self.config.manifest_path)
        try:
            while not self._stop.wait(interval):
                try:
                    owned = manifest.heartbeat(
                        self.job.job_id,
                        worker=self.identity,
                        steps_completed=self.steps,
                        lease_seconds=self.config.lease_seconds,
                    )
                except Exception as exc:  # transient SQLite contention
                    self.log(f"Lease heartbeat failed temporarily: {exc!r}")
                    if time.monotonic() - last_success >= failure_limit:
                        self.lost = True
                        self.log(
                            f"Lease heartbeat failed for at least "
                            f"{failure_limit:.0f}s; treating ownership of "
                            f"{self.job.job_id} as lost."
                        )
                        return
                    continue
                if not owned:
                    self.lost = True
                    self.log(f"Lease ownership lost for {self.job.job_id}.")
                    return
                last_success = time.monotonic()
        finally:
            manifest.close()


def load_model_specs(config: RunConfig) -> dict[str, ModelSpec]:
    payload = storage.read_json(config.output_root / "models.json")
    if not payload:
        raise FileNotFoundError(
            f"{config.output_root / 'models.json'} is missing; run `plan` first."
        )
    specs = [ModelSpec.from_dict(item) for item in payload["models"]]
    return {spec.id: spec for spec in specs}


def _make_bridge(config: RunConfig, log_path: Path | None):
    if config.fake_model:
        from .fakes import FakeBridge

        return FakeBridge(config, log_path=log_path)

    from .bridge import CopilotBridge

    return CopilotBridge(config, log_path=log_path)


def run_worker(
    config: RunConfig,
    index: int = 0,
    *,
    models: Sequence[str] | None = None,
    games: Sequence[str] | None = None,
    seeds: Sequence[int] | None = None,
    max_jobs: int | None = None,
    retry_failed: bool = False,
) -> dict[str, int]:
    """Claim and run jobs until the queue drains, ``max_jobs``, or a stop signal."""
    signal.signal(signal.SIGINT, _request_stop)
    signal.signal(signal.SIGTERM, _request_stop)

    log = WorkerLogger(config.logs_dir / f"worker-{index:02d}.log", echo=config.verbose)
    config.use_scratch_tmpdir()
    specs = load_model_specs(config)
    manifest = Manifest(config.manifest_path)
    identity = worker_id(index)
    bridge = _make_bridge(config, config.logs_dir / f"bridge-{index:02d}.log")

    stats = {"done": 0, "skipped": 0, "failed": 0}
    stopped_for_resume = False
    current: Job | None = None
    selected_models = tuple(dict.fromkeys(models if models is not None else specs))
    last_model_pause_log = 0.0
    log(f"worker {identity} started (fake_model={config.fake_model})")

    try:
        while not _STOP["requested"]:
            if config.stop_file.exists():
                stopped_for_resume = True
                log("Stop file present; exiting.")
                break
            free_gb = shutil.disk_usage(config.output_root).free / (1024**3)
            if free_gb < config.min_free_gb:
                storage.atomic_write_text(
                    config.stop_file,
                    f"Automatic stop: only {free_gb:.2f} GiB free; "
                    f"minimum is {config.min_free_gb:.2f} GiB.\n",
                )
                stopped_for_resume = True
                log(
                    f"Only {free_gb:.2f} GiB free; wrote STOP and am exiting "
                    "before the manifest or partial trajectory can be corrupted."
                )
                break
            if max_jobs is not None and sum(stats.values()) >= max_jobs:
                break

            claim_models, model_retry_at, model_pauses = availability.runnable_models(
                config, selected_models
            )
            if not claim_models:
                paused_models = [pause["model"] for pause in model_pauses]
                if not paused_models or not manifest.has_runnable(
                    max_attempts=config.max_attempts,
                    models=paused_models,
                    games=games,
                    seeds=seeds,
                    retry_failed=retry_failed,
                ):
                    log("No claimable jobs left for the selected models.")
                    break
                now = time.time()
                if now - last_model_pause_log >= 60:
                    paused_names = ", ".join(pause["model"] for pause in model_pauses)
                    log(
                        f"All selected models are temporarily paused "
                        f"({paused_names}); next probe after "
                        f"{time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(model_retry_at))}."
                    )
                    last_model_pause_log = now
                _sleep_for_model_pause(config, model_retry_at)
                continue

            job = manifest.claim(
                identity,
                lease_seconds=config.lease_seconds,
                max_attempts=config.max_attempts,
                models=claim_models,
                games=games,
                seeds=seeds,
                retry_failed=retry_failed,
            )
            if job is None:
                paused_models = [pause["model"] for pause in model_pauses]
                if paused_models and manifest.has_runnable(
                    max_attempts=config.max_attempts,
                    models=paused_models,
                    games=games,
                    seeds=seeds,
                    retry_failed=retry_failed,
                ):
                    _sleep_for_model_pause(config, model_retry_at)
                    continue
                log("No claimable jobs left.")
                break

            current = job
            spec = specs.get(job.model)
            if spec is None:
                manifest.fail(
                    job.job_id,
                    f"Unknown model {job.model!r}",
                    worker=identity,
                )
                stats["failed"] += 1
                current = None
                continue

            try:
                with LeaseHeartbeat(config, job, identity, log) as heartbeat:
                    runner = episode.EpisodeRunner(
                        config,
                        bridge,
                        spec,
                        job,
                        log=log,
                        heartbeat=heartbeat,
                    )
                    result = runner.run()
                    heartbeat.ensure_owned()
            except KeyboardInterrupt:
                manifest.release(job.job_id, worker=identity)
                log(f"{job.job_id} interrupted; lease released.")
                break
            except episode.StopRequested:
                manifest.release(job.job_id, worker=identity)
                log(f"{job.job_id} paused at a step boundary by STOP.")
                current = None
                stopped_for_resume = True
                break
            except availability.ModelTemporarilyUnavailable as exc:
                steps = _steps_on_disk(config, job)
                pause = availability.defer_model(
                    config,
                    job.model,
                    exc.error,
                )
                updated = manifest.defer(
                    job.job_id,
                    f"{exc!r}",
                    steps_completed=steps,
                    worker=identity,
                )
                if updated:
                    retry_at = time.strftime(
                        "%Y-%m-%d %H:%M:%S",
                        time.localtime(float(pause["until"])),
                    )
                    log(
                        f"{job.job_id} deferred after {steps} steps without "
                        f"consuming attempt {job.attempts}; {job.model} will be "
                        f"probed again after {retry_at}: {exc.error}"
                    )
                else:
                    log(
                        f"{job.job_id} became unavailable after its lease was "
                        "reassigned."
                    )
                current = None
                continue
            except LeaseLost as exc:
                log(str(exc))
                try:
                    manifest.release(job.job_id, worker=identity)
                except Exception as release_error:
                    log(f"Could not release uncertain lease: {release_error!r}")
                current = None
                continue
            except Exception as exc:
                steps = _steps_on_disk(config, job)
                retry = job.attempts < config.max_attempts
                updated = manifest.fail(
                    job.job_id,
                    f"{exc!r}",
                    steps_completed=steps,
                    worker=identity,
                    retry=retry,
                )
                if not updated:
                    log(f"{job.job_id} failed after its lease was reassigned.")
                    current = None
                    continue
                if retry:
                    log(
                        f"{job.job_id} attempt {job.attempts}/"
                        f"{config.max_attempts} failed after {steps} steps; requeued: "
                        f"{exc!r}"
                    )
                else:
                    stats["failed"] += 1
                    log(f"{job.job_id} failed after {steps} steps: {exc!r}")
                log(traceback.format_exc())
                current = None
                continue

            if not manifest.complete(job.job_id, result.summary, worker=identity):
                log(f"{job.job_id} finished after its lease was reassigned.")
                current = None
                continue
            stats["skipped" if result.skipped else "done"] += 1
            log(
                f"{job.job_id} {'skipped (already done)' if result.skipped else 'done'}"
                f" steps={result.summary['total_steps']}"
                f" score={result.summary['score']}/{result.summary['max_score']}"
                f" duration={result.summary['duration']:.1f}s"
            )
            current = None
    finally:
        if current is not None:
            manifest.release(current.job_id, worker=identity)
        try:
            bridge.stop()
        except Exception:  # pragma: no cover - shutdown is best effort
            pass
        manifest.close()
        stats["stop_requested"] = int(stopped_for_resume or _STOP["requested"])
        log(f"worker {identity} finished: {stats}")

    return stats


def _steps_on_disk(config: RunConfig, job: Job) -> int:
    steps_path = episode.partial_dir(config, job) / "steps.jsonl"
    records, _ = storage.read_jsonl_prefix(steps_path)
    return len(records)


def _quarantine_pending_action(
    config: RunConfig,
    job: Job,
    *,
    worker: str,
    reason: str,
) -> Path | None:
    """Remove a pending action that may have killed the native interpreter."""
    pending_path = episode.partial_dir(config, job) / "pending.json"
    pending = storage.read_json(pending_path)
    if not isinstance(pending, dict):
        return None

    step = int(pending.get("step") or (_steps_on_disk(config, job) + 1))
    quarantine_dir = episode.native_crash_dir(config, job)
    quarantine_dir.mkdir(parents=True, exist_ok=True)
    quarantine_path = quarantine_dir / (
        f"pending-step-{step:03d}-attempt-{job.attempts:02d}-" f"{time.time_ns()}.json"
    )
    os.replace(pending_path, quarantine_path)

    pending["native_crash"] = {
        "worker": worker,
        "reason": reason,
        "quarantined_at": time.time(),
    }
    storage.atomic_write_json(quarantine_path, pending)

    meta_path = pending_path.parent / "meta.json"
    meta = storage.read_json(meta_path)
    if isinstance(meta, dict):
        pending_session_id = pending.get("session_id")
        inflight_session_id = meta.get("inflight_session_id")
        inflight_step = meta.get("inflight_model_step")
        pending_matches_inflight = (
            inflight_step is not None
            and int(inflight_step) == step
            and pending_session_id is not None
            and pending_session_id == inflight_session_id
        )
        sessions = meta.get("sessions")
        if isinstance(sessions, list):
            reasons = {
                pending_session_id: "native-crash-pending-action",
            }
            if inflight_session_id and not pending_matches_inflight:
                reasons[inflight_session_id] = "native-crash-inflight-call"
            for session in sessions:
                session_id = session.get("session_id")
                if session_id in reasons:
                    session["resumable"] = False
                    session["ended_at"] = time.time()
                    session["end_reason"] = reasons[session_id]
        if pending_matches_inflight:
            meta.pop("inflight_model_step", None)
            meta.pop("inflight_session_id", None)
        storage.atomic_write_json(meta_path, meta)

    usage = pending.get("usage") or []
    entry: dict[str, Any] = {
        "step": step,
        "status": "native_crash_pending_action",
        "session_id": pending.get("session_id"),
        "model": job.model,
        "reasoning_effort": job.reasoning_effort,
        "timestamp": time.time(),
        "action": pending.get("action", ""),
        "error": reason[:500],
    }
    source_call_id = pending.get("call_id")
    call_recorded = source_call_id and any(
        row.get("call_id") == source_call_id
        for row in episode.read_call_telemetry(config, job)
    )
    if call_recorded:
        entry["source_call_id"] = source_call_id
        entry["api_calls"] = 0
    else:
        entry["call_id"] = source_call_id or uuid.uuid4().hex
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
    episode.append_call_telemetry(config, job, entry)
    return quarantine_path


def _quarantine_crashed_jobs(
    config: RunConfig,
    jobs: Sequence[Job],
    *,
    worker: str,
    reason: str,
    log,
) -> set[str]:
    quarantined_job_ids: set[str] = set()
    for job in jobs:
        try:
            quarantined = _quarantine_pending_action(
                config,
                job,
                worker=worker,
                reason=reason,
            )
        except Exception as exc:
            log(f"Could not quarantine pending action for {job.job_id}: {exc!r}")
            continue
        if quarantined is not None:
            pending = storage.read_json(quarantined)
            step = int(pending.get("step") or -1) if isinstance(pending, dict) else -1
            limit = _native_fault_limit(config, job, step=step)
            if limit is not None:
                try:
                    storage.atomic_write_json(
                        episode.native_crash_limit_path(config, job),
                        limit,
                    )
                except Exception as exc:
                    log(
                        f"Could not record native fault limit for "
                        f"{job.job_id}: {exc!r}"
                    )
                    continue
                log(
                    f"{job.job_id} reached its native fault limit; the next "
                    "worker will finalize the partial trajectory."
                )
            quarantined_job_ids.add(job.job_id)
            log(
                f"Quarantined pending action for {job.job_id} "
                f"after native worker failure: {quarantined}"
            )
    return quarantined_job_ids


def _native_fault_limit(
    config: RunConfig,
    job: Job,
    *,
    step: int,
) -> dict[str, Any] | None:
    step_limit = max(1, config.max_worker_restarts)
    total_limit = step_limit * max(1, config.max_attempts)
    total_faults = 0
    step_faults = 0
    for path in episode.native_crash_dir(config, job).glob("pending-*.json"):
        pending = storage.read_json(path)
        if not isinstance(pending, dict):
            continue
        total_faults += 1
        if int(pending.get("step") or -1) == step:
            step_faults += 1
    if step_faults < step_limit and total_faults < total_limit:
        return None
    return {
        "job_id": job.job_id,
        "step": step,
        "step_faults": step_faults,
        "step_limit": step_limit,
        "total_faults": total_faults,
        "total_limit": total_limit,
        "created_at": time.time(),
    }


def _native_fault_is_recoverable(
    jobs: Sequence[Job],
    quarantined_job_ids: set[str],
) -> bool:
    return bool(jobs) and quarantined_job_ids == {job.job_id for job in jobs}


def _worker_entry(payload: dict[str, Any]) -> dict[str, int]:
    config = RunConfig.from_dict(payload["config"])
    return run_worker(
        config,
        payload["index"],
        models=payload.get("models"),
        games=payload.get("games"),
        seeds=payload.get("seeds"),
        max_jobs=payload.get("max_jobs"),
        retry_failed=payload.get("retry_failed", False),
    )


def _supervised_worker_entry(payload: dict[str, Any], result_path: str) -> None:
    try:
        result = _worker_entry(payload)
        storage.atomic_write_json(
            Path(result_path),
            {"index": payload["index"], "ok": True, "result": result},
        )
    except BaseException:
        storage.atomic_write_json(
            Path(result_path),
            {
                "index": payload["index"],
                "ok": False,
                "error": traceback.format_exc(),
            },
        )
        raise


def _resume_paused_slots(
    stop_file: Path,
    paused_slots: set[int],
    start_worker: Callable[[int], None],
) -> None:
    """Refill slots that stopped if STOP disappears before the pool exits."""
    if stop_file.exists():
        return
    for index in sorted(paused_slots):
        start_worker(index)
    paused_slots.clear()


def run_pool(
    config: RunConfig,
    *,
    models: Sequence[str] | None = None,
    games: Sequence[str] | None = None,
    seeds: Sequence[int] | None = None,
    max_jobs: int | None = None,
    retry_failed: bool = False,
) -> list[dict[str, int]]:
    """Fan the worker loop out over ``config.concurrency`` processes."""
    config.ensure_dirs()
    if config.stop_file.exists():
        if os.environ.get("TALES_PRESERVE_STOP") == "1":
            return []
        config.stop_file.unlink()

    indices = range(
        config.worker_index_offset,
        config.worker_index_offset + config.concurrency,
    )
    payloads = {
        index: {
            "config": config.to_dict(),
            "index": index,
            "models": list(models) if models else None,
            "games": list(games) if games else None,
            "seeds": list(seeds) if seeds else None,
            "max_jobs": max_jobs,
            "retry_failed": retry_failed,
        }
        for index in indices
    }

    context = mp.get_context("spawn")
    active: dict[int, mp.Process] = {}
    paused_slots: set[int] = set()
    restarts = {index: 0 for index in indices}
    results: list[dict[str, int]] = []
    fatal_errors: list[str] = []
    worker_errors: dict[int, str] = {}
    result_paths = {
        index: config.scratch_dir / f"worker-result-{os.getpid()}-{index}.json"
        for index in indices
    }

    def start_worker(index: int) -> None:
        result_paths[index].unlink(missing_ok=True)
        process = context.Process(
            target=_supervised_worker_entry,
            args=(payloads[index], str(result_paths[index])),
            name=f"copilot-eval-{index:02d}",
        )
        process.start()
        active[index] = process

    for index in indices:
        start_worker(index)

    try:
        while active:
            _resume_paused_slots(config.stop_file, paused_slots, start_worker)
            for index, process in list(active.items()):
                if process.is_alive():
                    continue
                process.join()
                del active[index]
                message = storage.read_json(result_paths[index])
                result_paths[index].unlink(missing_ok=True)
                if message:
                    if message["ok"]:
                        results.append(message["result"])
                    else:
                        worker_errors[index] = message["error"]
                stopped_for_resume = bool(
                    message
                    and message.get("ok") is True
                    and message["result"].get("stop_requested")
                )
                if stopped_for_resume:
                    paused_slots.add(index)
                    _resume_paused_slots(
                        config.stop_file,
                        paused_slots,
                        start_worker,
                    )
                    continue
                if process.exitcode == 0 or (message and message.get("ok") is True):
                    continue

                identity = worker_id(index, pid=process.pid)
                reason = (
                    f"Worker {identity} exited unexpectedly with "
                    f"code {process.exitcode}"
                )
                with Manifest(config.manifest_path) as manifest:
                    crashed_jobs = manifest.running_jobs_for_worker(identity)
                    quarantined_job_ids: set[str] = set()
                    if _is_native_fault_exit(process.exitcode):
                        quarantined_job_ids = _quarantine_crashed_jobs(
                            config,
                            crashed_jobs,
                            worker=identity,
                            reason=reason,
                            log=WorkerLogger(
                                config.logs_dir / f"worker-{index:02d}.log"
                            ),
                        )
                    recoverable_native_fault = _native_fault_is_recoverable(
                        crashed_jobs,
                        quarantined_job_ids,
                    )
                    manifest.requeue_worker(
                        identity,
                        max_attempts=config.max_attempts,
                        reason=reason,
                        preserve_attempt_budget=recoverable_native_fault,
                    )
                if config.stop_file.exists():
                    continue
                if not recoverable_native_fault:
                    restarts[index] += 1
                if crashed_jobs and (
                    recoverable_native_fault
                    or restarts[index] <= config.max_worker_restarts
                ):
                    start_worker(index)
                elif crashed_jobs:
                    fatal_errors.append(
                        f"Worker slot {index} exceeded "
                        f"{config.max_worker_restarts} restarts.\n"
                        f"{worker_errors.get(index, '')}"
                    )
                else:
                    fatal_errors.append(
                        f"{reason} after runnable jobs were exhausted.\n"
                        f"{worker_errors.get(index, '')}"
                    )
            if active:
                time.sleep(0.5)
    except KeyboardInterrupt:  # pragma: no cover - interactive only
        storage.atomic_write_text(config.stop_file, "Interrupted by operator.\n")
        for process in active.values():
            process.join(timeout=10)
            if process.is_alive():
                process.terminate()
        for process in active.values():
            process.join(timeout=10)
        raise
    finally:
        for path in result_paths.values():
            path.unlink(missing_ok=True)

    with Manifest(config.manifest_path) as manifest:
        manifest.mark_exhausted(max_attempts=config.max_attempts)

    if fatal_errors:
        raise RuntimeError(
            "One or more worker processes failed:\n" + "\n".join(fatal_errors[-5:])
        )
    return results


def bootstrap_sys_path() -> None:
    """Make ``copilot_evals`` and ``tales`` importable from spawned workers."""
    import sys

    scripts_dir = Path(__file__).resolve().parents[1]
    repo_root = scripts_dir.parent
    for path in (str(scripts_dir), str(repo_root)):
        if path not in sys.path:
            sys.path.insert(0, path)
    existing = [p for p in os.environ.get("PYTHONPATH", "").split(os.pathsep) if p]
    missing = [p for p in (str(scripts_dir), str(repo_root)) if p not in existing]
    if missing:
        os.environ["PYTHONPATH"] = os.pathsep.join([*missing, *existing])
