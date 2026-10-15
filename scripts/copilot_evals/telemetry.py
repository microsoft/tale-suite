"""Model-call telemetry kept next to (but out of) the 20-column HF rows."""

from __future__ import annotations

import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

from . import storage
from .config import RunConfig
from .models import slugify

_SUM_KEYS = (
    "input_tokens",
    "output_tokens",
    "reasoning_tokens",
    "cache_read_tokens",
    "cache_write_tokens",
    "duration_ms",
    "api_calls",
    "cost",
    "aiu_nano",
)


def episodes_path(config: RunConfig) -> Path:
    return config.telemetry_dir / "episodes.jsonl.gz"


def episodes_work_path(config: RunConfig) -> Path:
    return episodes_path(config).with_suffix("")


def episode_records_dir(config: RunConfig) -> Path:
    return config.telemetry_dir / "episodes"


def episode_record_path(config: RunConfig, *, model: str, job_id: str) -> Path:
    return episode_records_dir(config) / slugify(model) / f"{job_id}.json.gz"


def invalidated_episode_path(config: RunConfig, *, model: str, job_id: str) -> Path:
    return (
        config.telemetry_dir
        / "invalidated-episodes"
        / slugify(model)
        / f"{job_id}.json"
    )


def aggregate_path(config: RunConfig) -> Path:
    return config.telemetry_dir / "aggregate.json"


def episode_telemetry(
    *,
    job,
    model,
    record: dict[str, Any],
    calls: Iterable[dict[str, Any]],
    summary: dict[str, Any],
    divergences: Iterable[str],
) -> dict[str, Any]:
    calls = list(calls)
    totals = {key: 0 for key in _SUM_KEYS}
    statuses: dict[str, int] = defaultdict(int)
    for call in calls:
        statuses[call.get("status", "unknown")] += 1
        for key in _SUM_KEYS:
            totals[key] += call.get(key) or 0

    return {
        "job_id": job.job_id,
        "model": job.model,
        "model_family": model.provider_family,
        "reasoning_effort": job.reasoning_effort,
        "game": job.game,
        "seed": job.seed,
        "attempts": job.attempts,
        "total_steps": record["total_steps"],
        "max_steps": record["max_steps"],
        "score": record["score"],
        "max_score": record["max_score"],
        "normalized_score": record["normalized_score"],
        "thinking_available": record["thinking_available"],
        "total_thinking_tokens": record["total_thinking_tokens"],
        "episode_token_usage": record["episode_token_usage"],
        "duration": summary.get("duration"),
        "termination_reason": summary.get("termination_reason"),
        "termination_step": summary.get("termination_step"),
        "call_statuses": dict(statuses),
        "divergences": list(divergences),
        "finished_at": time.time(),
        **{f"total_{key}": totals[key] for key in _SUM_KEYS},
    }


def write_episode_telemetry_shard(config: RunConfig, payload: dict[str, Any]) -> None:
    storage.atomic_write_json(
        episode_record_path(
            config,
            model=payload["model"],
            job_id=payload["job_id"],
        ),
        payload,
    )
    invalidated_episode_path(
        config,
        model=payload["model"],
        job_id=payload["job_id"],
    ).unlink(missing_ok=True)


def write_episode_telemetry(config: RunConfig, payload: dict[str, Any]) -> None:
    write_episode_telemetry_shard(config, payload)
    storage.append_jsonl(episodes_work_path(config), payload)


def remove_episode_telemetry(
    config: RunConfig,
    *,
    model: str,
    job_id: str,
) -> None:
    storage.atomic_write_json(
        invalidated_episode_path(config, model=model, job_id=job_id),
        {
            "job_id": job_id,
            "model": model,
            "invalidated_at": time.time(),
        },
    )
    episode_record_path(config, model=model, job_id=job_id).unlink(missing_ok=True)


def invalidated_episode_ids(config: RunConfig) -> set[str]:
    ids: set[str] = set()
    for path in (config.telemetry_dir / "invalidated-episodes").glob("*/*.json"):
        payload = storage.read_json(path)
        if isinstance(payload, dict) and payload.get("job_id"):
            ids.add(payload["job_id"])
    return ids


def find_shared_episode(config: RunConfig, job_id: str) -> dict[str, Any] | None:
    if job_id in invalidated_episode_ids(config):
        return None
    found = None
    for path in (episodes_path(config), episodes_work_path(config)):
        for payload in storage.iter_jsonl(path):
            if isinstance(payload, dict) and payload.get("job_id") == job_id:
                found = payload
    return found


def iter_episode_telemetry(config: RunConfig):
    invalidated = invalidated_episode_ids(config)
    seen: set[str] = set()
    for path in sorted(episode_records_dir(config).glob("*/*.json.gz")):
        payload = storage.read_json(path)
        if not isinstance(payload, dict) or not payload.get("job_id"):
            continue
        if payload["job_id"] in invalidated:
            continue
        seen.add(payload["job_id"])
        yield payload

    fallback: dict[str, dict[str, Any]] = {}
    for path in (episodes_path(config), episodes_work_path(config)):
        for payload in storage.iter_jsonl(path):
            if not isinstance(payload, dict):
                continue
            job_id = payload.get("job_id")
            if job_id:
                fallback[job_id] = payload
    for job_id, payload in fallback.items():
        if job_id not in seen and job_id not in invalidated:
            yield payload


def finalize_episode_stream(
    config: RunConfig,
    episodes: list[dict[str, Any]],
) -> None:
    storage.rewrite_jsonl(episodes_path(config), episodes)
    episodes_work_path(config).unlink(missing_ok=True)


def manifest_snapshot(config: RunConfig):
    if not config.manifest_path.exists():
        return None

    from .manifest import Manifest

    with Manifest(config.manifest_path) as manifest:
        return {
            "counts": manifest.status_counts(),
            "rows": manifest.conn.execute(
                "SELECT * FROM jobs WHERE status='done' ORDER BY job_id"
            ).fetchall(),
        }


def reconcile_manifest(
    config: RunConfig,
    episodes: list[dict[str, Any]],
    snapshot,
) -> dict[str, Any]:
    if snapshot is None:
        return {
            "enabled": False,
            "valid": True,
            "drained": True,
            "error_count": 0,
            "errors": [],
        }

    counts = snapshot["counts"]
    drained = (
        counts["running"] == 0 and counts["pending"] == 0 and counts["failed"] == 0
    )
    if drained:
        from .episode import telemetry_path, telemetry_work_path
        from .manifest import Job

    by_job: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for episode in episodes:
        by_job[str(episode.get("job_id"))].append(episode)

    errors = []
    valid = 0
    valid_call_logs = 0
    expected_ids = set()
    for row in snapshot["rows"]:
        job_id = row["job_id"]
        expected_ids.add(job_id)
        matches = by_job.get(job_id, [])
        if len(matches) != 1:
            errors.append(
                {
                    "job_id": job_id,
                    "error": f"expected one episode telemetry row, found {len(matches)}",
                }
            )
            continue

        episode = matches[0]
        expected_identity = {
            "job_id": job_id,
            "model": row["model"],
            "game": row["game"],
            "seed": row["seed"],
            "reasoning_effort": row["reasoning_effort"],
            "attempts": row["attempts"],
            "total_steps": row["steps_completed"],
            "score": row["score"],
            "max_score": row["max_score"],
            "normalized_score": row["normalized_score"],
        }
        mismatches = [
            f"{field}={episode.get(field)!r}, expected {expected!r}"
            for field, expected in expected_identity.items()
            if episode.get(field) != expected
        ]
        if mismatches:
            errors.append({"job_id": job_id, "error": "; ".join(mismatches)})
            continue

        if drained:
            job = Job.from_row(row)
            final_path = telemetry_path(config, job)
            work_path = telemetry_work_path(config, job)
            if work_path.exists():
                errors.append(
                    {
                        "job_id": job_id,
                        "error": f"working call telemetry still exists: {work_path}",
                    }
                )
                continue
            if not final_path.exists():
                errors.append(
                    {
                        "job_id": job_id,
                        "error": f"finalized call telemetry is missing: {final_path}",
                    }
                )
                continue

            calls, damaged = storage.read_jsonl_prefix(final_path)
            if damaged:
                errors.append(
                    {
                        "job_id": job_id,
                        "error": f"finalized call telemetry is damaged: {final_path}",
                    }
                )
                continue
            if any(not isinstance(call, dict) for call in calls):
                errors.append(
                    {
                        "job_id": job_id,
                        "error": "finalized call telemetry contains a non-object row",
                    }
                )
                continue

            call_statuses: dict[str, int] = defaultdict(int)
            call_totals = {key: 0 for key in _SUM_KEYS}
            for call in calls:
                call_statuses[call.get("status", "unknown")] += 1
                for key in _SUM_KEYS:
                    call_totals[key] += call.get(key) or 0
            call_mismatches = []
            if dict(call_statuses) != episode.get("call_statuses"):
                call_mismatches.append(
                    f"call_statuses={dict(call_statuses)!r}, "
                    f"expected {episode.get('call_statuses')!r}"
                )
            for key, total in call_totals.items():
                episode_total = episode.get(f"total_{key}")
                if total != episode_total:
                    call_mismatches.append(
                        f"total_{key}={total!r}, expected {episode_total!r}"
                    )
            if call_mismatches:
                errors.append({"job_id": job_id, "error": "; ".join(call_mismatches)})
                continue
            valid_call_logs += 1
        valid += 1

    unexpected_ids = sorted(set(by_job) - expected_ids) if drained else []
    for job_id in unexpected_ids:
        errors.append(
            {
                "job_id": job_id,
                "error": "episode telemetry is not a completed manifest job",
            }
        )

    return {
        "enabled": True,
        "valid": not errors,
        "drained": drained,
        "manifest_counts": counts,
        "completed_jobs": len(snapshot["rows"]),
        "episode_rows": len(episodes),
        "valid_jobs": valid,
        "call_logs_checked": len(snapshot["rows"]) if drained else 0,
        "valid_call_logs": valid_call_logs,
        "error_count": len(errors),
        "errors": errors[:50],
    }


def aggregate(config: RunConfig) -> dict[str, Any]:
    """Roll ``telemetry/episodes.jsonl`` up per model and overall."""
    per_model: dict[str, dict[str, Any]] = {}
    overall = {
        "episodes": 0,
        "steps": 0,
        "call_statuses": {},
        **{f"total_{key}": 0 for key in _SUM_KEYS},
    }

    snapshot = manifest_snapshot(config)
    episodes = list(iter_episode_telemetry(config))
    for episode in episodes:
        model = episode.get("model", "unknown")
        bucket = per_model.setdefault(
            model,
            {
                "model": model,
                "model_family": episode.get("model_family"),
                "reasoning_effort": episode.get("reasoning_effort"),
                "episodes": 0,
                "steps": 0,
                "score_sum": 0.0,
                "normalized_score_sum": 0.0,
                "call_statuses": {},
                **{f"total_{key}": 0 for key in _SUM_KEYS},
            },
        )
        bucket["episodes"] += 1
        bucket["steps"] += episode.get("total_steps") or 0
        bucket["score_sum"] += episode.get("score") or 0
        bucket["normalized_score_sum"] += episode.get("normalized_score") or 0.0
        overall["episodes"] += 1
        overall["steps"] += episode.get("total_steps") or 0
        for status, count in (episode.get("call_statuses") or {}).items():
            bucket["call_statuses"][status] = (
                bucket["call_statuses"].get(status, 0) + count
            )
            overall["call_statuses"][status] = (
                overall["call_statuses"].get(status, 0) + count
            )
        for key in _SUM_KEYS:
            bucket[f"total_{key}"] += episode.get(f"total_{key}") or 0
            overall[f"total_{key}"] += episode.get(f"total_{key}") or 0

    for bucket in per_model.values():
        episode_count = max(bucket["episodes"], 1)
        bucket["mean_normalized_score"] = bucket["normalized_score_sum"] / episode_count
        bucket["mean_steps"] = bucket["steps"] / episode_count
        bucket["call_statuses"] = dict(sorted(bucket["call_statuses"].items()))
    overall["call_statuses"] = dict(sorted(overall["call_statuses"].items()))

    payload = {
        "generated_at": time.time(),
        "overall": overall,
        "models": sorted(per_model.values(), key=lambda item: item["model"]),
        "manifest_reconciliation": reconcile_manifest(config, episodes, snapshot),
    }
    reconciliation = payload["manifest_reconciliation"]
    if reconciliation["valid"] and reconciliation["drained"]:
        finalize_episode_stream(config, episodes)
    storage.atomic_write_json(aggregate_path(config), payload, indent=2)
    return payload
