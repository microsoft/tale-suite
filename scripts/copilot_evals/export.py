"""Streaming export to ``raw/jericho.jsonl`` and ``data/jericho/train.parquet``.

Trajectories are read one file at a time and flushed in batches, so exporting
95k+ episodes never materialises the whole dataset in memory.
"""

from __future__ import annotations

import gzip
import json
from pathlib import Path
from typing import Any, Iterator

from . import schema, storage
from .config import RunConfig


def arrow_schema():
    import pyarrow as pa

    return pa.schema(
        [
            pa.field("transcript_id", pa.string()),
            pa.field("framework", pa.string()),
            pa.field("game", pa.string()),
            pa.field("model", pa.string()),
            pa.field("agent_type", pa.string()),
            pa.field("total_steps", pa.int64()),
            pa.field("score", pa.int64()),
            pa.field("max_score", pa.int64()),
            pa.field("normalized_score", pa.float64()),
            pa.field("turns", pa.string()),
            pa.field("score_progression", pa.string()),
            pa.field("thinking_available", pa.bool_()),
            pa.field("total_thinking_tokens", pa.int64()),
            pa.field("llm", pa.string()),
            pa.field("seed", pa.int64()),
            pa.field("reasoning_effort", pa.string()),
            pa.field("max_steps", pa.int64()),
            # ``tags`` is a list<null> upstream; keep the published dtype.
            pa.field("tags", pa.list_(pa.null())),
            pa.field("episode_token_usage", pa.float64()),
            pa.field("model_family", pa.string()),
        ]
    )


def iter_trajectory_files(root: Path) -> Iterator[Path]:
    yield from sorted(
        path
        for path in Path(root).rglob("*")
        if path.is_file()
        and (path.name.endswith(".json") or path.name.endswith(".json.gz"))
    )


def iter_records(trajectories_dir: Path, *, on_error=None) -> Iterator[dict[str, Any]]:
    for path in iter_trajectory_files(trajectories_dir):
        record = storage.read_json(path)
        if record is None:
            if on_error:
                on_error(path, "unreadable or truncated JSON")
            continue
        try:
            yield schema.validate_trajectory(record)
        except schema.SchemaError as exc:
            if on_error:
                on_error(path, str(exc))


def records_to_batch(records: list[dict[str, Any]]):
    import pyarrow as pa

    table_schema = arrow_schema()
    columns = {name: [] for name in schema.HF_COLUMNS}
    for record in records:
        for name in schema.HF_COLUMNS:
            columns[name].append(record[name])
    arrays = [pa.array(columns[field.name], type=field.type) for field in table_schema]
    return pa.RecordBatch.from_arrays(arrays, schema=table_schema)


def export(
    config: RunConfig,
    *,
    batch_size: int = 200,
    limit: int | None = None,
    expected_rows: int | None = None,
    jsonl_path: Path | None = None,
    parquet_path: Path | None = None,
) -> dict[str, Any]:
    """Write the JSONL + Parquet dataset. Returns export statistics."""
    import pyarrow.parquet as pq

    config.ensure_dirs()
    jsonl_path = jsonl_path or config.raw_dir / f"{config.framework}.jsonl.gz"
    parquet_path = parquet_path or config.data_dir / "train.parquet"
    jsonl_path.parent.mkdir(parents=True, exist_ok=True)
    parquet_path.parent.mkdir(parents=True, exist_ok=True)

    errors: list[dict[str, str]] = []

    def note(path: Path, message: str) -> None:
        errors.append({"path": str(path), "error": message})

    tmp_jsonl = jsonl_path.with_name(f".{jsonl_path.name}.tmp")
    tmp_parquet = parquet_path.with_suffix(parquet_path.suffix + ".tmp")

    exported = 0
    batch: list[dict[str, Any]] = []
    writer = None
    try:
        raw_opener = gzip.open if jsonl_path.suffix == ".gz" else open
        with raw_opener(tmp_jsonl, "wt", encoding="utf-8") as handle:
            writer = pq.ParquetWriter(tmp_parquet, arrow_schema(), compression="zstd")
            for record in iter_records(config.trajectories_dir, on_error=note):
                handle.write(json.dumps(record) + "\n")
                batch.append(record)
                exported += 1
                if len(batch) >= batch_size:
                    writer.write_batch(records_to_batch(batch))
                    batch.clear()
                if limit is not None and exported >= limit:
                    break
            if batch:
                writer.write_batch(records_to_batch(batch))
                batch.clear()
    finally:
        if writer is not None:
            writer.close()

    tmp_jsonl.replace(jsonl_path)
    tmp_parquet.replace(parquet_path)
    output_validation = verify_outputs(
        jsonl_path,
        parquet_path,
        expected_rows=exported if expected_rows is None else expected_rows,
    )

    stats = {
        "exported": exported,
        "errors": errors[:50],
        "error_count": len(errors),
        "jsonl": str(jsonl_path),
        "parquet": str(parquet_path),
        "output_validation": output_validation,
    }
    storage.atomic_write_json(
        config.output_root / "export_report.json", stats, indent=2
    )
    return stats


def verify_outputs(
    jsonl_path: Path,
    parquet_path: Path,
    *,
    expected_rows: int,
) -> dict[str, Any]:
    """Verify final export row counts and the exact published schema."""
    import pyarrow.parquet as pq

    jsonl_rows = 0
    jsonl_columns = None
    with gzip.open(jsonl_path, "rt", encoding="utf-8") as handle:
        for line in handle:
            jsonl_rows += 1
            if jsonl_columns is None:
                jsonl_columns = list(json.loads(line))

    parquet = pq.ParquetFile(parquet_path)
    parquet_rows = parquet.metadata.num_rows
    parquet_schema = parquet.schema_arrow
    expected_columns = list(schema.HF_COLUMNS)
    jsonl_schema_matches = (
        jsonl_columns == expected_columns if jsonl_rows else expected_rows == 0
    )
    parquet_schema_matches = parquet_schema.equals(
        arrow_schema(),
        check_metadata=False,
    )
    valid = (
        jsonl_rows == expected_rows
        and parquet_rows == expected_rows
        and jsonl_schema_matches
        and parquet_schema_matches
    )
    return {
        "valid": valid,
        "expected_rows": expected_rows,
        "jsonl_rows": jsonl_rows,
        "parquet_rows": parquet_rows,
        "jsonl_columns": jsonl_columns,
        "jsonl_schema_matches": jsonl_schema_matches,
        "parquet_columns": parquet_schema.names,
        "parquet_schema_matches": parquet_schema_matches,
    }


def verify(config: RunConfig, *, limit: int | None = None) -> dict[str, Any]:
    """Validate every trajectory on disk against the HF schema."""
    if limit is None and config.manifest_path.exists():
        return verify_manifest_trajectories(config)

    checked = 0
    errors: list[dict[str, str]] = []

    def note(path: Path, message: str) -> None:
        errors.append({"path": str(path), "error": message})

    for _ in iter_records(config.trajectories_dir, on_error=note):
        checked += 1
        if limit is not None and checked >= limit:
            break
    return {
        "valid": checked,
        "invalid": len(errors),
        "errors": errors[:50],
        "manifest_reconciled": False,
    }


def verify_manifest_trajectories(config: RunConfig) -> dict[str, Any]:
    """Reconcile every completed trajectory with its manifest job."""
    from .episode import trajectory_path
    from .manifest import Job, Manifest

    errors: list[dict[str, str]] = []
    valid = 0
    with Manifest(config.manifest_path) as manifest:
        counts = manifest.status_counts()
        rows = manifest.conn.execute(
            "SELECT * FROM jobs WHERE status='done' ORDER BY job_id"
        ).fetchall()

    expected_paths: set[Path] = set()
    for row in rows:
        job = Job.from_row(row)
        path = trajectory_path(config, job)
        expected_paths.add(path)
        record = storage.read_json(path)
        if not isinstance(record, dict):
            errors.append({"path": str(path), "error": "missing or unreadable JSON"})
            continue

        try:
            schema.validate_trajectory(record)
        except schema.SchemaError as exc:
            errors.append({"path": str(path), "error": str(exc)})
            continue

        mismatches = []
        expected_identity = {
            "transcript_id": job.job_id,
            "framework": config.framework,
            "game": job.game,
            "model": job.model,
            "agent_type": config.agent_type,
            "llm": job.model,
            "seed": job.seed,
            "reasoning_effort": job.reasoning_effort,
            "max_steps": job.max_steps,
            "total_steps": int(row["steps_completed"] or 0),
            "score": row["score"],
            "max_score": row["max_score"],
            "normalized_score": row["normalized_score"],
        }
        for field, expected in expected_identity.items():
            if record[field] != expected:
                mismatches.append(f"{field}={record[field]!r}, expected {expected!r}")
        if row["trajectory_path"] != str(path):
            mismatches.append(
                f"manifest trajectory_path={row['trajectory_path']!r}, "
                f"expected {str(path)!r}"
            )
        if mismatches:
            errors.append({"path": str(path), "error": "; ".join(mismatches)})
            continue
        valid += 1

    drained = (
        counts["running"] == 0 and counts["pending"] == 0 and counts["failed"] == 0
    )
    trajectory_files = None
    if drained:
        actual_paths = set(iter_trajectory_files(config.trajectories_dir))
        trajectory_files = len(actual_paths)
        for path in sorted(actual_paths - expected_paths):
            errors.append(
                {
                    "path": str(path),
                    "error": "trajectory is not a completed manifest job",
                }
            )
        if len(expected_paths) != counts["total"]:
            errors.append(
                {
                    "path": str(config.manifest_path),
                    "error": (
                        f"manifest has {len(expected_paths)} completed paths, "
                        f"expected {counts['total']}"
                    ),
                }
            )

    return {
        "valid": valid,
        "invalid": len(errors),
        "errors": errors[:50],
        "manifest_reconciled": True,
        "manifest_counts": counts,
        "trajectory_files": trajectory_files,
    }
