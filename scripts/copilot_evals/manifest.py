"""SQLite job manifest: planning, leasing, resume and status reporting."""

from __future__ import annotations

import hashlib
import os
import socket
import sqlite3
import time
from contextlib import contextmanager
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Iterable, Iterator, Sequence

SCHEMA = """
CREATE TABLE IF NOT EXISTS jobs (
    job_id            TEXT PRIMARY KEY,
    model             TEXT NOT NULL,
    model_slug        TEXT NOT NULL,
    game              TEXT NOT NULL,
    seed              INTEGER NOT NULL,
    reasoning_effort  TEXT NOT NULL,
    max_steps         INTEGER NOT NULL,
    shuffle_key       INTEGER NOT NULL,
    status            TEXT NOT NULL DEFAULT 'pending',
    attempts          INTEGER NOT NULL DEFAULT 0,
    worker            TEXT,
    lease_expires_at  REAL,
    started_at        REAL,
    finished_at       REAL,
    steps_completed   INTEGER NOT NULL DEFAULT 0,
    score             INTEGER,
    max_score         INTEGER,
    normalized_score  REAL,
    duration          REAL,
    error             TEXT,
    trajectory_path   TEXT,
    updated_at        REAL
);
CREATE INDEX IF NOT EXISTS jobs_status_idx ON jobs (status, shuffle_key);
CREATE INDEX IF NOT EXISTS jobs_model_idx ON jobs (model, status);
CREATE INDEX IF NOT EXISTS jobs_game_idx ON jobs (game, status);
CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT);
"""

PENDING = "pending"
RUNNING = "running"
DONE = "done"
FAILED = "failed"
STATUSES = (PENDING, RUNNING, DONE, FAILED)


def worker_id(index: int = 0, *, pid: int | None = None) -> str:
    return f"{socket.gethostname()}:{pid or os.getpid()}:{index}"


def shuffle_key(job_id: str) -> int:
    """Deterministic pseudo-random ordering key, so workers spread over models."""
    digest = hashlib.blake2b(job_id.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "big") & 0x7FFFFFFFFFFFFFFF


@dataclass(frozen=True)
class Job:
    job_id: str
    model: str
    model_slug: str
    game: str
    seed: int
    reasoning_effort: str
    max_steps: int
    attempts: int = 0
    steps_completed: int = 0

    @classmethod
    def from_row(cls, row: sqlite3.Row) -> "Job":
        return cls(
            job_id=row["job_id"],
            model=row["model"],
            model_slug=row["model_slug"],
            game=row["game"],
            seed=int(row["seed"]),
            reasoning_effort=row["reasoning_effort"],
            max_steps=int(row["max_steps"]),
            attempts=int(row["attempts"]),
            steps_completed=int(row["steps_completed"]),
        )


class Manifest:
    """Thin SQLite wrapper; safe for many concurrent worker processes."""

    def __init__(self, path: Path | str, *, timeout: float = 60.0) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(self.path), timeout=timeout)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute("PRAGMA synchronous=NORMAL")
        self.conn.execute("PRAGMA busy_timeout=60000")
        self.conn.executescript(SCHEMA)
        self.conn.commit()

    def close(self) -> None:
        self.conn.close()

    def __enter__(self) -> "Manifest":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()

    @contextmanager
    def _immediate(self) -> Iterator[sqlite3.Connection]:
        try:
            self.conn.execute("BEGIN IMMEDIATE")
            yield self.conn
            self.conn.commit()
        except BaseException:
            if self.conn.in_transaction:
                self.conn.rollback()
            raise

    # -- Planning --------------------------------------------------------

    def add_jobs(self, jobs: Iterable[dict[str, Any]], *, batch: int = 2000) -> int:
        """Insert planned jobs, ignoring ones already present. Returns #added."""
        added = 0
        pending: list[tuple] = []

        def flush() -> int:
            if not pending:
                return 0
            with self._immediate() as conn:
                before = conn.total_changes
                conn.executemany(
                    """
                    INSERT OR IGNORE INTO jobs
                        (job_id, model, model_slug, game, seed, reasoning_effort,
                         max_steps, shuffle_key, status, updated_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'pending', ?)
                    """,
                    pending,
                )
                inserted = conn.total_changes - before
            pending.clear()
            return inserted

        now = time.time()
        for job in jobs:
            pending.append(
                (
                    job["job_id"],
                    job["model"],
                    job["model_slug"],
                    job["game"],
                    int(job["seed"]),
                    job["reasoning_effort"],
                    int(job["max_steps"]),
                    shuffle_key(job["job_id"]),
                    now,
                )
            )
            if len(pending) >= batch:
                added += flush()
        added += flush()
        return added

    def set_meta(self, key: str, value: str) -> None:
        with self._immediate() as conn:
            conn.execute(
                "INSERT INTO meta (key, value) VALUES (?, ?) "
                "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
                (key, value),
            )

    def get_meta(self, key: str) -> str | None:
        row = self.conn.execute(
            "SELECT value FROM meta WHERE key = ?", (key,)
        ).fetchone()
        return row["value"] if row else None

    # -- Leasing ---------------------------------------------------------

    def _filter_clause(
        self,
        models: Sequence[str] | None,
        games: Sequence[str] | None,
        seeds: Sequence[int] | None,
    ) -> tuple[str, list[Any]]:
        clauses: list[str] = []
        params: list[Any] = []

        def add_filter(column: str, values: Sequence[Any] | None) -> None:
            if values is None:
                return
            if not values:
                clauses.append("0")
                return
            clauses.append(f"{column} IN ({','.join('?' * len(values))})")
            params.extend(values)

        add_filter("model", models)
        add_filter("game", games)
        add_filter(
            "seed", tuple(int(seed) for seed in seeds) if seeds is not None else None
        )
        return ("".join(f" AND {clause}" for clause in clauses), params)

    def claim(
        self,
        worker: str,
        *,
        lease_seconds: float = 1800.0,
        max_attempts: int = 5,
        models: Sequence[str] | None = None,
        games: Sequence[str] | None = None,
        seeds: Sequence[int] | None = None,
        retry_failed: bool = False,
        now: float | None = None,
    ) -> Job | None:
        """Atomically lease the next runnable job, recovering stale leases."""
        now = time.time() if now is None else now
        extra, params = self._filter_clause(models, games, seeds)

        # Split by status so the hot path (pending jobs) uses the
        # (status, shuffle_key) index instead of scanning the whole table.
        candidates: list[tuple[str, list[Any]]] = [
            (
                f"SELECT * FROM jobs WHERE status='pending' AND attempts < ?{extra}"
                " ORDER BY shuffle_key LIMIT 1",
                [max_attempts, *params],
            ),
            (
                "SELECT * FROM jobs WHERE status='running'"
                f" AND COALESCE(lease_expires_at, 0) < ? AND attempts < ?{extra}"
                " ORDER BY shuffle_key LIMIT 1",
                [now, max_attempts, *params],
            ),
        ]
        if retry_failed:
            candidates.append(
                (
                    f"SELECT * FROM jobs WHERE status='failed' AND attempts < ?{extra}"
                    " ORDER BY shuffle_key LIMIT 1",
                    [max_attempts, *params],
                )
            )

        with self._immediate() as conn:
            row = None
            for query, query_params in candidates:
                row = conn.execute(query, query_params).fetchone()
                if row is not None:
                    break
            if row is None:
                return None
            conn.execute(
                """
                UPDATE jobs
                   SET status='running', worker=?, attempts=attempts+1,
                       lease_expires_at=?, started_at=?, updated_at=?, error=NULL
                 WHERE job_id=?
                """,
                (worker, now + lease_seconds, now, now, row["job_id"]),
            )
        job = Job.from_row(row)
        return replace(job, attempts=job.attempts + 1)

    def heartbeat(
        self,
        job_id: str,
        *,
        worker: str | None = None,
        steps_completed: int | None = None,
        lease_seconds: float = 1800.0,
        now: float | None = None,
    ) -> bool:
        now = time.time() if now is None else now
        owner_clause = " AND status='running' AND worker=?" if worker else ""
        owner_params = [worker] if worker else []
        with self._immediate() as conn:
            if steps_completed is None:
                cursor = conn.execute(
                    "UPDATE jobs SET lease_expires_at=?, updated_at=? "
                    f"WHERE job_id=?{owner_clause}",
                    (now + lease_seconds, now, job_id, *owner_params),
                )
            else:
                cursor = conn.execute(
                    "UPDATE jobs SET lease_expires_at=?, updated_at=?, "
                    f"steps_completed=? WHERE job_id=?{owner_clause}",
                    (
                        now + lease_seconds,
                        now,
                        int(steps_completed),
                        job_id,
                        *owner_params,
                    ),
                )
            return cursor.rowcount == 1

    def complete(
        self, job_id: str, summary: dict[str, Any], *, worker: str | None = None
    ) -> bool:
        now = time.time()
        owner_clause = " AND status='running' AND worker=?" if worker else ""
        owner_params = [worker] if worker else []
        with self._immediate() as conn:
            cursor = conn.execute(
                f"""
                UPDATE jobs
                   SET status='done', finished_at=?, updated_at=?, error=NULL,
                       steps_completed=?, score=?, max_score=?, normalized_score=?,
                       duration=?, trajectory_path=?, lease_expires_at=NULL
                 WHERE job_id=?{owner_clause}
                """,
                (
                    now,
                    now,
                    int(summary.get("total_steps") or 0),
                    summary.get("score"),
                    summary.get("max_score"),
                    summary.get("normalized_score"),
                    summary.get("duration"),
                    summary.get("trajectory_path"),
                    job_id,
                    *owner_params,
                ),
            )
            return cursor.rowcount == 1

    def fail(
        self,
        job_id: str,
        error: str,
        *,
        steps_completed: int = 0,
        worker: str | None = None,
        retry: bool = False,
    ) -> bool:
        now = time.time()
        status = PENDING if retry else FAILED
        finished_at = None if retry else now
        owner_clause = " AND status='running' AND worker=?" if worker else ""
        owner_params = [worker] if worker else []
        with self._immediate() as conn:
            cursor = conn.execute(
                f"""
                UPDATE jobs
                   SET status=?, finished_at=?, updated_at=?, error=?,
                       steps_completed=MAX(steps_completed, ?), lease_expires_at=NULL
                 WHERE job_id=?{owner_clause}
                """,
                (
                    status,
                    finished_at,
                    now,
                    error[:4000],
                    int(steps_completed),
                    job_id,
                    *owner_params,
                ),
            )
            return cursor.rowcount == 1

    def release(self, job_id: str, *, worker: str | None = None) -> bool:
        """Return a leased job to the pool (graceful shutdown)."""
        now = time.time()
        owner_clause = " AND worker=?" if worker else ""
        owner_params = [worker] if worker else []
        with self._immediate() as conn:
            cursor = conn.execute(
                "UPDATE jobs SET status='pending', lease_expires_at=NULL, "
                "updated_at=?, attempts=MAX(attempts-1, 0) WHERE job_id=? "
                f"AND status='running'{owner_clause}",
                (now, job_id, *owner_params),
            )
            return cursor.rowcount == 1

    def defer(
        self,
        job_id: str,
        error: str,
        *,
        steps_completed: int = 0,
        worker: str | None = None,
    ) -> bool:
        """Return a transiently blocked job without consuming an attempt."""
        now = time.time()
        owner_clause = " AND worker=?" if worker else ""
        owner_params = [worker] if worker else []
        with self._immediate() as conn:
            cursor = conn.execute(
                """
                UPDATE jobs
                   SET status='pending', finished_at=NULL, lease_expires_at=NULL,
                       updated_at=?, attempts=MAX(attempts-1, 0), error=?,
                       steps_completed=MAX(steps_completed, ?)
                 WHERE job_id=? AND status='running'
                """
                + owner_clause,
                (
                    now,
                    error[:4000],
                    int(steps_completed),
                    job_id,
                    *owner_params,
                ),
            )
            return cursor.rowcount == 1

    def requeue_worker(
        self,
        worker: str,
        *,
        max_attempts: int,
        reason: str,
        preserve_attempt_budget: bool = False,
    ) -> int:
        """Release leases held by a dead worker, failing exhausted jobs."""
        now = time.time()
        with self._immediate() as conn:
            if preserve_attempt_budget:
                cursor = conn.execute(
                    """
                    UPDATE jobs
                       SET status='pending', attempts=MAX(attempts-1, 0),
                           finished_at=NULL, lease_expires_at=NULL, updated_at=?,
                           error=?
                     WHERE status='running' AND worker=?
                    """,
                    (now, reason[:4000], worker),
                )
                return cursor.rowcount
            cursor = conn.execute(
                """
                UPDATE jobs
                   SET status=CASE WHEN attempts >= ? THEN 'failed' ELSE 'pending' END,
                       finished_at=CASE WHEN attempts >= ? THEN ? ELSE NULL END,
                       lease_expires_at=NULL, updated_at=?, error=?
                 WHERE status='running' AND worker=?
                """,
                (max_attempts, max_attempts, now, now, reason[:4000], worker),
            )
            return cursor.rowcount

    def running_jobs_for_worker(self, worker: str) -> list[Job]:
        rows = self.conn.execute(
            "SELECT * FROM jobs WHERE status='running' AND worker=?",
            (worker,),
        ).fetchall()
        return [Job.from_row(row) for row in rows]

    def mark_exhausted(self, *, max_attempts: int) -> int:
        """Move unclaimable pending jobs to failed for accurate status."""
        now = time.time()
        with self._immediate() as conn:
            cursor = conn.execute(
                """
                UPDATE jobs
                   SET status='failed', finished_at=?, updated_at=?,
                       error=COALESCE(error, 'Maximum attempts exhausted')
                 WHERE status='pending' AND attempts >= ?
                """,
                (now, now, max_attempts),
            )
            return cursor.rowcount

    def has_runnable(
        self,
        *,
        max_attempts: int,
        models: Sequence[str] | None = None,
        games: Sequence[str] | None = None,
        seeds: Sequence[int] | None = None,
        retry_failed: bool = False,
        now: float | None = None,
    ) -> bool:
        now = time.time() if now is None else now
        extra, params = self._filter_clause(models, games, seeds)
        conditions = [
            "(status='pending' AND attempts < ?)",
            "(status='running' AND COALESCE(lease_expires_at, 0) < ? "
            "AND attempts < ?)",
        ]
        query_params: list[Any] = [max_attempts, now, max_attempts]
        if retry_failed:
            conditions.append("(status='failed' AND attempts < ?)")
            query_params.append(max_attempts)
        row = self.conn.execute(
            f"SELECT 1 FROM jobs WHERE ({' OR '.join(conditions)}){extra} LIMIT 1",
            (*query_params, *params),
        ).fetchone()
        return row is not None

    def requeue_stale(self, *, now: float | None = None) -> int:
        now = time.time() if now is None else now
        with self._immediate() as conn:
            cursor = conn.execute(
                "UPDATE jobs SET status='pending', lease_expires_at=NULL, updated_at=? "
                "WHERE status='running' AND COALESCE(lease_expires_at, 0) < ?",
                (now, now),
            )
            return cursor.rowcount

    def requeue_failed(self, *, reset_attempts: bool = True) -> int:
        now = time.time()
        attempts = ", attempts=0" if reset_attempts else ""
        with self._immediate() as conn:
            cursor = conn.execute(
                f"UPDATE jobs SET status='pending', updated_at=?{attempts} "
                "WHERE status='failed'",
                (now,),
            )
            return cursor.rowcount

    # -- Reporting -------------------------------------------------------

    def status_counts(self) -> dict[str, int]:
        rows = self.conn.execute(
            "SELECT status, COUNT(*) AS n FROM jobs GROUP BY status"
        ).fetchall()
        counts = {status: 0 for status in STATUSES}
        for row in rows:
            counts[row["status"]] = int(row["n"])
        counts["total"] = sum(counts[status] for status in STATUSES)
        return counts

    def per_model_counts(self) -> list[dict[str, Any]]:
        rows = self.conn.execute(
            """
            SELECT model,
                   COUNT(*) AS total,
                   SUM(status='done') AS done,
                   SUM(status='failed') AS failed,
                   SUM(status='running') AS running,
                   SUM(status='pending') AS pending,
                   SUM(COALESCE(steps_completed, 0)) AS steps,
                   AVG(CASE WHEN status='done' THEN normalized_score END) AS mean_norm
              FROM jobs GROUP BY model ORDER BY model
            """
        ).fetchall()
        return [dict(row) for row in rows]

    def failures(self, limit: int = 20) -> list[dict[str, Any]]:
        rows = self.conn.execute(
            "SELECT job_id, model, game, seed, attempts, error FROM jobs "
            "WHERE status='failed' ORDER BY updated_at DESC LIMIT ?",
            (limit,),
        ).fetchall()
        return [dict(row) for row in rows]

    def get(self, job_id: str) -> dict[str, Any] | None:
        row = self.conn.execute(
            "SELECT * FROM jobs WHERE job_id = ?", (job_id,)
        ).fetchone()
        return dict(row) if row else None
