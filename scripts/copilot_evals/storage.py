"""Crash-safe file helpers used by the resumable runner."""

from __future__ import annotations

import fcntl
import gzip
import json
import os
import tempfile
import zlib
from pathlib import Path
from typing import Any, Iterator


def _fsync_dir(path: Path) -> None:
    fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def atomic_write_text(path: Path, text: str) -> None:
    """Write ``text`` to ``path`` atomically (tmp file + fsync + rename)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(dir=str(path.parent), prefix=f".{path.name}.")
    tmp = Path(tmp_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, path)
        _fsync_dir(path.parent)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise


def atomic_write_bytes(path: Path, payload: bytes) -> None:
    """Write bytes atomically (tmp file + fsync + rename)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(dir=str(path.parent), prefix=f".{path.name}.")
    tmp = Path(tmp_name)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, path)
        _fsync_dir(path.parent)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise


def atomic_write_gzip_text(path: Path, text: str) -> None:
    """Write gzip-compressed text atomically."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(dir=str(path.parent), prefix=f".{path.name}.")
    tmp = Path(tmp_name)
    try:
        with os.fdopen(fd, "wb") as raw:
            with gzip.GzipFile(
                fileobj=raw, mode="wb", compresslevel=6, mtime=0
            ) as compressed:
                compressed.write(text.encode("utf-8"))
            raw.flush()
            os.fsync(raw.fileno())
        os.replace(tmp, path)
        _fsync_dir(path.parent)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise


def atomic_write_json(path: Path, payload: Any, *, indent: int | None = None) -> None:
    text = json.dumps(payload, indent=indent, default=str)
    if Path(path).suffix == ".gz":
        atomic_write_gzip_text(path, text)
    else:
        atomic_write_text(path, text)


def read_json(path: Path) -> Any | None:
    """Read JSON, returning ``None`` for missing or truncated files."""
    try:
        path = Path(path)
        if path.suffix == ".gz":
            with gzip.open(path, "rt", encoding="utf-8") as handle:
                text = handle.read()
        else:
            text = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return None
    except (EOFError, OSError, UnicodeDecodeError, zlib.error):
        return None
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return None


def append_jsonl(path: Path, payload: Any, *, fsync: bool = True) -> None:
    """Append one JSON line under an exclusive lock (multi-process safe)."""
    append_jsonl_many(path, [payload], fsync=fsync)


def append_jsonl_many(path: Path, payloads: list[Any], *, fsync: bool = True) -> None:
    """Append several JSON lines in one locked write."""
    if not payloads:
        return
    path = Path(path)
    if path.suffix == ".gz":
        raise ValueError(
            "Appending gzip members is not crash-safe; append to an uncompressed "
            "working JSONL file and atomically compress it when finalized."
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    blob = "".join(
        json.dumps(payload, default=str) + "\n" for payload in payloads
    ).encode("utf-8")
    with open(path, "a+b") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        try:
            handle.seek(0, os.SEEK_END)
            _truncate_torn_jsonl_suffix(handle)
            handle.seek(0, os.SEEK_END)
            handle.write(blob)
            handle.flush()
            if fsync:
                os.fsync(handle.fileno())
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def _truncate_torn_jsonl_suffix(handle) -> bool:
    """Remove an incomplete final JSONL line from a locked binary file."""
    handle.seek(0, os.SEEK_END)
    size = handle.tell()
    if not size:
        return False

    handle.seek(-1, os.SEEK_END)
    if handle.read(1) == b"\n":
        return False

    cursor = size
    while cursor:
        chunk_start = max(0, cursor - 8192)
        handle.seek(chunk_start)
        chunk = handle.read(cursor - chunk_start)
        newline = chunk.rfind(b"\n")
        if newline >= 0:
            handle.truncate(chunk_start + newline + 1)
            return True
        cursor = chunk_start

    handle.truncate(0)
    return True


def iter_jsonl(path: Path, *, skip_invalid: bool = True) -> Iterator[Any]:
    """Yield JSON lines, tolerating a truncated final line after a crash."""
    path = Path(path)
    if not path.exists():
        return
    opener = gzip.open if path.suffix == ".gz" else open
    try:
        handle = opener(path, "rt", encoding="utf-8")
    except (EOFError, OSError, zlib.error):
        return
    try:
        with handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    yield json.loads(line)
                except json.JSONDecodeError:
                    if skip_invalid:
                        continue
                    raise
    except (EOFError, OSError, UnicodeDecodeError, zlib.error):
        if not skip_invalid:
            raise


def read_jsonl_prefix(path: Path) -> tuple[list[Any], bool]:
    """Read the valid prefix of a JSONL file.

    Returns ``(records, damaged)`` and stops at the first malformed non-empty
    line. Callers can atomically rewrite ``records`` to remove a torn suffix.
    """
    records: list[Any] = []
    damaged = False
    path = Path(path)
    if not path.exists():
        return records, damaged

    opener = gzip.open if path.suffix == ".gz" else open
    try:
        handle = opener(path, "rt", encoding="utf-8")
        with handle:
            for line in handle:
                if not line.strip():
                    continue
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    damaged = True
                    break
    except (EOFError, OSError, UnicodeDecodeError, zlib.error):
        damaged = True
    return records, damaged


def rewrite_jsonl(path: Path, payloads: list[Any]) -> None:
    """Rewrite a JSONL file atomically (used to drop a torn trailing line)."""
    text = "".join(json.dumps(p, default=str) + "\n" for p in payloads)
    if Path(path).suffix == ".gz":
        atomic_write_gzip_text(path, text)
    else:
        atomic_write_text(path, text)


def remove_tree(path: Path) -> None:
    import shutil

    shutil.rmtree(path, ignore_errors=True)
