"""Model-specific backoff for temporary Copilot API availability outages."""

from __future__ import annotations

import re
import time
from pathlib import Path
from typing import Iterable

from . import storage
from .config import RunConfig
from .models import slugify

_MODEL_UNAVAILABLE_RE = re.compile(
    r"""model\s+["']([^"']{1,120})["']\s+is\s+"""
    r"""(?:not\s+available|unavailable)\b""",
    re.IGNORECASE,
)


class ModelTemporarilyUnavailable(RuntimeError):
    """A planned model is temporarily unavailable through the Copilot API."""

    def __init__(self, model: str, error: str) -> None:
        self.model = model
        self.error = error
        super().__init__(f"Model {model!r} is temporarily unavailable: {error}")


def is_model_unavailable_error(
    error: BaseException,
    *,
    model: str | None = None,
) -> bool:
    """Recognize definitive model-availability failures from the SDK bridge."""
    message = str(error)
    lowered = message.casefold()
    if (
        "requested model is not available" in lowered
        or "requested model is unavailable" in lowered
    ):
        return True
    match = _MODEL_UNAVAILABLE_RE.search(message)
    if match is None:
        return False
    return model is None or match.group(1).casefold() == model.casefold()


def model_pause_path(config: RunConfig, model: str) -> Path:
    return config.output_root / "model-availability" / f"{slugify(model)}.json"


def defer_model(
    config: RunConfig,
    model: str,
    error: str,
    *,
    now: float | None = None,
) -> dict:
    """Pause one model with bounded exponential backoff."""
    now = time.time() if now is None else float(now)
    path = model_pause_path(config, model)
    current = storage.read_json(path)
    if not isinstance(current, dict) or current.get("model") != model:
        current = {}

    current_until = float(current.get("until") or 0)
    if current_until > now:
        failures = max(int(current.get("failures") or 1), 1)
        retry_seconds = float(
            current.get("retry_seconds") or max(current_until - now, 0)
        )
        until = current_until
    else:
        failures = max(int(current.get("failures") or 0), 0) + 1
        base = max(float(config.model_unavailable_retry_seconds), 1.0)
        maximum = max(float(config.model_unavailable_max_retry_seconds), base)
        retry_seconds = min(base * (2 ** min(failures - 1, 20)), maximum)
        until = now + retry_seconds

    payload = {
        "model": model,
        "until": until,
        "retry_seconds": retry_seconds,
        "failures": failures,
        "first_unavailable_at": float(current.get("first_unavailable_at") or now),
        "last_unavailable_at": now,
        "last_error": str(error)[:4000],
    }
    storage.atomic_write_json(path, payload, indent=2)
    return payload


def clear_model_pause(config: RunConfig, model: str) -> None:
    """Clear prior backoff after the SDK accepts the model again."""
    model_pause_path(config, model).unlink(missing_ok=True)


def active_model_pauses(
    config: RunConfig,
    models: Iterable[str] | None = None,
    *,
    now: float | None = None,
) -> list[dict]:
    """Return active pause records in deterministic model order."""
    now = time.time() if now is None else float(now)
    if models is None:
        paths = sorted((config.output_root / "model-availability").glob("*.json"))
    else:
        paths = [model_pause_path(config, model) for model in models]

    pauses: list[dict] = []
    for path in paths:
        payload = storage.read_json(path)
        if not isinstance(payload, dict):
            continue
        model = payload.get("model")
        until = float(payload.get("until") or 0)
        if not isinstance(model, str) or until <= now:
            continue
        pauses.append(payload)
    return sorted(pauses, key=lambda item: item["model"])


def runnable_models(
    config: RunConfig,
    models: Iterable[str],
    *,
    now: float | None = None,
) -> tuple[tuple[str, ...], float | None, list[dict]]:
    """Split requested models into runnable and temporarily paused sets."""
    ordered = tuple(dict.fromkeys(models))
    pauses = active_model_pauses(config, ordered, now=now)
    paused = {pause["model"] for pause in pauses}
    runnable = tuple(model for model in ordered if model not in paused)
    retry_at = min((float(pause["until"]) for pause in pauses), default=None)
    return runnable, retry_at, pauses
