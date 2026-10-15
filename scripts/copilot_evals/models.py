"""Model discovery, reasoning-effort selection and provider-family mapping."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Iterable, Sequence

# Highest supported reasoning effort wins; ``none`` is never *selected*, it is
# only the label used when a model exposes no configurable effort at all.
EFFORT_PRIORITY = ("max", "xhigh", "high", "medium", "low")
NO_EFFORT_LABEL = "none"

# ``auto`` is an alias that resolves to another model server-side, so it is not
# a distinct model to evaluate.
EXCLUDED_MODEL_IDS = frozenset({"auto"})

_PROVIDER_PREFIXES: tuple[tuple[tuple[str, ...], str], ...] = (
    (("claude", "anthropic"), "anthropic"),
    (("gpt", "o1", "o3", "o4", "codex", "openai"), "openai"),
    (("grok", "xai"), "xai"),
    (("mai", "phi", "microsoft"), "microsoft"),
    (("gemini", "gemma", "google"), "google"),
    (("qwen",), "alibaba"),
    (("llama", "meta"), "meta"),
    (("deepseek",), "deepseek"),
    (("mistral", "mixtral"), "mistral"),
)

_SLUG_RE = re.compile(r"[^A-Za-z0-9._-]+")


def slugify(value: str) -> str:
    """Filesystem/transcript safe rendering of a model or game identifier."""
    return _SLUG_RE.sub("-", value.replace("/", "-")).strip("-")


def select_effort(supported: Sequence[str] | None) -> str | None:
    """Return the highest configurable reasoning effort, or ``None``."""
    available = {str(effort).lower() for effort in supported or ()}
    for effort in EFFORT_PRIORITY:
        if effort in available:
            return effort
    return None


def provider_family(model_id: str, sdk_family: str | None = None) -> str:
    """Map a model id to the provider family used by the HF ``model_family``."""
    for candidate in (model_id, sdk_family or ""):
        lowered = candidate.lower()
        for prefixes, family in _PROVIDER_PREFIXES:
            if any(lowered.startswith(prefix) for prefix in prefixes):
                return family
    return "unknown"


@dataclass(frozen=True)
class ModelSpec:
    """A concrete, callable model plus the effort we will evaluate it at."""

    id: str
    name: str
    sdk_family: str
    provider_family: str
    supported_efforts: tuple[str, ...]
    effort: str | None
    details: dict[str, Any] = field(default_factory=dict, compare=False, repr=False)

    @property
    def effort_label(self) -> str:
        return self.effort or NO_EFFORT_LABEL

    @property
    def slug(self) -> str:
        return slugify(self.id)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "sdk_family": self.sdk_family,
            "provider_family": self.provider_family,
            "supported_efforts": list(self.supported_efforts),
            "effort": self.effort,
            "effort_label": self.effort_label,
            "details": self.details,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ModelSpec":
        return cls(
            id=data["id"],
            name=data.get("name") or data["id"],
            sdk_family=data.get("sdk_family") or data["id"],
            provider_family=data.get("provider_family")
            or provider_family(data["id"], data.get("sdk_family")),
            supported_efforts=tuple(data.get("supported_efforts") or ()),
            effort=data.get("effort"),
            details=data.get("details") or {},
        )


def build_model_spec(raw: dict[str, Any]) -> ModelSpec:
    """Convert one ``list_models`` bridge record into a :class:`ModelSpec`."""
    model_id = raw["id"]
    supported = tuple(
        str(e).lower() for e in raw.get("supported_reasoning_efforts") or ()
    )
    sdk_family = raw.get("family") or model_id
    details = {
        key: raw.get(key)
        for key in (
            "policy_state",
            "context_window",
            "max_prompt_tokens",
            "max_output_tokens",
            "billing",
            "model_picker_category",
            "model_picker_price_category",
            "default_reasoning_effort",
        )
        if raw.get(key) is not None
    }
    return ModelSpec(
        id=model_id,
        name=raw.get("name") or model_id,
        sdk_family=sdk_family,
        provider_family=provider_family(model_id, sdk_family),
        supported_efforts=supported,
        effort=select_effort(supported),
        details=details,
    )


def build_model_specs(
    raw_models: Iterable[dict[str, Any]],
    *,
    include: Iterable[str] | None = None,
    exclude: Iterable[str] = EXCLUDED_MODEL_IDS,
) -> list[ModelSpec]:
    """Build the evaluated model matrix from the SDK model listing.

    Every concrete model returned by the SDK is kept (including preview or
    internal-only display names); only aliases such as ``auto`` are dropped.
    """
    excluded = {model_id.lower() for model_id in exclude}
    wanted = {model_id.lower() for model_id in include} if include else None

    specs: list[ModelSpec] = []
    seen: set[str] = set()
    for raw in raw_models:
        model_id = str(raw.get("id") or "")
        if not model_id or model_id.lower() in excluded or model_id in seen:
            continue
        if wanted is not None and model_id.lower() not in wanted:
            continue
        seen.add(model_id)
        specs.append(build_model_spec(raw))

    if wanted is not None:
        missing = wanted - {spec.id.lower() for spec in specs}
        if missing:
            raise ValueError(f"Unknown model id(s): {sorted(missing)}")

    specs.sort(key=lambda spec: spec.id)
    return specs
