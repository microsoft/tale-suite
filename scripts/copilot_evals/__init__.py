"""Resumable Copilot SDK x Jericho evaluation pipeline.

Modules:
    config      run configuration + output layout
    models      model discovery, effort selection, provider families
    games       qualified Jericho games and gym env construction
    bridge      JSONL RPC client for scripts/copilot_model_bridge.mjs
    prompts     system/observation/nudge prompts for the game_action tool
    schema      Hugging Face trajectory schema (20 columns)
    storage     atomic + append-only crash-safe file helpers
    manifest    SQLite job manifest with leases and resume
    episode     per-episode runner with step level checkpointing
    worker      multi-process worker pool
    telemetry   model-call telemetry sidecar
    export      streaming JSONL + Parquet export
    cli         command line interface
"""

from .config import RunConfig
from .models import ModelSpec, build_model_specs, select_effort
from .schema import HF_COLUMNS, StepRecord, build_trajectory, validate_trajectory

__all__ = [
    "HF_COLUMNS",
    "ModelSpec",
    "RunConfig",
    "StepRecord",
    "build_model_specs",
    "build_trajectory",
    "select_effort",
    "validate_trajectory",
]
