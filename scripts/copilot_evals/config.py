"""Configuration and on-disk layout for the Copilot Jericho evaluation batch."""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, fields
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

FRAMEWORK = "jericho"
AGENT_TYPE = "react"
DEFAULT_SPLIT = "jericho-new"
DEFAULT_MAX_STEPS = 300
DEFAULT_SEEDS = (202411061, 202411062, 202411063, 202411064, 202411065)
DEFAULT_OUTPUT_DIRNAME = "jericho-new-api-highest-300steps-5seeds"
DEFAULT_OUTPUT_ROOT = REPO_ROOT.parent / "tale-suite-evals" / DEFAULT_OUTPUT_DIRNAME

# One Copilot CLI runtime (~160 MB resident) plus one Jericho VM per worker.
DEFAULT_CONCURRENCY = 8
DEFAULT_CALL_TIMEOUT = 600.0
DEFAULT_LEASE_SECONDS = 7200.0
DEFAULT_MAX_ATTEMPTS = 5
DEFAULT_MAX_WORKER_RESTARTS = 20
DEFAULT_EXPORT_BATCH_SIZE = 200
DEFAULT_MIN_FREE_GB = 8.0
DEFAULT_MODEL_UNAVAILABLE_RETRY_SECONDS = 15 * 60.0
DEFAULT_MODEL_UNAVAILABLE_MAX_RETRY_SECONDS = 6 * 60 * 60.0

BRIDGE_SCRIPT = REPO_ROOT / "scripts" / "copilot_model_bridge.mjs"
SDK_GLOB = "/root/.cache/copilot/pkg/*/*/copilot-sdk/index.js"
DEFAULT_CLI_PATH = "/usr/local/bin/copilot"


def default_sdk_path() -> str | None:
    """Locate the bundled Copilot SDK, preferring the newest package version."""
    explicit = os.environ.get("COPILOT_SDK_PATH")
    if explicit:
        return explicit

    import glob

    candidates = sorted(glob.glob(SDK_GLOB))
    return candidates[-1] if candidates else None


@dataclass
class RunConfig:
    """Everything a worker needs to know to execute jobs."""

    output_root: Path = DEFAULT_OUTPUT_ROOT
    split: str = DEFAULT_SPLIT
    max_steps: int = DEFAULT_MAX_STEPS
    seeds: tuple[int, ...] = DEFAULT_SEEDS
    agent_type: str = AGENT_TYPE
    framework: str = FRAMEWORK
    game_seed: int | None = None
    admissible_commands: bool = False
    concurrency: int = DEFAULT_CONCURRENCY
    worker_index_offset: int = 0
    call_timeout: float = DEFAULT_CALL_TIMEOUT
    lease_seconds: float = DEFAULT_LEASE_SECONDS
    max_attempts: int = DEFAULT_MAX_ATTEMPTS
    max_worker_restarts: int = DEFAULT_MAX_WORKER_RESTARTS
    max_call_retries: int = 3
    max_tool_retries: int = 3
    model_unavailable_retry_seconds: float = DEFAULT_MODEL_UNAVAILABLE_RETRY_SECONDS
    model_unavailable_max_retry_seconds: float = (
        DEFAULT_MODEL_UNAVAILABLE_MAX_RETRY_SECONDS
    )
    native_step_timeout: float = 30.0
    reasoning_summary: str = "concise"
    keep_sessions: bool = False
    fake_model: bool = False
    on_divergence: str = "restart"
    min_free_gb: float = DEFAULT_MIN_FREE_GB
    sdk_path: str | None = None
    cli_path: str = DEFAULT_CLI_PATH
    copilot_home: str = os.environ.get("COPILOT_HOME", str(Path.home() / ".copilot"))
    node_bin: str = "node"
    bridge_log_level: str = "error"
    verbose: bool = False

    def __post_init__(self) -> None:
        self.output_root = Path(self.output_root).expanduser().resolve()
        self.seeds = tuple(int(seed) for seed in self.seeds)
        if self.sdk_path is None:
            self.sdk_path = default_sdk_path()

    # -- Layout ----------------------------------------------------------

    @property
    def manifest_path(self) -> Path:
        return self.output_root / "manifest.sqlite3"

    @property
    def trajectories_dir(self) -> Path:
        return self.output_root / "trajectories"

    @property
    def partial_dir(self) -> Path:
        return self.output_root / "partial"

    @property
    def telemetry_dir(self) -> Path:
        return self.output_root / "telemetry"

    @property
    def logs_dir(self) -> Path:
        return self.output_root / "logs"

    @property
    def raw_dir(self) -> Path:
        return self.output_root / "raw"

    @property
    def data_dir(self) -> Path:
        return self.output_root / "data" / self.framework

    @property
    def stop_file(self) -> Path:
        return self.output_root / "STOP"

    @property
    def scratch_dir(self) -> Path:
        return self.output_root / "scratch"

    def use_scratch_tmpdir(self) -> None:
        """Stage downloads/temp files inside the output root, not /tmp."""
        self.scratch_dir.mkdir(parents=True, exist_ok=True)
        os.environ["TMPDIR"] = str(self.scratch_dir)

    def ensure_dirs(self) -> None:
        for path in (
            self.output_root,
            self.trajectories_dir,
            self.partial_dir,
            self.telemetry_dir,
            self.logs_dir,
            self.raw_dir,
            self.data_dir,
            self.scratch_dir,
        ):
            path.mkdir(parents=True, exist_ok=True)

    # -- Serialization ---------------------------------------------------

    def to_dict(self) -> dict:
        data = asdict(self)
        data["output_root"] = str(self.output_root)
        data["seeds"] = list(self.seeds)
        return data

    @classmethod
    def from_dict(cls, data: dict) -> "RunConfig":
        known = {f.name for f in fields(cls)}
        kwargs = {key: value for key, value in data.items() if key in known}
        if "seeds" in kwargs:
            kwargs["seeds"] = tuple(kwargs["seeds"])
        return cls(**kwargs)

    def save(self) -> Path:
        self.ensure_dirs()
        path = self.output_root / "config.json"
        path.write_text(json.dumps(self.to_dict(), indent=2, sort_keys=True))
        return path

    @classmethod
    def load(cls, output_root: Path | str) -> "RunConfig":
        path = Path(output_root).expanduser().resolve() / "config.json"
        return cls.from_dict(json.loads(path.read_text()))
