"""Command line interface for the Copilot × Jericho evaluation batch."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path
from typing import Any, Sequence

from . import availability
from . import export as export_module
from . import games as games_module
from . import storage, telemetry
from .config import (
    DEFAULT_EXPORT_BATCH_SIZE,
    DEFAULT_MAX_STEPS,
    DEFAULT_OUTPUT_ROOT,
    DEFAULT_SEEDS,
    DEFAULT_SPLIT,
    RunConfig,
)
from .manifest import Manifest
from .models import build_model_specs
from .schema import transcript_id


def _add_common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--output-root",
        default=str(DEFAULT_OUTPUT_ROOT),
        help="Directory holding the manifest, trajectories and exports. "
        "Default: %(default)s",
    )
    parser.add_argument(
        "--sdk-path", default=None, help="Path to the Copilot SDK index.js."
    )
    parser.add_argument(
        "--cli-path", default=None, help="Path to the Copilot CLI runtime."
    )
    parser.add_argument(
        "--copilot-home", default=None, help="Copilot base directory (session store)."
    )
    parser.add_argument(
        "--reasoning-summary",
        choices=("none", "concise", "detailed"),
        default=None,
        help="Reasoning summary mode used to capture `thinking`. Default: concise",
    )
    parser.add_argument("-v", "--verbose", action="store_true")


def _add_selection(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--models", nargs="+", help="Restrict to these model ids.")
    parser.add_argument("--games", nargs="+", help="Restrict to these game env names.")
    parser.add_argument("--seeds", nargs="+", type=int, help="Restrict to these seeds.")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="run_copilot_jericho_evals.py",
        description=(
            "Generate Hugging Face compatible Jericho trajectories for every "
            "callable GitHub Copilot SDK model, at its highest supported "
            "reasoning effort."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_models = sub.add_parser("list-models", help="Show callable models + efforts.")
    _add_common(p_models)
    p_models.add_argument("--json", action="store_true", help="Emit JSON.")

    p_games = sub.add_parser("list-games", help="Show qualified games for a split.")
    _add_common(p_games)
    p_games.add_argument("--split", default=DEFAULT_SPLIT)
    p_games.add_argument("--json", action="store_true")

    p_plan = sub.add_parser("plan", help="Materialise the job manifest.")
    _add_common(p_plan)
    _add_selection(p_plan)
    p_plan.add_argument("--split", default=DEFAULT_SPLIT)
    p_plan.add_argument("--max-steps", type=int, default=DEFAULT_MAX_STEPS)
    p_plan.add_argument("--seed-list", nargs="+", type=int, default=list(DEFAULT_SEEDS))
    p_plan.add_argument("--game-seed", type=int, default=None)
    p_plan.add_argument(
        "--fake-model",
        action="store_true",
        help="Plan against the offline fake model (no API calls).",
    )

    p_download = sub.add_parser(
        "download", help="Pre-fetch every game ROM (run before `run`)."
    )
    _add_common(p_download)
    p_download.add_argument("--split", default=DEFAULT_SPLIT)
    p_download.add_argument("--games", nargs="+")

    p_run = sub.add_parser(
        "run",
        help="Disabled legacy runner; use run_canonical_copilot.py.",
    )
    _add_common(p_run)
    _add_selection(p_run)
    p_run.add_argument("--concurrency", type=int, default=None)
    p_run.add_argument(
        "--worker-index-offset",
        type=int,
        default=None,
        help="Start worker/log numbering at this offset.",
    )
    p_run.add_argument("--max-jobs", type=int, default=None)
    p_run.add_argument("--call-timeout", type=float, default=None)
    p_run.add_argument("--lease-seconds", type=float, default=None)
    p_run.add_argument("--max-attempts", type=int, default=None)
    p_run.add_argument("--max-worker-restarts", type=int, default=None)
    p_run.add_argument(
        "--model-unavailable-retry-seconds",
        type=float,
        default=None,
        help="Initial per-model retry delay after a model-availability failure.",
    )
    p_run.add_argument(
        "--model-unavailable-max-retry-seconds",
        type=float,
        default=None,
        help="Maximum per-model retry delay after repeated availability failures.",
    )
    p_run.add_argument("--native-step-timeout", type=float, default=None)
    p_run.add_argument(
        "--min-free-gb",
        type=float,
        default=None,
        help="Pause safely when output storage drops below this threshold.",
    )
    p_run.add_argument("--fake-model", action="store_true")
    p_run.add_argument("--keep-sessions", action="store_true")
    p_run.add_argument("--retry-failed", action="store_true")
    p_run.add_argument("--requeue-stale", action="store_true")
    p_run.add_argument("--on-divergence", choices=("warn", "restart"), default=None)

    p_status = sub.add_parser("status", help="Progress report.")
    _add_common(p_status)
    p_status.add_argument("--json", action="store_true")
    p_status.add_argument("--failures", type=int, default=10)

    p_export = sub.add_parser("export", help="Write JSONL + Parquet dataset.")
    _add_common(p_export)
    p_export.add_argument("--batch-size", type=int, default=DEFAULT_EXPORT_BATCH_SIZE)
    p_export.add_argument("--limit", type=int, default=None)

    p_verify = sub.add_parser("verify", help="Validate trajectories on disk.")
    _add_common(p_verify)
    p_verify.add_argument("--limit", type=int, default=None)

    p_requeue = sub.add_parser("requeue", help="Requeue stale/failed jobs.")
    _add_common(p_requeue)
    p_requeue.add_argument("--stale", action="store_true")
    p_requeue.add_argument("--failed", action="store_true")

    p_telemetry = sub.add_parser("telemetry", help="Aggregate model-call telemetry.")
    _add_common(p_telemetry)
    p_telemetry.add_argument("--json", action="store_true")

    return parser


def config_from_args(args: argparse.Namespace) -> RunConfig:
    """Load a saved config when present, then apply CLI overrides."""
    output_root = Path(args.output_root).expanduser().resolve()
    saved = output_root / "config.json"
    config = (
        RunConfig.load(output_root)
        if saved.exists()
        else RunConfig(output_root=output_root)
    )
    config.output_root = output_root

    overrides = {
        "split": getattr(args, "split", None),
        "max_steps": getattr(args, "max_steps", None),
        "game_seed": getattr(args, "game_seed", None),
        "concurrency": getattr(args, "concurrency", None),
        "worker_index_offset": getattr(args, "worker_index_offset", None),
        "call_timeout": getattr(args, "call_timeout", None),
        "lease_seconds": getattr(args, "lease_seconds", None),
        "max_attempts": getattr(args, "max_attempts", None),
        "max_worker_restarts": getattr(args, "max_worker_restarts", None),
        "model_unavailable_retry_seconds": getattr(
            args, "model_unavailable_retry_seconds", None
        ),
        "model_unavailable_max_retry_seconds": getattr(
            args, "model_unavailable_max_retry_seconds", None
        ),
        "native_step_timeout": getattr(args, "native_step_timeout", None),
        "min_free_gb": getattr(args, "min_free_gb", None),
        "sdk_path": getattr(args, "sdk_path", None),
        "cli_path": getattr(args, "cli_path", None),
        "copilot_home": getattr(args, "copilot_home", None),
        "reasoning_summary": getattr(args, "reasoning_summary", None),
        "on_divergence": getattr(args, "on_divergence", None),
    }
    for key, value in overrides.items():
        if value is not None:
            setattr(config, key, value)

    if getattr(args, "seed_list", None):
        config.seeds = tuple(args.seed_list)
    for flag in ("fake_model", "keep_sessions", "verbose"):
        if getattr(args, flag, False):
            setattr(config, flag, True)

    config.__post_init__()
    return config


# -- Commands ------------------------------------------------------------


def _discover_models(config: RunConfig) -> list[dict[str, Any]]:
    if config.fake_model:
        from .fakes import FakeBridge

        return FakeBridge(config).list_models()

    from .bridge import list_models

    return list_models(config)


def cmd_list_models(args: argparse.Namespace) -> int:
    config = config_from_args(args)
    specs = build_model_specs(_discover_models(config))
    if args.json:
        print(json.dumps([spec.to_dict() for spec in specs], indent=2))
        return 0

    width = max(len(spec.id) for spec in specs)
    print(f"{len(specs)} callable models (excluding the `auto` alias):")
    for spec in specs:
        efforts = ",".join(spec.supported_efforts) or "-"
        print(
            f"  {spec.id.ljust(width)}  effort={spec.effort_label:<6}"
            f"  family={spec.provider_family:<10} supported=[{efforts}]"
        )
    return 0


def cmd_list_games(args: argparse.Namespace) -> int:
    games = games_module.list_split_games(args.split)
    if args.json:
        print(json.dumps([game.to_dict() for game in games], indent=2))
        return 0
    print(f"{len(games)} qualified games in split {args.split!r}")
    for game in games[:10]:
        print(f"  {game.env_name}  ({game.title})")
    if len(games) > 10:
        print(f"  ... and {len(games) - 10} more")
    return 0


def cmd_plan(args: argparse.Namespace) -> int:
    config = config_from_args(args)
    config.ensure_dirs()

    specs = build_model_specs(_discover_models(config), include=args.models)
    games = games_module.filter_games(
        games_module.list_split_games(config.split), args.games
    )
    seeds = tuple(args.seeds) if args.seeds else config.seeds

    storage.atomic_write_json(
        config.output_root / "models.json",
        {
            "generated_at": time.time(),
            "models": [spec.to_dict() for spec in specs],
        },
        indent=2,
    )
    storage.atomic_write_json(
        config.output_root / "games.json",
        {
            "split": config.split,
            "count": len(games),
            "games": [game.to_dict() for game in games],
        },
        indent=2,
    )
    config.save()

    def jobs():
        for spec in specs:
            for game in games:
                for seed in seeds:
                    yield {
                        "job_id": transcript_id(
                            game.env_name,
                            spec.id,
                            config.agent_type,
                            spec.effort_label,
                            seed,
                        ),
                        "model": spec.id,
                        "model_slug": spec.slug,
                        "game": game.env_name,
                        "seed": seed,
                        "reasoning_effort": spec.effort_label,
                        "max_steps": config.max_steps,
                    }

    with Manifest(config.manifest_path) as manifest:
        added = manifest.add_jobs(jobs())
        manifest.set_meta("split", config.split)
        manifest.set_meta("planned_at", str(time.time()))
        counts = manifest.status_counts()

    total = len(specs) * len(games) * len(seeds)
    print(
        f"Planned {total} jobs ({len(specs)} models x {len(games)} games x "
        f"{len(seeds)} seeds); {added} new, {counts['total']} in manifest."
    )
    print(f"Worst case model calls: {total * config.max_steps:,}")
    print(f"Manifest: {config.manifest_path}")
    return 0


def cmd_download(args: argparse.Namespace) -> int:
    config = config_from_args(args)
    config.ensure_dirs()
    config.use_scratch_tmpdir()
    games = games_module.filter_games(
        games_module.list_split_games(args.split), args.games
    )
    print(f"Preparing {len(games)} game ROMs into the TALES cache...")
    games_module.download_games(games)
    print("Done. Concurrent workers can now start without racing on downloads.")
    return 0


def cmd_run(args: argparse.Namespace) -> int:
    print(
        "The legacy Copilot episode runner is disabled because it does not "
        "exactly implement benchmark.py and ReasoningAgent semantics. Use "
        "`python scripts/run_canonical_copilot.py ...` instead.",
        file=sys.stderr,
    )
    return 2


def _status_payload(config: RunConfig) -> dict[str, Any]:
    with Manifest(config.manifest_path) as manifest:
        counts = manifest.status_counts()
        per_model = manifest.per_model_counts()
        failures = manifest.failures(limit=50)
    return {
        "generated_at": time.time(),
        "output_root": str(config.output_root),
        "free_gb": shutil.disk_usage(config.output_root).free / (1024**3),
        "counts": counts,
        "per_model": per_model,
        "recent_failures": failures,
        "model_pauses": availability.active_model_pauses(config),
    }


def cmd_status(args: argparse.Namespace) -> int:
    config = config_from_args(args)
    if not config.manifest_path.exists():
        print("No manifest found; run `plan` first.", file=sys.stderr)
        return 2

    payload = _status_payload(config)
    storage.atomic_write_json(config.output_root / "status.json", payload, indent=2)
    if args.json:
        print(json.dumps(payload, indent=2))
        return 0

    counts = payload["counts"]
    done = counts["done"]
    total = counts["total"] or 1
    print(f"{config.output_root}")
    print(f"  free storage: {payload['free_gb']:.1f} GiB")
    print(
        f"  jobs: {counts['total']}  done={done} ({done / total:.1%})"
        f"  running={counts['running']}  pending={counts['pending']}"
        f"  failed={counts['failed']}"
    )
    for row in payload["per_model"]:
        mean = row["mean_norm"]
        mean_text = f"{mean:.3f}" if mean is not None else "n/a"
        print(
            f"  {row['model']:<26} done={row['done'] or 0:<6}"
            f" failed={row['failed'] or 0:<5} steps={row['steps'] or 0:<9}"
            f" mean_norm={mean_text}"
        )
    for pause in payload["model_pauses"]:
        retry_at = time.strftime(
            "%Y-%m-%d %H:%M:%S",
            time.localtime(float(pause["until"])),
        )
        print(
            f"  ~ {pause['model']} temporarily unavailable; "
            f"next probe after {retry_at}"
        )
    for failure in payload["recent_failures"][: args.failures]:
        print(f"  ! {failure['job_id']}: {str(failure['error'])[:120]}")
    return 0


def cmd_export(args: argparse.Namespace) -> int:
    config = config_from_args(args)
    expected_rows = None
    if args.limit is None and config.manifest_path.exists():
        with Manifest(config.manifest_path) as manifest:
            counts = manifest.status_counts()
        if counts["running"] == 0 and counts["pending"] == 0 and counts["failed"] == 0:
            expected_rows = counts["total"]
    stats = export_module.export(
        config,
        batch_size=args.batch_size,
        limit=args.limit,
        expected_rows=expected_rows,
    )
    validation = stats["output_validation"]
    print(
        f"Exported {stats['exported']} trajectories "
        f"({stats['error_count']} invalid) to:\n"
        f"  {stats['jsonl']}\n  {stats['parquet']}\n"
        f"Output rows: JSONL={validation['jsonl_rows']} "
        f"Parquet={validation['parquet_rows']} "
        f"expected={validation['expected_rows']}; "
        f"schema_ok={validation['parquet_schema_matches']}"
    )
    return 0 if stats["error_count"] == 0 and validation["valid"] else 1


def cmd_verify(args: argparse.Namespace) -> int:
    config = config_from_args(args)
    stats = export_module.verify(config, limit=args.limit)
    print(f"valid={stats['valid']} invalid={stats['invalid']}")
    for error in stats["errors"]:
        print(f"  ! {error['path']}: {error['error']}")
    return 0 if stats["invalid"] == 0 else 1


def cmd_requeue(args: argparse.Namespace) -> int:
    config = config_from_args(args)
    with Manifest(config.manifest_path) as manifest:
        if args.stale or not (args.stale or args.failed):
            print(f"Requeued {manifest.requeue_stale()} stale jobs.")
        if args.failed:
            print(f"Requeued {manifest.requeue_failed()} failed jobs.")
    return 0


def cmd_telemetry(args: argparse.Namespace) -> int:
    config = config_from_args(args)
    payload = telemetry.aggregate(config)
    reconciliation = payload["manifest_reconciliation"]
    if args.json:
        print(json.dumps(payload, indent=2))
        return 0 if reconciliation["valid"] else 1
    overall = payload["overall"]
    print(
        f"episodes={overall['episodes']} steps={overall['steps']} "
        f"api_calls={overall['total_api_calls']} "
        f"input={overall['total_input_tokens']:,} "
        f"output={overall['total_output_tokens']:,} "
        f"reasoning={overall['total_reasoning_tokens']:,}"
    )
    for row in payload["models"]:
        print(
            f"  {row['model']:<26} episodes={row['episodes']:<6}"
            f" calls={row['total_api_calls']:<8}"
            f" mean_norm={row['mean_normalized_score']:.3f}"
        )
    if reconciliation["enabled"]:
        print(
            f"manifest_ok={reconciliation['valid']} "
            f"completed_jobs={reconciliation['completed_jobs']} "
            f"episode_rows={reconciliation['episode_rows']}"
        )
        for error in reconciliation["errors"]:
            print(f"  ! {error['job_id']}: {error['error']}")
    return 0 if reconciliation["valid"] else 1


COMMANDS = {
    "list-models": cmd_list_models,
    "list-games": cmd_list_games,
    "plan": cmd_plan,
    "download": cmd_download,
    "run": cmd_run,
    "status": cmd_status,
    "export": cmd_export,
    "verify": cmd_verify,
    "requeue": cmd_requeue,
    "telemetry": cmd_telemetry,
}


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return COMMANDS[args.command](args)
