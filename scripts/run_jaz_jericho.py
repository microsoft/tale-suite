#!/usr/bin/env python3
"""Run the JAZ Harness-as-a-Language adaptation on TALES/Jericho."""

from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.copilot_evals import games as games_module
from scripts.copilot_evals import storage
from scripts.jaz_jericho_harness import (
    DEFAULT_GAMES,
    DEFAULT_MODELS,
    DEFAULT_OUTPUT_ROOT,
    DEFAULT_SEEDS,
    JAZRunConfig,
    aggregate_results,
    run_jaz_episode,
)


def unique(values):
    return list(dict.fromkeys(values))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate Copilot models in a persistent JAZ Python REPL with "
            "recursive invoke and a safe TALES/Jericho step interface."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--models", nargs="+", default=list(DEFAULT_MODELS))
    parser.add_argument("--games", nargs="+", default=None)
    parser.add_argument(
        "--split",
        choices=("jericho-original", "jericho-new", "jericho-all"),
        help="Use every game in this split instead of the default Zork I pilot.",
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=list(DEFAULT_SEEDS))
    parser.add_argument("--reasoning-effort", default="max")
    parser.add_argument("--max-model-calls", type=int, default=500)
    parser.add_argument(
        "--action-safety-limit",
        type=int,
        default=5000,
        help=(
            "Non-scoring runaway guard. A run that reaches this many step() "
            "calls is invalid rather than successfully budget-complete."
        ),
    )
    parser.add_argument(
        "--return-guard-scope",
        choices=("root", "tree"),
        default="root",
        help=(
            "Root allows bounded sub-invokes to return; tree restricts every "
            "sub-invoke to tail-delegating the unfinished episode."
        ),
    )
    parser.add_argument(
        "--transport",
        choices=("custom", "official"),
        default="custom",
        help=(
            "Copilot adapter. Custom retains detailed reasoning/usage telemetry; "
            "official uses JAZ's CopilotLLM backend."
        ),
    )
    parser.add_argument("--max-recursion-depth", type=int, default=8)
    parser.add_argument("--context-warn-fraction", type=float, default=0.70)
    parser.add_argument("--call-timeout", type=float, default=600.0)
    parser.add_argument("--repl-timeout", type=float, default=900.0)
    parser.add_argument("--repl-memory-limit-mb", type=int, default=2048)
    parser.add_argument("--native-step-timeout", type=float, default=30.0)
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--sdk-path")
    parser.add_argument("--cli-path", default="/usr/local/bin/copilot")
    parser.add_argument(
        "--copilot-home",
        default=str(Path.home() / ".copilot"),
    )
    parser.add_argument("--bridge-log-level", default="error")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def selected_games(args: argparse.Namespace) -> list[str]:
    if args.split:
        available = games_module.list_split_games(args.split)
        if args.games:
            available = games_module.filter_games(available, args.games)
        return unique(game.env_name for game in available)
    return unique(args.games or DEFAULT_GAMES)


def build_configs(args: argparse.Namespace) -> list[JAZRunConfig]:
    output_root = args.output_root.expanduser().resolve()
    return [
        JAZRunConfig(
            model=model,
            game=game,
            seed=seed,
            output_root=output_root,
            reasoning_effort=args.reasoning_effort,
            max_model_calls=args.max_model_calls,
            action_safety_limit=args.action_safety_limit,
            return_guard_scope=args.return_guard_scope,
            transport=args.transport,
            max_recursion_depth=args.max_recursion_depth,
            context_warn_fraction=args.context_warn_fraction,
            call_timeout=args.call_timeout,
            repl_timeout=args.repl_timeout,
            repl_memory_limit_mb=args.repl_memory_limit_mb,
            native_step_timeout=args.native_step_timeout,
            max_retries=args.max_retries,
            sdk_path=args.sdk_path,
            cli_path=args.cli_path,
            copilot_home=args.copilot_home,
            bridge_log_level=args.bridge_log_level,
            force=args.force,
        )
        for model in unique(args.models)
        for game in selected_games(args)
        for seed in unique(args.seeds)
    ]


def _run_and_report(config: JAZRunConfig) -> dict:
    return run_jaz_episode(config)


def failed_worker_summary(config: JAZRunConfig, exc: Exception) -> dict:
    failure_path = config.output_root / "runner-failures" / f"{config.run_id}.json"
    summary = {
        "run_id": config.run_id,
        "status": "failed_runner",
        "score_valid": False,
        "skipped": False,
        "model": config.model,
        "reasoning_effort": config.reasoning_effort,
        "harness": "jaz-invoke",
        "paper": "arXiv:2609.26891",
        "duration_seconds": None,
        "result": None,
        "error_type": type(exc).__name__,
        "error": str(exc),
        "config": config.to_dict(),
        "game_result": {
            "game": config.game,
            "seed": config.seed,
            "score": 0,
            "max_score": 0,
            "normalized_score": 0.0,
            "completion_reason": None,
        },
        "model_usage": {},
        "artifacts": {
            "run_dir": str(config.run_dir),
            "runner_failure": str(failure_path),
        },
        "runner_traceback": "".join(
            traceback.format_exception(type(exc), exc, exc.__traceback__)
        ),
        "recorded_at": time.time(),
    }
    storage.atomic_write_json(failure_path, summary, indent=2)
    return summary


def main() -> int:
    args = build_parser().parse_args()
    if args.concurrency <= 0:
        raise SystemExit("--concurrency must be positive")
    configs = build_configs(args)
    if args.dry_run:
        print(json.dumps([config.to_dict() for config in configs], indent=2))
        return 0

    output_root = configs[0].output_root
    output_root.mkdir(parents=True, exist_ok=True)
    summaries: list[dict] = []
    workers = min(args.concurrency, len(configs))
    with ProcessPoolExecutor(
        max_workers=workers,
        max_tasks_per_child=1,
    ) as executor:
        future_to_config = {
            executor.submit(_run_and_report, config): config for config in configs
        }
        for future in as_completed(future_to_config):
            config = future_to_config[future]
            try:
                summary = future.result()
            except Exception as exc:
                summary = failed_worker_summary(config, exc)
            summaries.append(summary)
            result = summary["game_result"]
            print(
                f"{summary['run_id']}: {summary['status']} "
                f"{result['score']}/{result['max_score']} "
                f"({result['normalized_score']:.2%})",
                flush=True,
            )

    results = aggregate_results(configs, summaries)
    storage.atomic_write_json(
        output_root / "RESULTS.json",
        results,
        indent=2,
    )
    for model, aggregate in results["aggregates"].items():
        mean = aggregate["mean_normalized_score"]
        rendered = "n/a" if mean is None else f"{mean:.4%}"
        print(
            f"{model}: mean={rendered}, valid={aggregate['valid_runs']}/"
            f"{aggregate['runs']}, calls={aggregate['total_model_calls']}, "
            f"AIU={aggregate['total_ai_credits']:.6f}"
        )
    print(f"Results: {output_root / 'RESULTS.json'}")
    return (
        1
        if any(str(summary["status"]).startswith("failed") for summary in summaries)
        else 0
    )


if __name__ == "__main__":
    raise SystemExit(main())
