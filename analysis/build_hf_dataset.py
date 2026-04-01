"""Build a Hugging Face dataset from TALES wandb trajectories.

Downloads JSONL rollout files from wandb runs, enriches each step with
run metadata (model, game, framework, seed, etc.), and saves the result
as a Parquet dataset ready for ``datasets.load_dataset()``.

Usage:
    python analysis/build_hf_dataset.py --output analysis/tales_dataset
    python analysis/build_hf_dataset.py --models claude-opus-4.6 gpt-5.4-mini --max-steps 300 400
    python analysis/build_hf_dataset.py --frameworks jericho scienceworld --games JerichoEnvZork1
    python analysis/build_hf_dataset.py --cache analysis/hf_cache.json --output analysis/tales_dataset
"""

import argparse
import json
import sys
import tempfile
from pathlib import Path

import pandas as pd
import wandb

WANDB_PROJECT = "pearls-lab/text-games-benchmark"

METADATA_COLUMNS = [
    "run_id",
    "model",
    "game",
    "framework",
    "agent_type",
    "seed",
    "game_seed",
    "max_steps",
]

ROLLOUT_COLUMNS = [
    "Step",
    "Score",
    "Max Score",
    "Normalized Score",
    "Moves",
    "Observation",
    "Action",
    "Feedback",
    "Prompt",
    "Response",
    "Thinking",
    "Token Usage",
    "Prompt Tokens",
    "Response Tokens",
    "Thinking Tokens",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("analysis/tales_dataset"),
        help="Output directory for the dataset (default: analysis/tales_dataset).",
    )
    parser.add_argument(
        "--format",
        choices=["parquet", "jsonl"],
        default="parquet",
        help="Output format (default: parquet).",
    )

    filt = parser.add_argument_group("filtering")
    filt.add_argument(
        "--models",
        nargs="+",
        default=None,
        help="Only include these models (e.g., claude-opus-4.6 gpt-5.4-mini).",
    )
    filt.add_argument(
        "--frameworks",
        nargs="+",
        default=None,
        help="Only include these frameworks (e.g., jericho textworld scienceworld).",
    )
    filt.add_argument(
        "--games",
        nargs="+",
        default=None,
        help="Only include these game names (e.g., JerichoEnvZork1 ScienceWorldBoil).",
    )
    filt.add_argument(
        "--max-steps",
        type=int,
        nargs="+",
        default=None,
        help="Only include runs with these max_steps values (e.g., 300 400).",
    )
    filt.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=None,
        help="Only include runs with these seed values.",
    )

    parser.add_argument(
        "--cache",
        type=Path,
        default=None,
        help="JSON file tracking which run IDs have been downloaded. "
        "Enables incremental builds — only new runs are fetched.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="List matching runs without downloading trajectories.",
    )
    return parser.parse_args()


def fetch_runs(args: argparse.Namespace) -> list:
    """Fetch finished wandb runs matching the provided filters.

    Prefers the longest run per (model, game, seed) tuple.
    """
    api = wandb.Api()

    filters: dict = {"state": "finished"}
    if args.max_steps and len(args.max_steps) == 1:
        filters["config.max_steps"] = args.max_steps[0]
    if args.models and len(args.models) == 1:
        filters["config.llm"] = args.models[0]
    if args.frameworks and len(args.frameworks) == 1:
        filters["config.framework"] = args.frameworks[0]

    print("Querying wandb for runs...")
    all_runs = list(api.runs(WANDB_PROJECT, filters=filters, order="-created_at"))
    print(f"  Found {len(all_runs)} finished runs (server-side filter).")

    # Client-side filtering for multi-value filters.
    max_steps_set = set(args.max_steps) if args.max_steps else None
    models_set = set(args.models) if args.models else None
    frameworks_set = set(args.frameworks) if args.frameworks else None
    games_set = set(args.games) if args.games else None
    seeds_set = set(args.seeds) if args.seeds else None

    filtered = []
    for r in all_runs:
        cfg = r.config
        if max_steps_set and cfg.get("max_steps") not in max_steps_set:
            continue
        if models_set and cfg.get("llm") not in models_set:
            continue
        if frameworks_set and cfg.get("framework") not in frameworks_set:
            continue
        if games_set and cfg.get("game") not in games_set:
            continue
        if seeds_set and cfg.get("seed") not in seeds_set:
            continue
        filtered.append(r)

    # Deduplicate: keep longest run per (model, game, seed).
    filtered.sort(key=lambda r: r.config.get("max_steps", 0) or 0, reverse=True)
    seen: set[tuple] = set()
    selected = []
    for r in filtered:
        key = (
            r.config.get("llm", "unknown"),
            r.config.get("game", "unknown"),
            r.config.get("seed"),
        )
        if key in seen:
            continue
        seen.add(key)
        selected.append(r)

    print(f"  After filtering & dedup: {len(selected)} runs.")
    return selected


def download_trajectory(run) -> pd.DataFrame | None:
    """Download the JSONL rollout from a wandb run and return as DataFrame."""
    rollout_file = None
    for f in run.files():
        if f.name.endswith(".jsonl"):
            rollout_file = f
            break

    if rollout_file is None:
        return None

    with tempfile.TemporaryDirectory() as tmpdir:
        rollout_file.download(root=tmpdir, replace=True)
        filepath = f"{tmpdir}/{rollout_file.name}"
        df = pd.read_json(filepath, orient="records", lines=True)

    return df


def build_dataset(runs: list, cached_ids: set[str]) -> pd.DataFrame:
    """Download trajectories for all runs and build a single DataFrame."""
    parts = []
    skipped_cache = 0
    skipped_no_data = 0

    for i, run in enumerate(runs):
        cfg = run.config
        model = cfg.get("llm", "unknown")
        game = cfg.get("game", "unknown")
        seed = cfg.get("seed")
        max_steps = cfg.get("max_steps", 0)

        if run.id in cached_ids:
            skipped_cache += 1
            continue

        print(
            f"  [{i + 1}/{len(runs)}] {game} / {model} "
            f"(seed={seed}, max_steps={max_steps})"
        )

        traj = download_trajectory(run)
        if traj is None or traj.empty:
            print(f"    ⚠ No trajectory found, skipping.")
            skipped_no_data += 1
            continue

        # Attach metadata columns.
        traj["run_id"] = run.id
        traj["model"] = model
        traj["game"] = game
        traj["framework"] = cfg.get("framework", "unknown")
        traj["agent_type"] = cfg.get("agent_type", "unknown")
        traj["seed"] = seed
        traj["game_seed"] = cfg.get("game_seed")
        traj["max_steps"] = max_steps

        parts.append(traj)

    if skipped_cache:
        print(f"  Skipped {skipped_cache} already-cached run(s).")
    if skipped_no_data:
        print(f"  Skipped {skipped_no_data} run(s) with no trajectory data.")

    if not parts:
        return pd.DataFrame()

    return pd.concat(parts, ignore_index=True)


def load_cache(cache_path: Path | None) -> tuple[set[str], pd.DataFrame | None]:
    """Load cache index and any previously saved data."""
    if cache_path is None or not cache_path.exists():
        return set(), None

    with open(cache_path) as f:
        cache = json.load(f)

    cached_ids = set(cache.get("run_ids", []))
    data_path = cache.get("data_path")

    prev_df = None
    if data_path and Path(data_path).exists():
        print(f"Loading previously cached data from {data_path}")
        if data_path.endswith(".parquet"):
            prev_df = pd.read_parquet(data_path)
        else:
            prev_df = pd.read_json(data_path, orient="records", lines=True)

    print(f"Cache: {len(cached_ids)} previously downloaded run(s).")
    return cached_ids, prev_df


def save_cache(cache_path: Path, run_ids: set[str], data_path: str) -> None:
    """Save cache index."""
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with open(cache_path, "w") as f:
        json.dump({"run_ids": sorted(run_ids), "data_path": data_path}, f, indent=2)


def main() -> None:
    args = parse_args()

    runs = fetch_runs(args)
    if not runs:
        print("No matching runs found. Exiting.", file=sys.stderr)
        sys.exit(1)

    if args.dry_run:
        print(f"\nDry run — {len(runs)} runs would be downloaded:\n")
        summary: dict[str, set] = {}
        for r in runs:
            model = r.config.get("llm", "unknown")
            game = r.config.get("game", "unknown")
            summary.setdefault(model, set()).add(game)
        for model in sorted(summary):
            print(f"  {model}: {len(summary[model])} games")
        return

    # Load cache for incremental builds.
    cached_ids, prev_df = load_cache(args.cache)

    # Download new trajectories.
    new_df = build_dataset(runs, cached_ids)

    # Merge with previously cached data.
    if prev_df is not None and not prev_df.empty:
        # Remove any previously cached runs that are no longer selected
        # (e.g., superseded by longer runs or filtered out).
        selected_ids = {r.id for r in runs}
        prev_df = prev_df[prev_df["run_id"].isin(selected_ids)]

        if not new_df.empty:
            df = pd.concat([prev_df, new_df], ignore_index=True)
        else:
            df = prev_df
    else:
        df = new_df

    if df.empty:
        print("No trajectory data collected. Exiting.", file=sys.stderr)
        sys.exit(1)

    # Reorder columns: metadata first, then rollout columns.
    all_cols = METADATA_COLUMNS + [c for c in ROLLOUT_COLUMNS if c in df.columns]
    extra = [c for c in df.columns if c not in all_cols]
    df = df[all_cols + extra]

    # Save dataset.
    args.output.mkdir(parents=True, exist_ok=True)
    if args.format == "parquet":
        out_path = args.output / "tales_trajectories.parquet"
        df.to_parquet(out_path, index=False)
    else:
        out_path = args.output / "tales_trajectories.jsonl"
        df.to_json(out_path, orient="records", lines=True)

    # Update cache.
    all_ids = cached_ids | set(df["run_id"].unique())
    if args.cache:
        save_cache(args.cache, all_ids, str(out_path))

    n_runs = df["run_id"].nunique()
    n_models = df["model"].nunique()
    n_games = df["game"].nunique()
    n_steps = len(df)
    print(f"\n✓ Dataset saved to {out_path}")
    print(f"  {n_runs} runs, {n_models} models, {n_games} games, {n_steps} steps")
    print(f"  Size: {out_path.stat().st_size / 1024 / 1024:.1f} MB")


if __name__ == "__main__":
    main()
