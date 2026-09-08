"""Average normalized score vs. step budget across all games.

For each model, takes runs with the specified max_steps, reads the
per-step normalized highscore, and plots TALES score at configurable
budget intervals.

Usage:
    python analysis/score_vs_budget.py [--cache analysis/data.csv] [--max-steps 300 400] [--budget-step 25]
"""

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import wandb

WANDB_PROJECT = "pearls-lab/text-games-benchmark"

FRAMEWORK_LABELS = {
    "jericho": "Jericho",
    "textworld": "TextWorld",
    "textworld_express": "TextWorldExpress",
    "alfworld": "ALFWorld",
    "scienceworld": "ScienceWorld",
}


def _infer_framework(game: str) -> str:
    """Infer framework from game name prefix (fallback for cached data)."""
    if game.startswith("Jericho"):
        return "Jericho"
    if game.startswith("TWX"):
        return "TextWorldExpress"
    if game.startswith("TW"):
        return "TextWorld"
    if game.startswith("ALFWorld"):
        return "ALFWorld"
    if game.startswith("ScienceWorld"):
        return "ScienceWorld"
    return "Unknown"


def fetch_runs(max_steps_list: list[int]) -> list:
    """Fetch runs, preferring the longest max_steps per (model, game, seed).

    Queries from largest to smallest max_steps. Once a (model, game, seed)
    tuple is covered by a longer run, shorter runs for that tuple are skipped.
    """
    api = wandb.Api()
    seen_keys: set[tuple] = set()
    selected: list = []

    for max_steps in sorted(max_steps_list, reverse=True):
        filters = {"config.max_steps": max_steps, "state": "finished"}
        print(f"Querying wandb for runs with max_steps={max_steps}...")
        runs = list(api.runs(WANDB_PROJECT, filters=filters, order="-created_at"))
        print(f"  Found {len(runs)} runs.")

        skipped = 0
        for r in runs:
            key = (
                r.config.get("llm", "unknown"),
                r.config.get("game", "unknown"),
                r.config.get("seed"),
            )
            if key in seen_keys:
                skipped += 1
                continue
            seen_keys.add(key)
            selected.append(r)

        if skipped:
            print(f"  Skipped {skipped} run(s) already covered by longer runs.")

    print(f"Total: {len(selected)} unique runs to process.")
    return selected


def build_history_table(runs: list) -> pd.DataFrame:
    """Download per-step normalized highscore for each run."""
    records = []
    for i, run in enumerate(runs):
        model = run.config.get("llm", "unknown")
        game = run.config.get("game", "unknown")
        seed = run.config.get("seed")
        run_max_steps = run.config.get("max_steps", 0)
        framework = run.config.get("framework", "unknown")
        framework = FRAMEWORK_LABELS.get(framework, framework)
        print(
            f"  [{i + 1}/{len(runs)}] {game} / {model} (seed={seed}, max_steps={run_max_steps})"
        )

        history = run.history(keys=["episode/normalized_highscore"], pandas=True)
        if history.empty:
            continue

        history = history.dropna(subset=["episode/normalized_highscore"])
        history = history.sort_values("_step")

        for _, row in history.iterrows():
            records.append(
                {
                    "run_id": run.id,
                    "model": model,
                    "game": game,
                    "framework": framework,
                    "seed": seed,
                    "max_steps": run_max_steps,
                    "step": int(row["_step"]),
                    "normalized_score": row["episode/normalized_highscore"],
                }
            )

    return pd.DataFrame(records)


def compute_budgets(history_df: pd.DataFrame, budgets: list[int]) -> pd.DataFrame:
    """From per-step history, compute the normalized highscore at each budget cutoff.

    Uses ``merge_asof`` to map each budget to the most recent step per run,
    avoiding a per-budget inner loop.
    """
    run_cols = ["run_id", "model", "game", "framework", "seed", "max_steps"]
    history_df = history_df.copy()
    history_df["step"] = history_df["step"].astype(int)
    history_df.sort_values(["run_id", "step"], inplace=True)

    budget_s = pd.Series(budgets, name="budget").sort_values()

    # For each run, merge_asof finds the latest step <= each budget.
    parts = []
    for key, grp in history_df.groupby(run_cols, sort=False):
        run_id, model, game, framework, seed, max_steps = key
        max_steps = int(max_steps)
        run_budgets = budget_s[budget_s <= max_steps].reset_index(drop=True)
        if run_budgets.empty:
            continue

        bdf = run_budgets.to_frame()
        merged = pd.merge_asof(
            bdf, grp[["step", "normalized_score"]], left_on="budget", right_on="step"
        )
        merged["normalized_score"] = merged["normalized_score"].fillna(0.0)
        merged["run_id"] = run_id
        merged["model"] = model
        merged["game"] = game
        merged["framework"] = framework
        merged["seed"] = seed
        parts.append(
            merged[
                [
                    "run_id",
                    "model",
                    "game",
                    "framework",
                    "seed",
                    "budget",
                    "normalized_score",
                ]
            ]
        )

    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


def plot_budget(
    df: pd.DataFrame, output: Path, continuous: bool = False, log_x: bool = False
) -> None:
    """Line plot: TALES metric (avg normalized score across all games per seed) vs. step budget."""
    models = sorted(df["model"].unique())
    cmap = plt.get_cmap("tab10")

    fig, ax = plt.subplots(figsize=(10, 6))

    # TALES metric: for each (model, seed, budget), average normalized_score across games.
    tales = (
        df.groupby(["model", "seed", "budget"])["normalized_score"].mean().reset_index()
    )
    tales.rename(columns={"normalized_score": "tales_score"}, inplace=True)

    for i, model in enumerate(models):
        mdf = tales[tales["model"] == model]
        # Aggregate across seeds (mean ± std) for each budget.
        agg = mdf.groupby("budget")["tales_score"].agg(["mean", "std", "count"])
        agg = agg.sort_index().dropna(subset=["mean"])
        agg["std"] = agg["std"].fillna(0)

        if agg.empty:
            continue

        ax.plot(
            agg.index,
            agg["mean"],
            marker=None if continuous else "o",
            label=model,
            color=cmap(i),
            linewidth=1.5 if continuous else 2,
        )
        if (agg["count"] > 1).any():
            ax.fill_between(
                agg.index,
                agg["mean"] - agg["std"],
                agg["mean"] + agg["std"],
                alpha=0.12,
                color=cmap(i),
            )

        # Annotate final point with the score value.
        last = agg.iloc[-1]
        ax.annotate(
            f"{last['mean']:.3f}",
            (agg.index[-1], last["mean"]),
            textcoords="offset points",
            xytext=(8, 0),
            fontsize=9,
            color=cmap(i),
        )

    budgets = sorted(df["budget"].unique())
    n_games = df["game"].nunique()

    ax.set_xlabel("Step Budget", fontsize=12)
    ax.set_ylabel("TALES Score", fontsize=12)
    ax.set_title(
        f"TALES Score vs. Step Budget\n(avg. normalized score across {n_games} games)",
        fontsize=14,
    )
    if log_x:
        ax.set_xscale("log")
        ax.xaxis.set_major_formatter(plt.ScalarFormatter())
    if not continuous:
        ax.set_xticks(budgets)
    ax.set_ylim(bottom=0)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=10)
    fig.tight_layout()

    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=150, bbox_inches="tight")
    print(f"Saved plot to {output}")


def print_table(df: pd.DataFrame) -> None:
    """Print TALES scores per model and step budget."""
    tales = (
        df.groupby(["model", "seed", "budget"])["normalized_score"].mean().reset_index()
    )
    pivot = (
        tales.groupby(["model", "budget"])["normalized_score"].mean().unstack("budget")
    )
    budgets = sorted(df["budget"].unique())
    pivot = pivot.reindex(columns=budgets)
    print("\nTALES Score per Model and Step Budget:")
    print(pivot.round(4).to_string())
    print()


def plot_per_framework(
    df: pd.DataFrame, output: Path, continuous: bool = False, log_x: bool = False
) -> None:
    """One subplot per framework, TALES-style metric (avg across games) vs. budget."""
    frameworks = sorted(df["framework"].unique())
    models = sorted(df["model"].unique())
    cmap = plt.get_cmap("tab10")
    model_colors = {m: cmap(i) for i, m in enumerate(models)}

    n_fw = len(frameworks)
    cols = min(3, n_fw)
    rows = (n_fw + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(6 * cols, 4.5 * rows), squeeze=False)

    budgets = sorted(df["budget"].unique())

    for idx, fw in enumerate(frameworks):
        ax = axes[idx // cols][idx % cols]
        fdf = df[df["framework"] == fw]
        n_games = fdf["game"].nunique()
        has_data = False

        for model in models:
            mdf = fdf[fdf["model"] == model]
            if mdf.empty:
                continue

            tales = (
                mdf.groupby(["seed", "budget"])["normalized_score"].mean().reset_index()
            )
            agg = tales.groupby("budget")["normalized_score"].agg(
                ["mean", "std", "count"]
            )
            agg = agg.sort_index().reindex(budgets).dropna(subset=["mean"])
            agg["std"] = agg["std"].fillna(0)

            if agg.empty:
                continue

            has_data = True
            ax.plot(
                agg.index,
                agg["mean"],
                marker=None if continuous else "o",
                label=model,
                color=model_colors[model],
                linewidth=1.2 if continuous else 1.8,
            )
            if (agg["count"] > 1).any():
                ax.fill_between(
                    agg.index,
                    agg["mean"] - agg["std"],
                    agg["mean"] + agg["std"],
                    alpha=0.12,
                    color=model_colors[model],
                )

        ax.set_title(f"{fw} ({n_games} games)", fontsize=11)
        ax.set_xlabel("Step Budget")
        ax.set_ylabel("Avg. Normalized Score")
        if log_x:
            ax.set_xscale("log")
            ax.xaxis.set_major_formatter(plt.ScalarFormatter())
        elif has_data and not continuous:
            ax.set_xticks(budgets)
        ax.set_ylim(-0.05, 1.05)
        ax.grid(True, alpha=0.3)

    for idx in range(n_fw, rows * cols):
        axes[idx // cols][idx % cols].set_visible(False)

    handles, labels = axes[0][0].get_legend_handles_labels()
    if handles:
        fig.legend(
            handles,
            labels,
            loc="lower center",
            ncol=min(len(models), 5),
            fontsize=9,
        )

    fig.suptitle(
        "TALES Score vs. Step Budget — by Framework",
        fontsize=14,
        y=1.01,
    )
    fig.tight_layout()

    fw_path = output.with_stem(output.stem + "_by_framework")
    fw_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(fw_path, dpi=150, bbox_inches="tight")
    print(f"Saved per-framework plot to {fw_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--max-steps",
        type=int,
        nargs="+",
        default=[300],
        help="Filter runs by config.max_steps (one or more values, default: 300). "
        "Budget range goes up to the largest value.",
    )
    parser.add_argument(
        "--budget-step",
        type=int,
        default=50,
        help="Step interval for budget cutoffs (default: 50). "
        "Ignored when --continuous is set.",
    )
    parser.add_argument(
        "--continuous",
        action="store_true",
        help="Plot a smooth curve using every step instead of discrete budget points.",
    )
    parser.add_argument(
        "--log-x",
        action="store_true",
        help="Use logarithmic scale for the x-axis (step budget).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output path for the plot (default: analysis/score_vs_budget_{max_steps}.png)",
    )
    parser.add_argument(
        "--cache",
        type=Path,
        default=None,
        help="Cache per-step history to a CSV file. The cache stores raw per-step "
        "data and is reusable across different --budget-step values.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    max_budget = max(args.max_steps)
    step = 1 if args.continuous else args.budget_step
    budgets = list(range(step, max_budget + 1, step))
    if args.continuous:
        label = "continuous"
    else:
        steps_label = "_".join(str(s) for s in sorted(args.max_steps))
        label = f"{steps_label}_step{args.budget_step}"
    output = args.output or Path(f"analysis/score_vs_budget_{label}.png")

    # --- Load / update per-step history cache ---
    if args.cache and args.cache.exists():
        print(f"Loading cached history from {args.cache}")
        history_df = pd.read_csv(args.cache, low_memory=False)
    else:
        history_df = pd.DataFrame()

    # Always query wandb for the current run list and fetch any missing runs.
    # fetch_runs returns only the best (longest) run per (model, game, seed).
    runs = fetch_runs(max_steps_list=args.max_steps)
    if not runs and history_df.empty:
        print("No runs found. Exiting.", file=sys.stderr)
        sys.exit(1)

    selected_ids = {r.id for r in runs}
    cached_ids = (
        set(history_df["run_id"].unique()) if "run_id" in history_df.columns else set()
    )

    # Remove cached runs not in the selected set (stale or superseded).
    obsolete_ids = cached_ids - selected_ids
    # Also remove runs with old cache format (NaN step values).
    if "step" in history_df.columns:
        old_fmt = set(history_df.loc[history_df["step"].isna(), "run_id"].unique())
        obsolete_ids |= old_fmt
    if obsolete_ids:
        reason = "superseded/stale/old-format"
        print(f"  Removing {len(obsolete_ids)} cached run(s) ({reason}).")
        history_df = history_df[~history_df["run_id"].isin(obsolete_ids)]
        cached_ids -= obsolete_ids

    new_runs = [r for r in runs if r.id not in cached_ids]

    if new_runs:
        print(
            f"  Fetching {len(new_runs)} new run(s) "
            f"(skipping {len(runs) - len(new_runs)} cached)..."
        )
        new_df = build_history_table(new_runs)
        history_df = (
            pd.concat([history_df, new_df], ignore_index=True)
            if not history_df.empty
            else new_df
        )

    if history_df.empty:
        print("No score data found in runs. Exiting.", file=sys.stderr)
        sys.exit(1)

    cache_changed = bool(new_runs or obsolete_ids)
    if args.cache and (cache_changed or not args.cache.exists()):
        args.cache.parent.mkdir(parents=True, exist_ok=True)
        history_df.to_csv(args.cache, index=False)
        print(f"{'Updated' if cache_changed else 'Saved'} cache at {args.cache}")

    # --- Compute budget table from per-step history ---
    if args.continuous:
        print("\nComputing budgets at every step (continuous)...")
    else:
        print(f"\nComputing budgets at step interval {args.budget_step}...")
    df = compute_budgets(history_df, budgets)

    n_models = df["model"].nunique()
    n_games = df["game"].nunique()
    print(f"Data: {n_models} models, {n_games} games, budgets={budgets}")

    # Ensure framework column exists and has no missing values.
    if "framework" not in df.columns:
        df["framework"] = df["game"].map(_infer_framework)
    else:
        mask = df["framework"].isna()
        if mask.any():
            df.loc[mask, "framework"] = df.loc[mask, "game"].map(_infer_framework)

    # Drop (model, budget) pairs where the model hasn't completed all games.
    games_per_mb = (
        df.groupby(["model", "budget"])["game"].nunique().reset_index(name="n_games")
    )
    incomplete_mb = games_per_mb[games_per_mb["n_games"] < n_games]

    if not incomplete_mb.empty:
        total_budgets_per_model = games_per_mb.groupby("model").size()
        incomplete_budgets_per_model = incomplete_mb.groupby("model").size()

        for model in incomplete_budgets_per_model.index:
            if incomplete_budgets_per_model[model] == total_budgets_per_model[model]:
                max_count = int(
                    games_per_mb.loc[games_per_mb["model"] == model, "n_games"].max()
                )
                print(
                    f"  ⚠ Dropping {model}: only {max_count}/{n_games} games completed"
                )
            elif args.continuous:
                n_inc = int(incomplete_budgets_per_model[model])
                print(f"  ⚠ Skipping {model} at {n_inc} incomplete budget point(s)")
            else:
                rows = incomplete_mb[incomplete_mb["model"] == model]
                for _, row in rows.iterrows():
                    print(
                        f"  ⚠ Skipping {model} at budget {int(row['budget'])}: "
                        f"only {int(row['n_games'])}/{n_games} games"
                    )

        drop_keys = set(zip(incomplete_mb["model"], incomplete_mb["budget"]))
        df = df[~df.apply(lambda r: (r["model"], r["budget"]) in drop_keys, axis=1)]

    if df.empty:
        print("No complete data remaining. Exiting.", file=sys.stderr)
        sys.exit(1)

    if not args.continuous:
        print_table(df)
    plot_budget(df, output=output, continuous=args.continuous, log_x=args.log_x)
    plot_per_framework(df, output=output, continuous=args.continuous, log_x=args.log_x)


if __name__ == "__main__":
    main()
