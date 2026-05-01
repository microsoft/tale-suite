import json
import logging
import os
import tempfile

import pandas as pd
import wandb

log = logging.getLogger("tales")

WANDB_PROJECT = os.environ.get("WANDB_PROJECT", "tales")

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


def find_matching_run(env_name, agent_params, game_seed, project=WANDB_PROJECT):
    """Find a matching wandb run based on game and agent config fields.

    Searches the wandb project for finished runs that match the core experiment
    identity: game, LLM model, agent type, LLM seed, and game seed. Fields
    that may change between runs (like context_limit or max_steps) are
    intentionally excluded from matching.

    Among matching runs, returns the one with the most completed steps.
    The caller is responsible for truncating the trajectory if needed.

    Args:
        env_name: The game/environment name (e.g., "JerichoEnvZork1").
        agent_params: Dict of agent parameters (from agent.params).
        game_seed: The game seed (can be None).
        project: The wandb project path.

    Returns:
        The run ID of the best matching run, or None if no match found.
    """
    api = wandb.Api()

    # Match on stable identity fields only (not context_limit or max_steps).
    llm = agent_params.get("llm")
    agent_type = agent_params.get("agent_type")
    seed = agent_params.get("seed")

    log.info(
        f"Searching for matching run: game={env_name}, llm={llm}, "
        f"agent_type={agent_type}, seed={seed}, game_seed={game_seed}"
    )

    # Use wandb config filters for the fields that are top-level in config.
    filters = {
        "config.game": env_name,
        "state": "finished",
    }
    if llm is not None:
        filters["config.llm"] = llm
    if agent_type is not None:
        filters["config.agent_type"] = agent_type

    try:
        runs = api.runs(project, filters=filters, order="-created_at")
    except wandb.errors.CommError as e:
        log.warning(f"Failed to search wandb runs: {e}")
        return None

    # Filter by seed and game_seed, collecting candidates.
    candidates = []
    for run in runs:
        cfg = run.config
        if seed is not None and cfg.get("seed") != seed:
            continue
        if game_seed is not None and cfg.get("game_seed") != game_seed:
            continue

        run_steps = run.summary.get("total/Env. Steps", 0) or 0
        candidates.append((run, run_steps))

    if not candidates:
        log.warning(
            f"No matching run found for: game={env_name}, llm={llm}, "
            f"agent_type={agent_type}, seed={seed}, game_seed={game_seed}"
        )
        return None

    # Pick the run with the most steps (will be truncated to nb_steps if needed).
    best_run, best_steps = max(candidates, key=lambda x: x[1])

    log.info(
        f"Found matching run: {best_run.name} "
        f"({best_steps} steps, id={best_run.id}, url={best_run.url})"
    )
    return best_run.id


def fetch_run_trajectory(run_id, project=WANDB_PROJECT):
    """Fetch run config and rollout trajectory from wandb.

    Args:
        run_id: The wandb run ID (e.g., "abc123de").
        project: The wandb project path (e.g., "entity/project").

    Returns:
        A tuple of (run_config, trajectory_df) where:
        - run_config is a dict with the original run's configuration
          plus metadata (run_id, run_url, run_name).
        - trajectory_df is a DataFrame with one row per step,
          columns matching ROLLOUT_COLUMNS.

    Raises:
        ValueError: If the run or rollout data cannot be found.
    """
    api = wandb.Api()
    run_path = f"{project}/{run_id}"
    log.info(f"Fetching run {run_path} from wandb...")

    try:
        run = api.run(run_path)
    except wandb.errors.CommError as e:
        raise ValueError(f"Could not find wandb run '{run_path}': {e}") from e

    # Extract config.
    run_config = dict(run.config)
    run_config["_run_id"] = run.id
    run_config["_run_url"] = run.url
    run_config["_run_name"] = run.name
    run_config["_run_state"] = run.state

    # Download the rollout JSONL file.
    trajectory_df = _download_rollout(run)

    log.info(
        f"Fetched trajectory: {len(trajectory_df)} steps from run '{run.name}' ({run.state})"
    )
    return run_config, trajectory_df


def _download_rollout(run):
    """Download and parse the rollout JSONL file from a wandb run."""
    rollout_file = None
    for f in run.files():
        if f.name.endswith(".jsonl"):
            rollout_file = f
            break

    if rollout_file is None:
        raise ValueError(
            f"No rollout JSONL file found in wandb run '{run.id}'. "
            f"Available files: {[f.name for f in run.files()]}"
        )

    with tempfile.TemporaryDirectory() as tmpdir:
        rollout_file.download(root=tmpdir, replace=True)
        filepath = f"{tmpdir}/{rollout_file.name}"
        df = pd.read_json(filepath, orient="records", lines=True)

    # Validate columns.
    missing = set(ROLLOUT_COLUMNS) - set(df.columns)
    if missing:
        log.warning(f"Rollout is missing expected columns: {missing}")

    return df
