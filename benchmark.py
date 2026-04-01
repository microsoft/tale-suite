import argparse
import datetime
import glob
import importlib
import json
import logging
import os
import sys
import time
from functools import partial
from os.path import join as pjoin

import gymnasium as gym
import pandas as pd
import wandb
from termcolor import colored
from tqdm import tqdm

import tales
from tales.logger import log, setup_logging
from tales.utils import NumpyEncoder
from tales.wandb_utils import fetch_run_trajectory, find_matching_run

os.environ["WANDB_MODE"] = "disabled"


def _make_state(obs, info):
    """Create mutable game state dict."""
    return {
        "step": 0,
        "score": 0,
        "moves": 0,
        "highscore": 0,
        "max_score": info["max_score"],
        "nb_wins": 0,
        "nb_losts": 0,
        "nb_resets": 0,
        "nb_invalid_actions": 0,
        "obs": obs,
        "done": False,
        "info": info,
        "results": [],
    }


def _step_env(env, state, action):
    """Execute an action and update state. Returns (prev_obs, feedback)."""
    prev_obs = state["obs"]

    if "\n" in action.strip():
        state["obs"] = "The game only allows one action per step."
    else:
        obs, _, done, info = env.step(action)
        state["obs"] = obs
        state["done"] = done
        state["info"] = info

    state["score"] = state["info"]["score"]
    state["moves"] = state["info"]["moves"]
    state["highscore"] = max(state["score"], state["highscore"])

    return prev_obs, state["info"]["feedback"]


def _check_invalid(state, action, admissible_commands):
    """Track invalid actions when admissible commands are enabled."""
    if (
        admissible_commands
        and state["info"]["admissible_commands"]
        and action not in state["info"]["admissible_commands"]
    ):
        state["nb_invalid_actions"] += 1


def _handle_done(env, agent, state, env_name, break_on_max=True):
    """Handle game-over (win/loss/reset). Returns True if loop should break."""
    if not state["done"]:
        return False

    if state["info"]["won"]:
        state["nb_wins"] += 1
        if state["highscore"] == state["max_score"]:
            log.debug(state["obs"])
            if break_on_max:
                return True  # Break: no reason to play more.
            else:
                return False  # Replay: don't break or reset.
    elif state["info"]["lost"]:
        state["nb_losts"] += 1

    # Restart the game to try for a better score.
    last_obs = state["obs"]
    obs, info = env.reset()
    state["obs"] = last_obs + "\n\n-= Restarting =-\n" + obs
    state["info"] = info
    agent.reset(state["obs"], info, env_name)
    state["nb_resets"] += 1
    log.debug(state["obs"])
    return False


def _record_step(state, prev_obs, action, feedback, token_stats, wandb_run):
    """Append step to results and log to wandb."""
    s = state
    norm_score = s["score"] / s["max_score"]
    norm_highscore = s["highscore"] / s["max_score"]

    wandb_run.log(
        {
            "episode/moves": s["moves"],
            "episode/score": s["score"],
            "episode/highscore": s["highscore"],
            "episode/normalized_score": norm_score,
            "episode/normalized_highscore": norm_highscore,
            "episode/token_usage": token_stats["nb_tokens"],
            "episode/token_usage_thinking": token_stats.get("nb_tokens_thinking", 0),
        },
        step=s["step"],
    )

    # fmt: off
    s["results"].append([
        s["step"], s["score"], s["max_score"], norm_score, s["moves"],
        prev_obs, action, feedback,
        token_stats["prompt"], token_stats["response"], token_stats.get("thinking"),
        token_stats["nb_tokens"], token_stats["nb_tokens_prompt"],
        token_stats["nb_tokens_response"], token_stats.get("nb_tokens_thinking", 0),
    ])
    # fmt: on


def replay_trajectory(
    env, agent, trajectory_df, state, wandb_run, args, env_name, start_time
):
    """Replay recorded actions through the environment (no LLM calls).

    Feeds each action from the trajectory to the environment, verifies
    observations match, and builds agent history for subsequent LLM play.
    """
    replay_steps = len(trajectory_df)
    log.info(colored(f"Replaying {replay_steps} steps...", "cyan"))

    replay_pbar = tqdm(
        trajectory_df.iterrows(),
        total=args.nb_steps,
        desc=f"  {env_name} (replay)",
        unit="steps",
        leave=False,
    )
    for _, row in replay_pbar:
        state["step"] = int(row["Step"])
        action = str(row["Action"])

        replay_pbar.set_postfix_str(
            f"Score: {state['info']['score']}/{state['info']['max_score']}"
            f" ({state['info']['score']/state['info']['max_score']:.1%})"
        )

        prev_obs, feedback = _step_env(env, state, action)
        _check_invalid(state, action, args.admissible_commands)

        # Verify replay fidelity.
        logged_obs = row.get("Observation")
        if logged_obs is not None and isinstance(logged_obs, str):
            if prev_obs.strip() != logged_obs.strip():
                log.warning(
                    f"Replay divergence at step {state['step']}:\n"
                    f"  Expected: {logged_obs[:200]!r}\n"
                    f"  Got:      {prev_obs[:200]!r}"
                )

        # Build agent history for subsequent LLM calls.
        agent.history.append((f"{prev_obs}\n> ", f"{action}\n"))

        msg = "{:5d}. Time: {:9.2f}\tScore: {:3d}\tMove: {:5d}\tAction: {:20s} (replay)"
        msg = msg.format(
            state["step"],
            time.time() - start_time,
            state["score"],
            state["moves"],
            action,
        )
        log.info(msg)

        # Use original token stats from the trajectory.
        token_stats = {
            "prompt": row.get("Prompt", "") or "",
            "response": row.get("Response", "") or "",
            "thinking": row.get("Thinking"),
            "nb_tokens": row.get("Token Usage", 0) or 0,
            "nb_tokens_prompt": row.get("Prompt Tokens", 0) or 0,
            "nb_tokens_response": row.get("Response Tokens", 0) or 0,
            "nb_tokens_thinking": row.get("Thinking Tokens", 0) or 0,
        }
        _record_step(state, prev_obs, action, feedback, token_stats, wandb_run)

        if not state["done"]:
            log.debug(state["obs"])

        _handle_done(env, agent, state, env_name, break_on_max=False)

    replay_pbar.close()

    if state["highscore"] == state["max_score"]:
        log.info(
            colored(
                f"Replay complete: game already won with max score "
                f"({state['highscore']}/{state['max_score']}). No further steps needed.",
                "green",
            )
        )
        return args.nb_steps  # Signal: skip the play loop.
    else:
        log.info(
            colored(
                f"Replay complete: {replay_steps} steps, score={state['score']}, "
                f"highscore={state['highscore']}. "
                f"LLM takes over from step {replay_steps + 1}.",
                "cyan",
            )
        )
        return replay_steps


def play_with_agent(
    env, agent, state, wandb_run, args, env_name, start_time, start_step
):
    """Play the game with the LLM agent from start_step to nb_steps."""
    pbar = tqdm(
        range(start_step, args.nb_steps + 1),
        initial=start_step - 1,
        total=args.nb_steps,
        desc=f"  {env_name}",
        unit="steps",
        leave=False,
    )
    for step in pbar:
        state["step"] = step
        pbar.set_postfix_str(
            f"Score: {state['info']['score']}/{state['info']['max_score']}"
            f" ({state['info']['score']/state['info']['max_score']:.1%})"
        )

        action, stats = agent.act(
            state["obs"], state["score"], state["done"], state["info"]
        )
        log.debug(colored(f"> {action}", "green"))

        if args.debug:
            breakpoint()

        prev_obs, feedback = _step_env(env, state, action)
        _check_invalid(state, action, args.admissible_commands)

        msg = "{:5d}. Time: {:9.2f}\tScore: {:3d}\tMove: {:5d}\tAction: {:20s}"
        msg = msg.format(
            step, time.time() - start_time, state["score"], state["moves"], action
        )
        log.info(msg)

        token_stats = {
            "prompt": stats["prompt"],
            "response": stats["response"],
            "thinking": stats.get("thinking"),
            "nb_tokens": stats["nb_tokens"],
            "nb_tokens_prompt": stats["nb_tokens_prompt"],
            "nb_tokens_response": stats["nb_tokens_response"],
            "nb_tokens_thinking": stats.get("nb_tokens_thinking", 0),
        }
        _record_step(state, prev_obs, action, feedback, token_stats, wandb_run)

        if not state["done"]:
            log.debug(state["obs"])

        if _handle_done(env, agent, state, env_name, break_on_max=True):
            break


def evaluate(agent, env_name, args):
    # Fetch trajectory if continuing from a previous run.
    trajectory_df = None
    continue_from = getattr(args, "continue_from", None)
    if continue_from:
        # Auto-find matching run if no explicit run ID was provided.
        if continue_from == "auto":
            continue_from = find_matching_run(env_name, agent.params, args.game_seed)
            if continue_from is None:
                log.info(
                    colored(
                        f"No matching previous run found for {env_name}. "
                        f"Running from scratch.",
                        "yellow",
                    )
                )

    if continue_from:
        original_config, trajectory_df = fetch_run_trajectory(continue_from)

        # Validate that the game matches.
        original_game = original_config.get("game")
        if original_game and original_game != env_name:
            raise ValueError(
                f"Environment mismatch: --continue-from run played '{original_game}' "
                f"but current run targets '{env_name}'."
            )

        # Override game_seed from original run to ensure deterministic replay.
        original_seed = original_config.get("game_seed")
        if original_seed is not None and original_seed != args.game_seed:
            log.info(
                f"Overriding --game-seed from {args.game_seed} to {original_seed} "
                f"(from original run)."
            )
            args.game_seed = original_seed

        # Truncate trajectory if it has more steps than the target.
        if len(trajectory_df) > args.nb_steps:
            log.info(
                f"Trajectory has {len(trajectory_df)} steps but target is {args.nb_steps}. "
                f"Truncating replay to {args.nb_steps} steps."
            )
            trajectory_df = trajectory_df.iloc[: args.nb_steps]

        replay_steps = len(trajectory_df)
        log.info(
            colored(
                f"Continuing from run {continue_from}: "
                f"replaying {replay_steps} steps, then LLM takes over up to {args.nb_steps}.",
                "cyan",
            )
        )

    env_params = (
        f"a{int(args.admissible_commands)}_s{args.game_seed}_steps{args.nb_steps}"
    )
    logdir = pjoin(args.log_dir, f"{env_name}")
    os.makedirs(logdir, exist_ok=True)
    summary_file = pjoin(logdir, f"{env_params}.json")
    rollouts_file = pjoin(logdir, f"{env_params}.jsonl")
    log_file = pjoin(logdir, f"{env_params}.log")

    # Create new file handler for this env evaluation.
    fh = log.add_new_file_handler(log_file)

    # Check if the game has already been evaluated.
    if not args.force_all and os.path.exists(summary_file):
        log.info(f"Previous evaluation found: {summary_file}")
        with open(summary_file) as reader:
            summary = json.load(reader)

        log.info(f"Previous evaluation status: {summary['status']}")
        if not args.force_failed or summary["status"] == "finished":
            log.info(colored("Skipped, already done.", "yellow"))
            log.removeHandler(fh)
            return summary

    run_name = f"{env_name} - {agent.uid}"
    if args.wandb:  # and not args.force_all:
        # Check if there already exists a run with the same name using Wandb API.
        wandb_api = wandb.Api()
        wandb_runs = wandb_api.runs(
            "pearls-lab/text-games-benchmark",
            filters={
                "display_name": run_name,
                "tags": {"$ne": "without-help"},
            },
        )
        if wandb_runs:
            wandb_run = wandb_runs[0]
            log.info(f"Previous evaluation found: {wandb_run.url} ({wandb_run.state})")
            if wandb_run.state in ("finished", "running"):
                log.info(colored("Skipped, already exists.", "yellow"))
                log.removeHandler(fh)
                summary = {
                    "status": wandb_run.state,
                    "env_name": env_name,
                    "env_params": env_params,
                    "wandb_run_id": wandb_run.id,
                    "wandb_url": wandb_run.url,
                    "nb_steps": wandb_run.summary["total/Env. Steps"],
                    "nb_moves": wandb_run.summary["total/Game Moves"],
                    "nb_invalid_actions": wandb_run.summary["total/Invalid Actions"],
                    "nb_losts": wandb_run.summary["total/Losts"],
                    "nb_wins": wandb_run.summary["total/Wins"],
                    "nb_resets": wandb_run.summary["total/Resets"],
                    "highscore": wandb_run.summary["final/Highscore"],
                    "max_score": wandb_run.summary["final/Game Max Score"],
                    "norm_score": wandb_run.summary["final/Normalized Score"],
                    "duration": wandb_run.summary["final/Duration"],
                }
                return summary

    # initialize wandb
    wandb_config = {
        "version": tales.__version__,
        "game": env_name,
        "framework": tales.env2task[env_name],
        "agent": agent.uid,
        "max_steps": args.nb_steps,
        "game_seed": args.game_seed,
        "admissible_commands": args.admissible_commands,
        **agent.params,
    }
    if trajectory_df is not None:
        wandb_config["continued_from_run_id"] = original_config["_run_id"]
        wandb_config["continued_from_run_url"] = original_config["_run_url"]
        wandb_config["replay_steps"] = len(trajectory_df)
    wandb_run = wandb.init(
        project="tales",
        config=wandb_config,
        reinit=True,
        name=run_name,
    )

    env = gym.make(
        f"tales/{env_name}-v0",
        disable_env_checker=True,
        admissible_commands=args.admissible_commands,
    )

    log.debug(f"Using {env.__class__.__name__}")
    log.debug(f"Playing {env_name}")

    start_time = time.time()
    obs, info = env.reset(seed=args.game_seed)

    agent = agent.new()
    agent.reset(obs, info, env_name)

    log.debug(f"Environment reset.\n{obs}\n")

    state = _make_state(obs, info)

    wandb_run.log(
        {
            "episode/moves": 0,
            "episode/score": 0,
            "episode/highscore": 0,
            "episode/normalized_score": 0,
            "episode/normalized_highscore": 0,
            "episode/token_usage": 0,
        },
        step=0,
    )

    # Replay phase (if continuing from a previous run).
    replay_steps = 0
    if trajectory_df is not None:
        replay_steps = replay_trajectory(
            env, agent, trajectory_df, state, wandb_run, args, env_name, start_time
        )

    # Play phase (LLM-driven).
    status = "running"
    try:
        play_with_agent(
            env,
            agent,
            state,
            wandb_run,
            args,
            env_name,
            start_time,
            start_step=replay_steps + 1,
        )
        status = "finished"

    except KeyboardInterrupt as e:
        status = "killed"
        log.critical(colored(f"{env_name} (killed)", "red"))
        log.error(str(e))
        time.sleep(1)
        if args.debug:
            raise

    except Exception as e:
        status = "failed"
        log.critical(colored(f"{env_name} ({e!r})", "red"))
        log.error(str(e), exc_info=True)
        if args.debug:
            raise

    env.close()

    final_stats = {
        "nb_steps": state["step"],
        "nb_moves": state["moves"],
        "nb_invalid_actions": state["nb_invalid_actions"],
        "nb_losts": state["nb_losts"],
        "nb_wins": state["nb_wins"],
        "nb_resets": state["nb_resets"],
        "highscore": state["highscore"],
        "max_score": state["max_score"],
        "norm_score": state["highscore"] / state["max_score"],
        "duration": time.time() - start_time,
    }

    # fmt: off
    columns = [
        "Step", "Score", "Max Score", "Normalized Score", "Moves",
        "Observation", "Action", "Feedback",
        "Prompt", "Response", "Thinking",
        "Token Usage", "Prompt Tokens", "Response Tokens", "Thinking Tokens",
    ]
    # fmt: on
    df = pd.DataFrame(state["results"], columns=columns)
    df.to_json(rollouts_file, orient="records", lines=True)

    wandb_stats = {
        "total/Env. Steps": final_stats["nb_steps"],
        "total/Game Moves": final_stats["nb_moves"],
        "total/Invalid Actions": final_stats["nb_invalid_actions"],
        "total/Losts": final_stats["nb_losts"],
        "total/Wins": final_stats["nb_wins"],
        "total/Resets": final_stats["nb_resets"],
        "total/Tokens": df["Token Usage"].sum(),
        "total/Prompt Tokens": df["Prompt Tokens"].sum(),
        "total/Response Tokens": df["Response Tokens"].sum(),
        "total/Thinking Tokens": df["Thinking Tokens"].sum(),
        "final/Highscore": final_stats["highscore"],
        "final/Game Max Score": final_stats["max_score"],
        "final/Normalized Score": final_stats["norm_score"],
        "final/Duration": final_stats["duration"],
    }
    wandb_run.log(
        {"episode/rollout": wandb.Table(dataframe=df), **wandb_stats},
        step=final_stats["nb_steps"],
    )

    # Save summary.
    summary = {
        "status": status,
        "env_name": env_name,
        "env_params": env_params,
        "wandb_run_id": wandb_run.id,
        "wandb_url": wandb_run.url,
        **final_stats,
        **wandb_stats,
    }

    with open(summary_file, "w") as f:
        json.dump(summary, f, indent=2, sort_keys=True, cls=NumpyEncoder)

    wandb.save(rollouts_file)
    wandb.save(log_file)
    wandb.save(summary_file)

    wandb_run.finish(exit_code=int(status != "finished"))

    log.removeHandler(fh)
    return summary


def benchmark(agent, args):
    # Log how many envs we are about to evaluate.
    log.critical("Evaluating {} interactive text environments:".format(len(args.envs)))

    mean_score = 0
    total_time = 0.0
    total_steps = 0
    total_invalid = 0

    nb_envs = 0
    max_env_name = max(map(len, args.envs))
    for env in tqdm(args.envs, desc="Benchmarking", unit="game", leave=False):
        summary = evaluate(agent, env, args)

        nb_envs += 1

        total_time += summary["duration"]  # In seconds
        total_steps += summary["nb_steps"]
        total_invalid += summary["nb_invalid_actions"]

        msg = (
            f"{env.ljust(max_env_name)}"
            f"  Steps: {summary['nb_steps']:4d}/{args.nb_steps:4d}"
            f"  Time: {datetime.timedelta(seconds=int(summary['duration']))}"
            f"{summary['nb_resets']:4d} resets"
            f"  Score: {summary['highscore']:3d}/{summary['max_score']:3d} ({summary['norm_score']:6.2%})"
        )

        log.critical(msg)

        mean_score += summary["norm_score"]

    if nb_envs > 0 and total_time > 0:
        log.critical(
            f"Mean score (over {nb_envs} games) = {mean_score / nb_envs:8.2%} of total possible"
        )
        log.critical(f"Total time {total_time:9.2f} seconds")
        log.critical(f"Total {total_invalid} invalid actions")
        log.critical(f"Avg. speed: {total_steps / total_time:8.2f} steps per second")


def pretty_print_tasks(num_cols: int = 3, disable_print: bool = False):
    output = []

    max_justify = max(
        len(env_name) for task in tales.envs_per_task.values() for env_name in task
    )

    for task in sorted(tales.envs_per_task):
        task_output = f"{'=' * 5} {task} {'=' * 5}\n"

        # Reference: https://stackoverflow.com/a/33464001
        for count, env_id in enumerate(sorted(tales.envs_per_task[task]), 1):
            # Print column with justification.
            task_output += env_id.ljust(max_justify) + " "

            # Once all rows printed, switch to new column.
            if count % num_cols == 0:
                task_output = task_output.rstrip(" ")

                if count != len(tales.envs_per_task[task]):
                    task_output += "\n"

        output.append(task_output.rstrip(" "))

    if disable_print:
        return "\n".join(output)
    else:
        print("\n".join(output))


def exit_listing_agents(agent=None):
    msg = ""
    if agent is not None:
        msg += "Unknown agent: {}\n\n".format(agent)

    msg += "Available agents:\n  "
    msg += "\n  ".join(sorted(tales.agent.AGENTS))
    print(msg)
    sys.exit(1)


def _maybe_load_agent_module():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument(
        "--agent",
        default="agents/*.py",
        help="Load external python file(s). Useful to register custom agent on-the-fly. Default: %(default)s",
    )
    args, _ = parser.parse_known_args()
    if args.agent:
        print(f"Importing agent(s) from {args.agent}.")
        for agent_file in glob.glob(args.agent):
            print(f"Importing {agent_file}...")
            agent_dirname = os.path.dirname(agent_file)
            agent_filename, _ = os.path.splitext(os.path.basename(agent_file))
            if f"{agent_dirname}.{agent_filename}" in sys.modules:
                continue

            spec = importlib.util.spec_from_file_location(
                f"{agent_dirname}.{agent_filename}", agent_file
            )
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)


def parse_args():
    # fmt: off

    description = "Benchmark some agent on interactive text environments."
    general_parser = argparse.ArgumentParser(add_help=False, description=description)
    general_parser.add_argument("--agent", default="./agents/*",
                                help="Load external python file(s). Useful to register custom agent on-the-fly. Default: %(default)s")

    parser = argparse.ArgumentParser(parents=[general_parser])
    subparsers = parser.add_subparsers(dest="subcommand", title='Available agents to benchmark')

    def _add_general_settings(parser):

        parser.formatter_class = argparse.RawTextHelpFormatter
        general_group = parser.add_argument_group('General settings')

        general_group.add_argument("--envs", metavar="env", nargs="+", choices=tales.envs + tales.tasks,
                            help="Interactive text environments to evaluate the agent(s)."
                                f" Available:\n{pretty_print_tasks(disable_print=True)}")
        general_group.add_argument("--game-seed", type=int,
                            help="Seed for the game. Default: game-specific one.")
        general_group.add_argument("--nb-steps", type=int, default=100,
                            help="Maximum number of steps per game.")
        general_group.add_argument("--admissible-commands", action="store_true",
                            help="Enable admissible commands.")

        general_group.add_argument("--log-dir", default="logs",
                            help="Folder where to save verbose log information.")

        general_group.add_argument("--wandb", action="store_true",
                            help="Log to wandb")
        general_group.add_argument("-ff", "--force-all", action="store_true",
                            help="Force overwriting existing log files.")
        general_group.add_argument("-f", "--force-failed", action="store_true",
                            help="Force overwriting only log files that have failed.")
        general_group.add_argument("--debug", action="store_true",
                            help="Debug mode.")
        general_group.add_argument("--continue-from", dest="continue_from",
                            nargs="?", const="auto",
                            help="Continue from a previous wandb run. "
                                 "Pass a run ID to replay a specific run, or use without a value to auto-find "
                                 "a matching run based on the current config (game, agent, seed).")

        subgroup = general_group.add_mutually_exclusive_group()
        subgroup.add_argument(
            "-v", "--verbose", dest="logging_level",
            action="store_const", const=logging.INFO, default=logging.CRITICAL,
            help="Display actions taken.",
        )
        subgroup.add_argument(
            "-vv", "--very-verbose", dest="logging_level",
            action="store_const", const=logging.DEBUG, default=logging.CRITICAL,
            help="Display actions and game observations.",
        )
        subgroup.add_argument(
            "--logging-level", dest="logging_level", default=logging.CRITICAL,
            choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
            help="Set a specific logging level",
        )

    _add_general_settings(parser)

    agent_parsers = []
    for challenge_name, (desc, _, add_agent_arguments) in tales.agent.AGENTS.items():
        agent_parser = subparsers.add_parser(challenge_name, help=desc)
        add_agent_arguments(agent_parser)
        _add_general_settings(agent_parser)
        agent_parsers.append(agent_parser)

    return parser.parse_args()
    # fmt: on


def main():
    _maybe_load_agent_module()
    args = parse_args()

    if args.subcommand is None:
        print("Need to specify which type of agent to benchmark.")
        exit_listing_agents(args.subcommand)

    # Instanciate the agent.
    _, Agent, _ = tales.agent.AGENTS[args.subcommand]
    agent = Agent(**vars(args))
    agent.new = partial(Agent, **vars(args))

    # Create logging directory.
    args.log_dir = pjoin(args.log_dir, f"tales_{agent.uid.replace('/', '-')}")
    os.makedirs(args.log_dir, exist_ok=True)
    setup_logging(args)
    log.critical(
        colored(f"Logs will be saved in {os.path.abspath(args.log_dir)}", "magenta")
    )

    if args.wandb:
        os.environ["WANDB_MODE"] = "online"
        os.environ.pop("WANDB_RUN_ID", None)

    args.envs = args.envs or tales.envs
    args.envs = [  # Expand tasks into their respective environments.
        env
        for task in args.envs
        for env in (tales.envs_per_task[task] if task in tales.tasks else [task])
    ]

    benchmark(agent, args)


if __name__ == "__main__":
    main()
