import argparse
import datetime
import importlib
import logging
import os
import platform
import sys
import time
from functools import partial
from os.path import join as pjoin
from typing import List

import gymnasium as gym
import pandas as pd
import wandb
from termcolor import colored
from tqdm import tqdm

import twbench
from twbench.utils import log

os.environ["WANDB_MODE"] = "disabled"


def evaluate(agent, env_name, args, wandb_run):

    env = gym.make(
        f"twbench/{env_name}-v0",
        disable_env_checker=True,
        admissible_commands=args.admissible_commands,
    )
    env.unwrapped.seed(args.game_seed)

    log.debug("Using {}".format(env.__class__.__name__))

    start_time = time.time()
    obs, infos = env.reset()

    agent = agent.new()
    agent.reset(obs, infos)

    log.debug(f"Environment reset.\n{obs}\n")

    max_score = infos["max_score"]
    nb_resets = 0
    nb_wins = 0
    nb_losts = 0
    nb_resets = 0
    nb_invalid_actions = 0
    highscore = 0
    score = 0
    done = False
    results = []

    # Hard coding for personal testing
    max_tokens = 10000

    # for step in tqdm(
    #     range(1, args.nb_steps + 1), desc=f"  {env_name}", unit="steps", leave=False
    # ):
    progress_bar = tqdm(
        total=max_tokens, desc=f"  {env_name}", unit="tokens generated", leave=False
    )
    tokens_generated = 0
    step = 0
    while tokens_generated < max_tokens:
        action, stats = agent.act(obs, score, done, infos)
        step += 1

        progress_bar.update(stats["new_tokens"])
        tokens_generated += stats["new_tokens"]

        log.debug(colored(f"> {action}", "green"))

        if args.debug:
            breakpoint()

        prev_obs = obs
        obs, _, done, infos = env.step(action)
        score = infos["score"]
        moves = infos["moves"]
        feedback = infos["feedback"]

        if (
            args.admissible_commands
            and infos["admissible_commands"]
            and action not in infos["admissible_commands"]
        ):
            nb_invalid_actions += 1

        msg = "{:5d}. Time: {:9.2f}\tScore: {:3d}\tMove: {:5d}\tAction: {:20s}"
        msg = msg.format(step, time.time() - start_time, score, moves, action)
        log.info(msg)
        norm_score = score / max_score

        wandb_run.log(
            {
                "episode/moves": moves,
                "episode/score": score,
                "episode/normalized_score": norm_score,
            },
            step=step + 1,
        )

        # fmt: off
        #table.add_data(
        results.append([
            step, score, max_score, norm_score, moves,
            prev_obs, action, feedback, stats["prompt"], stats["response"], stats["nb_tokens"]
        ])
        # fmt: on

        log.debug(obs)

        if done:
            highscore = max(score, highscore)

            if infos["won"]:
                nb_wins += 1
                if highscore == max_score:
                    break  # No reason to play that game more.
            elif infos["lost"]:
                nb_losts += 1
            else:
                assert True, "Games should either end with a win or a fail."

            # Replay the game in the hope of achieving a better score.
            last_obs = obs
            obs, infos = env.reset()
            obs = last_obs + "\n-= Restarting =-\n" + obs
            agent.reset(obs, infos)
            nb_resets += 1

            log.debug(f"Environment reset.\n{obs}\n")

    env.close()

    # Keep highest score.
    highscore = max(score, highscore)

    stats = {
        "nb_steps": step,
        "nb_moves": moves,
        "nb_invalid_actions": nb_invalid_actions,
        "nb_losts": nb_losts,
        "nb_wins": nb_wins,
        "nb_resets": nb_resets,
        "highscore": highscore,
        "max_score": max_score,
        "norm_score": highscore / max_score,
        "duration": time.time() - start_time,
    }

    return stats, results


def benchmark(agent, games, args):
    # Log games we are about to evaluate.
    log.critical("Evaluating {} games:".format(len(games)))

    mean_score = 0
    total_time = 0.0
    total_steps = 0
    total_invalid = 0

    nb_games = 0
    max_game_name = max(len(os.path.basename(game)) for game in games)
    with tqdm(total=len(games), desc="Benchmarking", unit="game", leave=False) as pbar:
        for game in games:
            total_steps = 0
            game_name = os.path.basename(game)
            logfile = pjoin(
                args.log_dir,
                f"{game_name}",
                f"a{int(args.admissible_commands)}_s{args.game_seed}.jsonl",
            )
            os.makedirs(os.path.dirname(logfile), exist_ok=True)
            if os.path.exists(logfile) and not args.force:
                pbar.write(colored(f"{game_name} (skip done)", "yellow"))
                log.info(
                    f"Skipping {game_name} evaluation, already saved in {logfile}."
                )
                pbar.update(1)
                continue  # Skip games that have already been evaluated.

            pbar.set_postfix_str(game_name)

            wandb_config = {
                "game": game_name,
                "agent": agent.uid,
                # "llm": args.llm,
                # "seed": args.seed,
                # "context": args.context_limit,
                # "act-temp": args.act_temp,
                # "cot-temp": args.cot_temp,
                # "conversation": args.conversation,
                "max_steps": args.nb_steps,
                "admissible_commands": args.admissible_commands,
                **agent.params,
            }
            wandb_run = wandb.init(
                project="text-games-benchmark", config=wandb_config, reinit=True
            )

            try:
                stats, results = evaluate(agent, game, args, wandb_run)
            except ValueError as e:
                pbar.write(colored(f"{game_name} (error)", "red"))
                log.error(str(e))
                pbar.update(1)
                if args.debug:
                    raise

                continue  # Skip not supported games.

            nb_games += 1

            total_time += stats["duration"]  # In seconds
            total_steps += stats["nb_steps"]
            total_invalid += stats["nb_invalid_actions"]

            msg = (
                f"{game_name.ljust(max_game_name)}"
                f"  Steps: {stats['nb_steps']:4d}/{args.nb_steps:4d}"
                f"  Time: {datetime.timedelta(seconds=int(stats['duration']))}"
                f"{stats['nb_resets']:4d} resets"
                f"  Score: {stats['highscore']:3d}/{stats['max_score']:3d} ({stats['norm_score']:6.2%})"
            )

            log.info(msg)
            pbar.write(msg)
            pbar.update(1)

            mean_score += stats["norm_score"]

            # fmt: off
            columns = [
                "Step", "Score", "Max Score", "Normalized Score", "Moves",
                "Observation", "Action", "Feedback", "Prompt", "Response", "Token Usage"
            ]
            # fmt: on
            df = pd.DataFrame(results, columns=columns)
            df.to_json(logfile, orient="records", lines=True)

            wandb_run.log(
                {
                    "episode/rollout": wandb.Table(dataframe=df),
                    "total/Env. Steps": stats["nb_steps"],
                    "total/Game Moves": stats["nb_moves"],
                    "total/Invalid Actions": stats["nb_invalid_actions"],
                    "total/Losts": stats["nb_losts"],
                    "total/Wins": stats["nb_wins"],
                    "total/Resets": stats["nb_resets"],
                    "final/Highscore": stats["highscore"],
                    "final/Game Max Score": stats["max_score"],
                    "final/Normalized Score": stats["norm_score"],
                    "final/Duration": stats["duration"],
                }
            )
            wandb_run.finish()

    if nb_games > 0 and total_time > 0:
        log.critical(
            "Mean score (over {} games) = {:8.4f}% of total possible".format(
                nb_games, mean_score / nb_games
            )
        )
        log.critical("Total time {:9.2f} seconds".format(total_time))
        log.critical("Total {} invalid actions".format(total_invalid))
        log.critical(
            "Avg. speed: {:8.2f} steps per second".format(total_steps / total_time)
        )


class TqdmLoggingHandler(logging.Handler):
    def __init__(self, level=logging.NOTSET):
        super().__init__(level)

    def emit(self, record):
        try:
            msg = self.format(record)
            tqdm.write(msg)
            self.flush()
        except (KeyboardInterrupt, SystemExit):
            raise
        except Exception:
            self.handleError(record)


def setup_logging(args):
    log.setLevel(logging.DEBUG)

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    fh = logging.FileHandler(pjoin(args.log_dir, f"{timestamp}.log"), mode="w")
    formatter = logging.Formatter("%(asctime)s: %(message)s")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(formatter)
    log.addHandler(fh)

    ch = TqdmLoggingHandler()
    formatter = logging.Formatter("%(message)s")
    ch.setFormatter(formatter)
    log.addHandler(ch)

    ch.setLevel(logging.CRITICAL)
    if args.verbose:
        ch.setLevel(logging.INFO)

    if args.very_verbose:
        ch.setLevel(logging.DEBUG)


def pretty_print_tasks(num_cols: int = 3, disable_print: bool = False):
    output = []

    max_justify = max(
        len(env_name) for task in twbench.envs_per_task.values() for env_name in task
    )

    for task in sorted(twbench.envs_per_task):
        task_output = f"{'=' * 5} {task} {'=' * 5}\n"

        # Reference: https://stackoverflow.com/a/33464001
        for count, env_id in enumerate(sorted(twbench.envs_per_task[task]), 1):
            # Print column with justification.
            task_output += env_id.ljust(max_justify) + " "

            # Once all rows printed, switch to new column.
            if count % num_cols == 0:
                task_output = task_output.rstrip(" ")

                if count != len(twbench.envs_per_task[task]):
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
    msg += "\n  ".join(sorted(twbench.agent.AGENTS))
    print(msg)
    sys.exit(1)


def _maybe_load_agent_module():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--agent")
    args, _ = parser.parse_known_args()
    if args.agent:
        print(f"Importing agent(s) from {args.agent}.")

        import importlib

        spec = importlib.util.spec_from_file_location("twbench.agents", args.agent)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)


def parse_args():
    # fmt: off

    description = "Benchmark some agent on interactive text environments."
    general_parser = argparse.ArgumentParser(add_help=False, description=description)
    general_parser.add_argument("--agent", default="./agents/random.py",
                                help="Load an external python file. Useful to register custom challenges on-the-fly. Default: %(default)s")

    parser = argparse.ArgumentParser(parents=[general_parser])
    subparsers = parser.add_subparsers(dest="subcommand", title='Agents to benchmark')

    # parser.add_argument("--llm", default="gpt-4o-mini",
    #                     help="LLM to be used for evaluation. Default: %(default)s")
    # parser.add_argument("--seed", type=int, default=20241001,
    #                     help="Seed for LLM (not all endpoints support this). Default: %(default)s")
    # parser.add_argument("--cot-temp", type=float, default=0.0,
    #                     help="Temperature for LLM when doing chaint-of-thoughts. Default: %(default)s")
    # parser.add_argument("--act-temp", type=float, default=0.0,
    #                     help="Temperature for LLM when taking actions. Default: %(default)s")
    # parser.add_argument("--context-limit", type=int, default=10,
    #                     help="Limit context for LLM (in conversation turns). Default: %(default)s")
    # parser.add_argument("--conversation", action="store_true",
    #                     help="Enable conversation mode. Otherwise, use single prompt.")

    def _add_general_settings(parser):

        parser.formatter_class = argparse.RawTextHelpFormatter
        general_group = parser.add_argument_group('General settings')

        #general_group = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
        general_group.add_argument("--envs", metavar="env", nargs="+", choices=twbench.envs + twbench.tasks,
                            help="Interactive text environments to evaluate the agent(s)."
                                f" Available:\n{pretty_print_tasks(disable_print=True)}")
        general_group.add_argument("--game-seed", type=int, default=20241001,
                            help="Seed for the game. Default: %(default)s.")
        general_group.add_argument("--nb-steps", type=int, default=1000,
                            help="Maximum number of steps per game.")
        general_group.add_argument("--admissible-commands", action="store_true",
                            help="Enable admissible commands.")

        general_group.add_argument("--log-dir", default="logs",
                            help="Folder where to save verbose log information.")

        general_group.add_argument("--wandb", action="store_true",
                            help="Log to wandb")
        general_group.add_argument("-f", "--force", action="store_true",
                            help="Force overwriting existing log files.")
        general_group.add_argument("-v", "--verbose", action="store_true",
                            help="Enable verbose mode.")
        general_group.add_argument("-vv", "--very-verbose", action="store_true",
                            help="Display actions taken.")
        general_group.add_argument("--debug", action="store_true",
                            help="Debug mode.")
        

    agent_parsers = []
    for challenge_name, (desc, _, add_agent_arguments) in twbench.agent.AGENTS.items():
        agent_parser = subparsers.add_parser(challenge_name, help=desc)
        add_agent_arguments(agent_parser)
        _add_general_settings(agent_parser)
        agent_parsers.append(agent_parser)

    #return parser, agent_parsers
    return parser.parse_args()
    # fmt: on


def main():
    _maybe_load_agent_module()
    args = parse_args()

    if args.subcommand is None:
        print("Need to specify which type of agent to benchmark.")
        exit_listing_agents(args.subcommand)

    args.verbose = args.verbose or args.very_verbose

    # # Dynamically load agent class.
    # path, klass = args.agent.split(":")
    # spec = importlib.util.spec_from_file_location("twbench.agents", path)
    # mod = importlib.util.module_from_spec(spec)
    # spec.loader.exec_module(mod)
    # if not hasattr(mod, klass):
    #     msg = "python file '{}' has no class '{}'".format(path, klass)
    #     raise AttributeError(msg)

    # Instanciate the agent.
    # Agent = getattr(mod, klass)
    # agent = Agent(**vars(args))
    _, Agent, _ = twbench.agent.AGENTS[args.subcommand]
    agent = Agent(**vars(args))
    agent.new = partial(Agent, **vars(args))

    # Create logging directory.
    args.log_dir = pjoin(args.log_dir, f"tw-bench_{agent.uid}")
    os.makedirs(args.log_dir, exist_ok=True)
    setup_logging(args)

    if args.wandb:
        os.environ["WANDB_MODE"] = "online"

    # Log some info about the machine.
    log.info(f"args = {args}")
    log.info(f"system = {platform.system()}")
    log.info(f"server = {platform.uname()[1]}")
    log.info(f"working_dir = {os.getcwd()}")
    log.info(f"datetime = {datetime.datetime.now()}")

    args.envs = args.envs or twbench.envs
    args.envs = [  # Expand tasks into their respective environments.
        env
        for task in args.envs
        for env in (twbench.envs_per_task[task] if task in twbench.tasks else [task])
    ]

    benchmark(agent, args.envs, args)


if __name__ == "__main__":
    main()
