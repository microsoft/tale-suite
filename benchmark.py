import os
import glob
import time
from termcolor import colored
import wandb
import datetime
import logging
import argparse
import importlib
import platform

import gymnasium as gym
from tqdm import tqdm
import pandas as pd

import twbench


os.environ["WANDB_MODE"] = "disabled"
log = logging.getLogger("tw-bench")


def evaluate(agent, env_name, args, table):
    env = gym.make(f"twbench/{env_name}-v0", disable_env_checker=True, admissible_commands=args.admissible_commands)
    env.unwrapped.seed(args.game_seed)

    log.debug("Using {}".format(env.__class__.__name__))

    start_time = time.time()
    obs, infos = env.reset()
    agent.reset(obs, infos)

    log.debug(f"Environment reset.\n{obs}\n")

    max_score = infos["max_score"]
    nb_losts = 0
    nb_invalid = 0
    highscore = 0
    score = 0
    done = False

    for step in range(1, args.nb_steps + 1):
        action, response = agent.act(obs, score, done, infos)
        log.debug(colored(f"> {action}", "green"))

        if args.debug:
            breakpoint()

        obs, _, done, infos = env.step(action)
        score = infos["score"]
        moves = infos["moves"]
        feedback = infos["feedback"]

        if args.admissible_commands and infos["admissible_commands"] and action not in infos["admissible_commands"]:
            nb_invalid += 1

        msg = "{:5d}. Time: {:9.2f}\tScore: {:3d}\tMove: {:5d}\tAction: {:20s}"
        msg = msg.format(step, time.time() - start_time, score, moves, action)
        log.info(msg)
        norm_score = 100.0 * score / max_score
        if args.enable_wandb:
            wandb.log({"Step": step, "Score": score, "Max Score": max_score, "Normalized Score": norm_score, "Moves": moves, "Context": agent.context_length()})
            if response:
                table.add_data(step, score, max_score, norm_score, moves, agent.context_length(), obs, action, feedback, response.messages, response.text(), response.token_usage)
            else:
                table.add_data(step, score, max_score, norm_score, moves, agent.context_length(), obs, action, feedback, None, None, None)

        log.debug(obs)

        if done:
            highscore = max(score, highscore)

            if infos["won"]:
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

            log.debug(f"Environment reset.\n{obs}\n")

    env.close()

    # Keep highest score.
    highscore = max(score, highscore)

    return step, nb_invalid, nb_losts, highscore, max_score, time.time() - start_time


def benchmark(agent, games, args):
    game_exclusion_list = []

    mean_score = 0
    total_time = 0.
    total_steps = 0
    total_invalid = 0

    nb_games = 0
    log_file = None
    games = sorted(games)
    max_game_name = max(len(os.path.basename(game)) for game in games)
    with tqdm(total=len(games), leave=False) as pbar:
        for game in games:
            total_steps = 0
            game_name = os.path.basename(game)
            log_file = os.path.join("logs", f"{game_name}_{args.llm}_{args.context}_s{args.seed}_t{args.temperature}_c{int(args.conversation)}_a{int(args.admissible_commands)}.json")
            if os.path.exists(log_file):
                log.info("Skipping game: {}".format(game_name))
                continue  # Skip games that have already been evaluated.
            pbar.set_postfix_str(game_name)

            table = None
            if args.enable_wandb:
                wandb_config = {
                    "game": game_name,
                    "llm": args.llm,
                    "seed": args.seed,
                    "context": args.context,
                    "temperature": args.temperature,
                    "conversation": args.conversation,
                    "admissible_commands": args.admissible_commands
                }
                run = wandb.init(
                    project="text-games-benchmark",
                    config=wandb_config,
                    reinit=True
                )

                # create a wandb table with corresponding columns
                columns = ["Step", "Score", "Max Score", "Normalized Score", "Moves", "Context", "Observation", "Action", "Feedback", "Input", "Output", "Token Usage"]
                table = wandb.Table(columns=columns)

            if game_name in game_exclusion_list:
                pbar.write("{} (skip)".format(game_name))
                log.info("Excluded game: {}".format(game_name))
                pbar.update(1)
                continue  # Skip excluded games.
            try:
                nb_steps, nb_invalid, nb_losts, final_score, max_score, seconds = evaluate(agent, game, args, table)
            except ValueError as e:
                pbar.write("{} (skip)".format(game_name))
                log.error(str(e))
                pbar.update(1)
                if args.debug:
                    raise

                continue  # Skip not supported games.

            nb_games += 1

            norm_score = 100.0 * final_score / max_score
            assert norm_score <= 100.0
            total_time += seconds
            total_steps += nb_steps
            total_invalid += nb_invalid

            msg = "{}\t{:5.0f} seconds\t{:4d} losts\tScore: {:3d}/{:3d} ({:6.2f}%)"
            msg = msg.format(game_name.ljust(max_game_name), seconds, nb_losts, final_score, max_score, norm_score)
            log.info(msg)
            if args.enable_wandb:
                wandb.log({"Total steps": total_steps, "Final score": final_score, "Normalized Score": norm_score})
            pbar.write(msg)
            pbar.update(1)

            mean_score += norm_score

            if args.enable_wandb:
                df = pd.DataFrame(table.data, columns=table.columns)
                df.to_json(log_file, orient="records", lines=True)
                run.log({"log_table": table})
                wandb.finish()

    if nb_games > 0 and total_time > 0:
        log.critical("Mean score (over {} games) = {:8.4f}% of total possible".format(nb_games, mean_score / nb_games))
        log.critical("Total time {:9.2f} seconds".format(total_time))
        log.critical("Total {} invalid actions".format(total_invalid))
        log.critical("Avg. speed: {:8.2f} steps per second".format(total_steps / total_time))
        # if args.enable_wandb:
        #     wandb.log({"Number of games": nb_games, "Mean score": mean_score / nb_games,  "Total time": total_time, "Invalid actions": total_invalid, "Avg. speed": total_steps / total_time})


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

    fh = logging.FileHandler(f'tw-bench_{args.llm}.log', mode='w')
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


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--envs", metavar="env", nargs="+", choices=twbench.env_list,
                        help="Interactive text environments to evaluate the agent(s)."
                            f" Available: {sorted(twbench.env_list)}")  # TODO: support: Jericho, TextWorld, etc...
    parser.add_argument("--agent", default="./agent_random.py:RandomAgent",
                        help="Full qualified class name to evaluate. Default: %(default)s")
    parser.add_argument("--llm", default="azure_openai",
                        help="LLM to be used for evaluation. Default: %(default)s")
    parser.add_argument("--nb-steps", type=int, default=1000,
                        help="Maximum number of steps per game.")
    parser.add_argument("--summary_out_file", default="summary.txt",
                        help="Summary information will be written to this file.")
    parser.add_argument("--log_file", default="tw_benchmark.log",
                        help="Verbose information will be written to this file.")
    parser.add_argument("--seed",  type=int, default=1234, help="Seed for LLM")
    parser.add_argument("--game-seed", type=int, default=20241001, help="Seed for the game. Default: %(default)s.")
    parser.add_argument("--temperature",  type=float, default=0.0, help="Temperature for LLM")
    parser.add_argument("--context",  type=int, default=10, help="Context for LLM")
    parser.add_argument("--enable_wandb", action="store_true", help="Log to wandb")
    parser.add_argument("--conversation", action="store_true", help="Enable conversation mode.")
    parser.add_argument("--admissible_commands", action="store_true", help="Enable admissible commands.")
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable verbose mode.")
    parser.add_argument("-vv", "--very-verbose", action="store_true", help="Display actions taken.")
    parser.add_argument("--debug", action="store_true", help="Debug mode.")
    return parser.parse_args()


def main():
    args = parse_args()
    setup_logging(args)
    args.verbose = args.verbose or args.very_verbose

    if args.enable_wandb:
        os.environ["WANDB_MODE"] = "online"

    # Dynamically load agent class.
    path, klass = args.agent.split(":")
    spec = importlib.util.spec_from_file_location("textworld.benchmark.agents", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    if not hasattr(mod, klass):
        msg = "python file '{}' has no class '{}'".format(path, klass)
        raise AttributeError(msg)

    Agent = getattr(mod, klass)

    # Log some info about the machine.
    log.info('system = {}'.format(platform.system()))
    log.info('server = {}'.format(platform.uname()[1]))
    log.info('working_dir = {}'.format(os.getcwd()))
    log.info('datetime = {}'.format(datetime.datetime.now()))

    agent = Agent(args.llm,
                  seed=args.seed,
                  temperature=args.temperature,
                  conversation=args.conversation,
                  context=args.context,
                  admissible_commands=args.admissible_commands)

    args.envs = args.envs or twbench.env_list
    benchmark(agent, args.envs, args)


if __name__ == "__main__":
    main()
