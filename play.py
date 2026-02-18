"""Lightweight single-game runner for tale-suite agents.

Usage examples:

    # Play a registered environment:
    python play.py api_reasoning --llm api-llama-4-scout --conversation --reasoning-effort medium --env JerichoEnvZork1

    # Play a custom Jericho z-file:
    python play.py api_reasoning --llm api-llama-4-scout --conversation --reasoning-effort medium --game-file /path/to/custom.z8
"""

import argparse
import glob
import importlib
import os
import sys
import time
from functools import partial

import gymnasium as gym
import textworld
from termcolor import colored
from textworld.envs.wrappers import Filter

import tales


def _maybe_load_agent_module():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument(
        "--agent",
        default="agents/*.py",
        help="Load external python file(s). Useful to register custom agent on-the-fly. Default: %(default)s",
    )
    args, _ = parser.parse_known_args()
    if args.agent:
        for agent_file in glob.glob(args.agent):
            agent_dirname = os.path.dirname(agent_file)
            agent_filename, _ = os.path.splitext(os.path.basename(agent_file))
            if f"{agent_dirname}.{agent_filename}" in sys.modules:
                continue

            spec = importlib.util.spec_from_file_location(
                f"{agent_dirname}.{agent_filename}", agent_file
            )
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)


def make_env_from_game_file(game_file, admissible_commands=False):
    """Create a textworld environment directly from a Jericho z-file."""
    infos = textworld.EnvInfos(
        score=True,
        max_score=True,
        won=True,
        lost=True,
        feedback=True,
        moves=True,
        admissible_commands=admissible_commands,
        extras=["walkthrough"],
    )
    return textworld.start(game_file, infos, wrappers=[Filter])


def parse_args():
    general_parser = argparse.ArgumentParser(add_help=False)
    general_parser.add_argument(
        "--agent",
        default="./agents/*",
        help="Load external python file(s). Default: %(default)s",
    )

    parser = argparse.ArgumentParser(
        parents=[general_parser],
        description="Play a single text-adventure game with an agent.",
    )
    subparsers = parser.add_subparsers(
        dest="subcommand", title="Available agents"
    )

    def _add_game_settings(p):
        group = p.add_argument_group("Game settings")
        env_group = group.add_mutually_exclusive_group(required=True)
        env_group.add_argument(
            "--env",
            help="Registered environment name (e.g. JerichoEnvZork1).",
        )
        env_group.add_argument(
            "--game-file",
            help="Path to a Jericho z-file (e.g. zork1.z5).",
        )
        group.add_argument(
            "--game-seed",
            type=int,
            help="Seed for the game. Default: game-specific.",
        )
        group.add_argument(
            "--nb-steps",
            type=int,
            default=100,
            help="Maximum number of steps. Default: %(default)s",
        )
        group.add_argument(
            "--admissible-commands",
            action="store_true",
            help="Enable admissible commands.",
        )
        group.add_argument(
            "-v", "--verbose",
            action="store_true",
            help="Print full observations (not just actions).",
        )

    for agent_name, (desc, _, add_agent_arguments) in tales.agent.AGENTS.items():
        agent_parser = subparsers.add_parser(agent_name, help=desc)
        add_agent_arguments(agent_parser)
        _add_game_settings(agent_parser)

    return parser.parse_args()


def play(agent, env, args):
    env_name = args.env or os.path.basename(args.game_file)

    obs, info = env.reset(seed=args.game_seed) if hasattr(env, "reset") else env.reset()
    agent.reset(obs, info, env_name)

    max_score = info["max_score"]
    score = 0
    done = False

    print(colored(f"Playing: {env_name}", "cyan"))
    print(colored(f"Max score: {max_score}", "cyan"))
    print()

    if args.verbose:
        print(obs)
        print()

    start_time = time.time()

    for step in range(1, args.nb_steps + 1):
        action, stats = agent.act(obs, score, done, info)

        # Print step info.
        print(f"  Step {step:4d}  |  > {action}  |  Score: {info['score']}/{max_score}")

        if "\n" in action.strip():
            obs = "The game only allows one action per step."
        else:
            obs, _, done, info = env.step(action)

        score = info["score"]

        if args.verbose:
            print(obs)
            print()

        if done:
            if info.get("won") and score == max_score:
                break
            # Restart on done.
            last_obs = obs
            obs, info = env.reset() if not hasattr(env, "reset") else env.reset()
            obs = last_obs + "\n\n-= Restarting =-\n" + obs
            agent.reset(obs, info, env_name)

    elapsed = time.time() - start_time

    print()
    print(colored("=" * 50, "cyan"))
    print(colored(f"  Final score: {score}/{max_score} ({score/max_score:.1%})", "green"))
    print(colored(f"  Steps: {step}", "cyan"))
    print(colored(f"  Time:  {elapsed:.1f}s", "cyan"))
    print(colored("=" * 50, "cyan"))


def main():
    _maybe_load_agent_module()
    args = parse_args()

    if args.subcommand is None:
        print("Specify an agent. Available:")
        for name in sorted(tales.agent.AGENTS):
            print(f"  {name}")
        sys.exit(1)

    # Instantiate agent.
    _, Agent, _ = tales.agent.AGENTS[args.subcommand]
    agent = Agent(**vars(args))
    agent.new = partial(Agent, **vars(args))

    # Create environment.
    if args.game_file:
        if not os.path.isfile(args.game_file):
            print(f"Error: game file not found: {args.game_file}")
            sys.exit(1)
        env = make_env_from_game_file(
            args.game_file, admissible_commands=args.admissible_commands
        )
    else:
        env = gym.make(
            f"tales/{args.env}-v0",
            disable_env_checker=True,
            admissible_commands=args.admissible_commands,
        )

    try:
        play(agent, env, args)
    except KeyboardInterrupt:
        print(colored("\nInterrupted.", "red"))
    finally:
        if hasattr(env, "close"):
            env.close()


if __name__ == "__main__":
    main()
