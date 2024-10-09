import argparse
import glob
import os
from os.path import join as pjoin

import pandas as pd


def parse_args():
    # fmt: off
    parser = argparse.ArgumentParser()
    parser.add_argument("--logs", metavar="path", nargs="+", default=["logs"],
                        help="Paths within which to look for .jsonl files.")
    return parser.parse_args()
    # fmt: on


def main():
    args = parse_args()

    results = []
    for logpath in args.logs:
        for logfile in glob.glob(pjoin(logpath, "**", "*.jsonl"), recursive=True):

            path, _ = os.path.splitext(logfile)
            _, agent, env_name, env_params = path.rsplit(os.path.sep, maxsplit=3)
            admissible_command, game_seed = env_params.split("_")
            admissible_command = bool(int(admissible_command[1]))
            agent = agent.split("_", maxsplit=1)[1]

            data = pd.read_json(logfile, lines=True)

            results.append(
                {
                    "agent": agent,
                    "env_name": env_name,
                    # "env_params": env_params,
                    "admissible_command": admissible_command,
                    "game_seed": game_seed,
                    "total_tokens": data["Token Usage"].sum(),
                    "avg_tokens_per_step": data["Token Usage"].mean(),
                    "norm_score": data["Normalized Score"].max(),
                    "nb_steps": data["Step"].max(),
                    # TODO: add more metrics: duration, nb_resets, nb_wins/losts, nb_invalid_actions, in-game moves
                }
            )
    df = pd.DataFrame.from_records(results)

    group = df.groupby(["agent", "admissible_command", "env_name"])
    columns = ["total_tokens", "avg_tokens_per_step", "norm_score", "nb_steps"]
    print(group[columns].mean())
    print()

    group = df.groupby(["agent", "admissible_command"])
    aggregated_results = group.agg(
        {
            "total_tokens": "sum",
            "avg_tokens_per_step": "mean",
            "norm_score": ["mean", "std"],
            "nb_steps": "mean",
        }
    )
    print(aggregated_results)


if __name__ == "__main__":
    main()
