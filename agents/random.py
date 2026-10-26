import argparse
import re

import numpy as np

import tales
from tales.agent import register
from tales.token import get_token_counter


class RandomAgent(tales.Agent):
    def __init__(self, **kwargs):
        self.seed = kwargs.get("seed", 1234)
        self.rng = np.random.RandomState(self.seed)
        self.token_counter = get_token_counter()

        # fmt:off
        self.actions = [
            "north", "south", "east", "west", "up", "down",
            "look", "inventory",
            "drop", "take", "take all",
            "eat", "attack",
            "wait", "YES",
        ]
        # fmt:on

    @property
    def uid(self):
        return f"RandomAgent_s{self.seed}"

    @property
    def params(self):
        return {
            "agent_type": "random",
            "seed": self.seed,
        }

    def act(self, obs, reward, done, info):
        stats = {
            "prompt": None,
            "response": None,
            "nb_tokens": self.token_counter(text=obs),
        }

        if "admissible_commands" in info:
            return self.rng.choice(info["admissible_commands"]), stats

        action = self.rng.choice(self.actions)
        if action in ["take", "drop", "eat", "attack"]:
            words = re.findall(
                r"\b[a-zA-Z]{4,}\b", obs
            )  # Extract words with 4 or more letters.
            if len(words) > 0:
                action += " " + self.rng.choice(words)

        return str(action), stats


def build_argparser(parser=None):
    parser = parser or argparse.ArgumentParser()
    group = parser.add_argument_group("RandomAgent settings")
    group.add_argument(
        "--seed",
        type=int,
        default=20241001,
        help="Random generator seed to select actions. Default: %(default)s",
    )
    return parser


register(
    name="random",
    desc=(
        "This agent will pick an action at random among a predefined set of actions or,"
        " if available, the admissible commands."
    ),
    klass=RandomAgent,
    add_arguments=build_argparser,
)
