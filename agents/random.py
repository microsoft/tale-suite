import re

import numpy as np

import twbench
from twbench.utils import count_tokens


class RandomAgent(twbench.Agent):
    def __init__(self, **kwargs):
        self.seed = kwargs.get("seed", 1234)
        self.rng = np.random.RandomState(self.seed)

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

    def act(self, obs, reward, done, infos):
        stats = {
            "prompt": None,
            "response": None,
            "nb_tokens": count_tokens(text=obs),
        }

        if "admissible_commands" in infos:
            return self.rng.choice(infos["admissible_commands"]), stats

        action = self.rng.choice(self.actions)
        if action in ["take", "drop", "eat", "attack"]:
            words = re.findall(
                r"\b[a-zA-Z]{4,}\b", obs
            )  # Extract words with 4 or more letters.
            if len(words) > 0:
                action += " " + self.rng.choice(words)

        return str(action), stats
