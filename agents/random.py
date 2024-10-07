import numpy as np

import twbench


class RandomAgent(twbench.Agent):
    def __init__(self, **kwargs):
        self.model = None
        self.seed = kwargs.get("seed", 1234)
        self.context = kwargs.get("context", 100)
        self.rng = np.random.RandomState(self.seed)
        self.conversation = None
        self.window = []
        # fmt:off
        self.actions = [
            "north", "south", "east", "west", "up", "down",
            "look", "inventory",
            "drop", "take", "take all",
            "eat", "attack",
            "wait", "YES",
        ]
        # fmt:on

    def act(self, obs, reward, done, infos):
        if "admissible_commands" in infos:
            self.actions = infos["admissible_commands"]

        action = self.rng.choice(self.actions)
        if action in ["take", "drop", "eat", "attack"]:
            words = obs.split()  # Observed words.
            words = [w for w in words if len(w) > 3]  # Ignore most stop words.
            if len(words) > 0:
                action += " " + self.rng.choice(words)

        return str(action), None
