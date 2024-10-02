import gymnasium as gym

import numpy as np
import textworld
from textworld.envs.wrappers import Filter

from . import textworld_data


class TextWorldEnv(gym.Env):

    def __init__(self, gamefile, admissible_commands=False, *args, **kwargs):
        self.infos = textworld.EnvInfos(
            score=True, max_score=True, won=True, lost=True,
            feedback=True, moves=True,
            admissible_commands=admissible_commands,
        )
        self.gamefile = gamefile
        self.env = None

    def seed(self, seed):
        self.env.seed(seed)

    def reset(self):
        if self.env is None:
            self.env = textworld.start(self.gamefile, self.infos, wrappers=[Filter])

        return self.env.reset()

    def step(self, action):
        return self.env.step(action)


class TWCookingEnv(TextWorldEnv):

    def __init__(self, difficulty, *args, **kwargs):
        self.gamefiles = sorted(textworld_data.get_cooking_game(difficulty))
        super().__init__(self.gamefiles[0], *args, **kwargs)

    def seed(self, seed):
        if self.env is not None:
            self.env.close()
            self.env = None

        self.rng = np.random.RandomState(seed)
        idx = self.rng.choice(len(self.gamefiles))
        self.gamefile = self.gamefiles[idx]
