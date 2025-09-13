import gymnasium as gym
import numpy as np
import textworld
from textworld.envs.wrappers import Filter

from . import textworld_data


class TextWorldEnv(gym.Env):
    def __init__(self, gamefile, admissible_commands=False, *args, **kwargs):
        self.infos = textworld.EnvInfos(
            score=True,
            max_score=True,
            won=True,
            lost=True,
            feedback=True,
            moves=True,
            admissible_commands=admissible_commands,
            extras=["walkthrough"],
        )
        self.gamefile = gamefile
        self.env = None

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed, options=options)

        if self.env is None:
            self.env = textworld.start(self.gamefile, self.infos, wrappers=[Filter])

        return self.env.reset()

    def step(self, action):
        return self.env.step(action)


class TWCookingEnv(TextWorldEnv):
    def __init__(self, all_gamefiles, start_gamefile, *args, **kwargs):
        self.gamefiles = all_gamefiles
        super().__init__(start_gamefile, *args, **kwargs)

    def reset(self, *, seed=None, options=None):
        if seed is not None:
            self.gamefile = self.gamefiles[seed % len(self.gamefiles)]
            if self.env is not None:
                self.env.close()
                self.env = None

        return super().reset(seed=seed, options=options)
