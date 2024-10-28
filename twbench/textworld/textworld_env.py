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

        self.env.seed(seed)
        obs, info = self.env.reset()
        walkthrough = info["extra.walkthrough"]
        for act in walkthrough:
            _, _, _, info_internal_eval = self.env.step(act)

        if info_internal_eval["score"] != info_internal_eval["max_score"]:
            valid_walkthrough = False
        else:
            valid_walkthrough = True

        _, _ = self.env.reset()
        info["valid_walkthrough"] = valid_walkthrough

        return obs, info

    def step(self, action):
        return self.env.step(action)


class TWCookingEnv(TextWorldEnv):

    def __init__(self, difficulty, *args, **kwargs):
        self.gamefiles = sorted(textworld_data.get_cooking_game(difficulty))
        super().__init__(self.gamefiles[0], *args, **kwargs)

    def reset(self, *, seed=None, options=None):
        if seed is not None:
            self.gamefile = self.gamefiles[seed % len(self.gamefiles)]
            if self.env is not None:
                self.env.close()
                self.env = None

        return super().reset(seed=seed, options=options)
