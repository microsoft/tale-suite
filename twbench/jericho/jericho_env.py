import gymnasium as gym

import textworld
from textworld.envs.wrappers import Filter

from .jericho_data import get_game


class JerichoEnv(gym.Env):

    def __init__(self, game, admissible_commands=False, *args, **kwargs):
        gamefile = get_game(game)
        self.infos = textworld.EnvInfos(
            score=True, max_score=True, won=True, lost=True,
            feedback=True, moves=True,
            admissible_commands=admissible_commands,
        )
        self.env = textworld.start(gamefile, self.infos, wrappers=[Filter])

    def seed(self, seed):
        self.env.seed(seed)

    def reset(self):
        return self.env.reset()

    def step(self, action):
        return self.env.step(action)
