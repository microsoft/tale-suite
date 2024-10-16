import gymnasium as gym
import textworld
from textworld.envs.wrappers import Filter

from . import jericho_data


class JerichoEnv(gym.Env):

    def __init__(self, game, admissible_commands=False, *args, **kwargs):
        gamefile = jericho_data.get_game(game)
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
        self.env = textworld.start(gamefile, self.infos, wrappers=[Filter])

    def reset(self, *, seed=None, options=None):
        self.env.seed(seed)
        return self.env.reset()

    def step(self, action):
        return self.env.step(action)
