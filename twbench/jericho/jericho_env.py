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
