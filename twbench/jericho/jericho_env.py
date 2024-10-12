import gymnasium as gym
import textworld
from textworld.envs.wrappers import Filter

from .jericho_data import get_game


class JerichoEnv(gym.Env):

    def __init__(self, game, admissible_commands=False, *args, **kwargs):
        gamefile = get_game(game)
        self.infos = textworld.EnvInfos(
            score=True,
            max_score=True,
            won=True,
            lost=True,
            feedback=True,
            moves=True,
            admissible_commands=admissible_commands,
        )
        self.env = textworld.start(gamefile, self.infos, wrappers=[Filter])

    def seed(self, seed):
        self.env.seed(seed)

    def reset(self, seed, options):
        self.seed(seed)
        return self.env.reset()

    def step(self, action):
        return self.env.step(action)
    
    def get_walkthrough(self):
        self.env.request_infos.extras.append("walkthrough")

        obs, infos = self.env.reset()
        
        if infos.get("extra.walkthrough") is None:
            msg = "WalkthroughAgent is only supported for games that have a walkthrough."
            raise NameError(msg)

        return infos.get("extra.walkthrough")
