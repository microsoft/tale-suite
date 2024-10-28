import gymnasium as gym
import numpy as np
import textworld_express


class TextWorldExpressEnv(gym.Env):

    def __init__(self, task_name, admissible_commands=False, *args, **kwargs):
        self.gameName = task_name
        self.admissible_commands = admissible_commands
        self.env = textworld_express.TextWorldExpressEnv(
            "/usr/local/lib/python3.10/site-packages/textworld_express/textworld-express.jar"
        )
        self.env.load(gameName=self.gameName, gameParams="")

    def reset(self, *, seed=None, options=None):
        obs, info = self.env.reset(seed=seed, gameFold="test", generateGoldPath=True)
        info["max_score"] = 100
        info["feedback"] = obs
        info["won"] = False
        info["lost"] = False
        info["admissible_commands"] = info["validActions"]
        info["extra.walkthrough"] = self.env.getGoldActionSequence()

        return obs, info

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        info["max_score"] = 100
        info["feedback"] = obs
        info["won"] = info["score"] == 100
        info["lost"] = info["score"] < 0
        info["admissible_commands"] = info["validActions"]
        return obs, reward, done, info

    def close(self):
        self.env.close()
