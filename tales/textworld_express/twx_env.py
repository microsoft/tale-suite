import gymnasium as gym
import numpy as np
import textworld_express as twx

from . import twx_data

TASKS = twx_data.TASKS


class TextWorldExpressEnv(gym.Env):

    def __init__(
        self, game_name, game_params, admissible_commands=False, split="test", *args, **kwargs
    ):
        self.game_name = game_name
        self.game_params = game_params
        self.admissible_commands = admissible_commands
        self.env = twx.TextWorldExpressEnv(envStepLimit=np.inf)
        self.seeds = twx_data.get_seeds(split=split, env=self.env)
        self.seed = self.seeds[0]

    def reset(self, *, seed=None, options=None):
        if seed is not None:
            self.seed = self.seeds[seed % len(self.seeds)]

        obs, info = self.env.reset(
            seed=self.seed,
            gameFold="test",
            gameName=self.game_name,
            gameParams=self.game_params,
            generateGoldPath=True,
        )

        # Add task description to the first observation.
        obs = info["taskDescription"] + "\n\n" + obs

        info["max_score"] = 100
        info["feedback"] = obs
        info["won"] = False
        info["lost"] = False
        info["moves"] = 0
        info["score"] = int(info["score"] * 100)
        info["admissible_commands"] = info["validActions"]
        info["extra.walkthrough"] = self.env.getGoldActionSequence()
        return obs, info

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        info["max_score"] = 100
        info["feedback"] = obs
        info["won"] = info["tasksuccess"]
        info["lost"] = info["taskfailure"]
        info["moves"] = info["numMoves"]
        info["score"] = int(info["score"] * 100)
        info["admissible_commands"] = info["validActions"]
        return obs, reward, done, info

    def close(self):
        self.env.close()
