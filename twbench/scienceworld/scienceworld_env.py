import gymnasium as gym
import numpy as np
import scienceworld

from . import scienceworld_data

TASK_NAMES = scienceworld_data.get_task_names()


class ScienceWorldEnv(gym.Env):

    def __init__(self, task_name, admissible_commands=False, *args, **kwargs):
        self.task_name = task_name
        self.admissible_commands = admissible_commands
        self.env = scienceworld.ScienceWorldEnv(self.task_name, envStepLimit=np.inf)
        self.variations = scienceworld_data.get_variations(
            self.task_name, split="test", env=self.env
        )
        self.variation = self.variations[0]

    def reset(self, *, seed=None, options=None):
        if seed is not None:
            self.variation = self.variations[seed % len(self.variations)]

        self.env.load(
            self.task_name, self.variation, simplificationStr="", generateGoldPath=True
        )
        obs, info = self.env.reset()

        # Add task description to the first observation.
        obs = info["taskDesc"] + "\n\n" + obs

        info["max_score"] = 100
        info["feedback"] = obs
        info["won"] = False
        info["lost"] = False
        info["admissible_commands"] = info["valid"]
        info["extra.walkthrough"] = self.env.get_gold_action_sequence()
        return obs, info

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        info["max_score"] = 100
        info["feedback"] = obs
        info["won"] = info["score"] == 100
        info["lost"] = info["score"] < 0
        info["admissible_commands"] = info["valid"]
        return obs, reward, done, info

    def close(self):
        self.env.close()
