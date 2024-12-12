import gymnasium as gym
import numpy as np
import textworld
import textworld.gym
from alfworld.agents.environment.alfred_tw_env import (
    AlfredDemangler,
    AlfredExpert,
    AlfredExpertType,
)
from alfworld.agents.utils.misc import add_task_to_grammar
from alfworld.info import ALFWORLD_DATA
from textworld.envs.wrappers import Filter

from . import alfworld_data
from .alfworld_data import TASK_TYPES

ALFWORLD_HELP = """\
Available commands:
  look:                             look around your current location
  inventory:                        check your current inventory
  go to [receptacle]:               move to a receptacle
  open [receptacle]:                open a receptacle
  close [receptacle]:               close a receptacle
  take [object] from [receptacle]:  take an object from a receptacle
  put [object] in/on [receptacle]:  place an object in or on a receptacle
  examine [something]:              examine a receptacle or an object
  use [object]:                     use an object
  heat [object] with [receptacle]:  heat an object using a receptacle
  clean [object] with [receptacle]: clean an object using a receptacle
  cool [object] with [receptacle]:  cool an object using a receptacle
  slice [object] with [object]:     slice an object using a sharp object
"""


class ALFWorldEnv(gym.Env):

    def __init__(self, gamefile, admissible_commands=False, *args, **kwargs):
        self.infos = textworld.EnvInfos(
            score=True,
            max_score=True,
            won=True,
            lost=True,
            feedback=True,
            moves=True,
            admissible_commands=admissible_commands,
            extras=["walkthrough", "expert_plan"],
        )
        self.gamefile = gamefile
        self.env = None

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed, options=options)

        if self.env is None:
            # expert = AlfredExpert(expert_type=AlfredExpertType.PLANNER)
            # expert = AlfredExpert(expert_type=AlfredExpertType.HANDCODED)
            self.env = textworld.start(
                self.gamefile, self.infos, wrappers=[Filter, AlfredDemangler()]
            )  # , expert])

        obs, info = self.env.reset()
        # obs += "\nNote: Make sure you navigate to the receptacle before interacting with it."
        # Add help message after "-= Welcome to TextWorld, ALFRED! =-"
        # before, after = obs.split("-= Welcome to TextWorld, ALFRED! =-\n")
        # obs = before + "-= Welcome to TextWorld, ALFRED! =-\n" + ALFWORLD_HELP + after

        # from ipdb import set_trace; set_trace()
        info["feedback"] = obs
        info["score"] = 0
        info["max_score"] = 1
        return obs, info

    def step(self, action):
        obs, done, reward, info = self.env.step(action)
        # if action == "help":
        #     obs = ALFWORLD_HELP

        # if obs == "Nothing happens.":
        #     obs = "Invalid command or this command can't be used in this context. Type 'help' for a list of available commands."

        info["feedback"] = obs
        info["score"] = int(done)
        info["max_score"] = 1
        return obs, done, reward, info


class ALFWorldTask(ALFWorldEnv):

    def __init__(self, task_type, split, *args, **kwargs):
        self.gamefiles = sorted(alfworld_data.get_alfworld_game(task_type, split))
        super().__init__(self.gamefiles[0], *args, **kwargs)

    def reset(self, *, seed=None, options=None):
        if seed is not None:
            self.gamefile = self.gamefiles[seed % len(self.gamefiles)]
            if self.env is not None:
                self.env.close()
                self.env = None

        return super().reset(seed=seed, options=options)
