import gymnasium as gym
import textworld
import textworld.gym
from alfworld.agents.environment.alfred_tw_env import AlfredDemangler
from textworld.envs.wrappers import Filter

from . import alfworld_data


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
            self.env = textworld.start(
                self.gamefile, self.infos, wrappers=[Filter, AlfredDemangler()]
            )

        obs, info = self.env.reset()
        info["feedback"] = obs
        info["score"] = 0
        info["max_score"] = 1
        return obs, info

    def step(self, action):
        obs, done, reward, info = self.env.step(action)
        # if obs == "Nothing happens.":
        #     obs = "Invalid command or this command can't be used in this context. Type 'help' for a list of available commands."

        info["feedback"] = obs
        info["score"] = int(done)
        info["max_score"] = 1
        return obs, done, reward, info


class ALFWorldTask(ALFWorldEnv):

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
