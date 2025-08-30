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
            
        return self.env.reset()

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
    

# class TWTrainCookingEnv(TextWorldEnv):
#     # Making this a seperate class for now, not sure if this is the best way
#     def __init__(self, difficulties = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10], games_per_difficulty=5, shuffle = False, *args, **kwargs):
#         self.gamefiles = []
#         self.shuffle = shuffle
#         for difficulty in difficulties:
#             gamefiles = sorted(textworld_data.get_cooking_game(difficulty, split="train"))
#             self.gamefiles += gamefiles[:games_per_difficulty]
#         if self.shuffle:
#             np.random.shuffle(self.gamefiles)
#         super().__init__(self.gamefiles[0], *args, **kwargs)

#     def reset(self, *, seed=None, options=None):

#         # Switch everytime during training
#         if self.shuffle:
#             # Shuffle all game files and take the first one
#             np.random.shuffle(self.gamefiles)
#             self.gamefile = self.gamefiles[0]
#             if self.env is not None:
#                 self.env.close()
#                 self.env = None

#         return super().reset(seed=seed, options=options)
