# This is literally just a wrapper to get the train and test-time splits. Is 99.99% just building on Marc's existing code.
import glob
from os.path import join as pjoin
from tales.textworld import textworld_data, textworld_env
from tales.textworld_express import twx_data, twx_env
from tales.alfworld import alfworld_data, alfworld_env


def get_textworld_env_splits(difficulties = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10], games_per_difficulty=1):
    # Returns a list of envs for training and test splits for Textworld-Cookingworld:
    # For training, we let the user specify difficulties and how many games per difficulty to include.
    # For testing, we use all difficulties from 1 to 10, and use one game each, similar to the evaluation in the original paper.
    textworld_data.prepare_twcooking_data()  # make sure the data is ready

    # Training split:
    # Get the game files:
    train_games_files = []
    for diff in difficulties:
        all_games = sorted(textworld_data.get_cooking_game(diff, split="train"))
        train_games_files.extend(all_games[:games_per_difficulty])

    # Testing split:
    test_games_files = []
    for i in range(1, 11):
        # Just get one game per difficulty for testing.
        # This is similar to the evaluation in the original paper.
        all_games = sorted(textworld_data.get_cooking_game(i, split="test"))
        test_games_files.append(all_games[0])

    return train_games_files, test_games_files

def get_alfworld_env_splits(games_per_task = 2):
    # For alfworld, we just generate the test split first and then condition the train split to not have the same files as the text split.
    alfworld_data.prepare_alfworld_data()  # make sure the data is ready
    test_games_files = []
    for task in alfworld_data.TASK_TYPES:
        game_files_seen = sorted(glob.glob(pjoin(alfworld_data.TALES_CACHE_ALFWORLD_VALID_SEEN, f"{task}*", "**", "*.tw-pddl")))
        game_files_unseen = sorted(glob.glob(pjoin(alfworld_data.TALES_CACHE_ALFWORLD_VALID_UNSEEN, f"{task}*", "**", "*.tw-pddl")))
        # The test split always only takes the first game file in the split.
        test_games_files.extend(game_files_seen[[0]])
        test_games_files.extend(game_files_unseen[[0]])

    # Assert we have the right number of files.
    assert len(test_games_files) == 2 * len(alfworld_data.TASK_TYPES)

    # Now, get the training split.
    # We want to make sure that the training split does not have any files that are in the test split.
    train_games_files = []
    for task in alfworld_data.TASK_TYPES:
        game_files_seen = sorted(glob.glob(pjoin(alfworld_data.TALES_CACHE_ALFWORLD_VALID_SEEN, f"{task}*", "**", "*.tw-pddl")))
        game_files_unseen = sorted(glob.glob(pjoin(alfworld_data.TALES_CACHE_ALFWORLD_VALID_UNSEEN, f"{task}*", "**", "*.tw-pddl")))
        # Remove any files that are in the test split.
        filtered_game_files_seen = [f for f in game_files_seen if not any(s in f for s in test_games_files)]
        filtered_game_files_unseen = [f for f in game_files_unseen if not any(s in f for s in test_games_files)]

        # Now get the requested number of games per task type
        train_games_files.extend(filtered_game_files_seen[:games_per_task])
        train_games_files.extend(filtered_game_files_unseen[:games_per_task])

    return train_games_files, test_games_files
    
class GeneralTALESEnv:
    # A general env wrapper such that the train/test files gotten from the above functions can easily just be plugged into an env and ran.
    # This returns a 'fake' batch env that will always deterministically cycle through the provided env file/seeds unless explicitly told to randomize (for training)
    # TODO: implement for Scienceworld and Jericho
    def __init__(self, env_name, split, *args, **kwargs):
        self.env_name = env_name
        self.split = split
        self.env_idx = 0
        self.kwargs = kwargs
        self.args = args
        self.game_files = None
        if env_name == "textworld":
            self.train_envs, self.test_envs = get_textworld_env_splits(**kwargs)
            if split == "train":
                self.game_files = self.train_envs
            else:
                self.game_files = self.test_envs
            self.env = textworld_env.TextWorldEnv(self.game_files[self.env_idx], 
                                                  *args, **kwargs)
        elif env_name == "twx":
            # Train/test in twx are just seed based.
            self.game_files = twx_data.TASKS
            self.env = twx_env.TextWorldExpressEnv(game_name = self.game_files[self.env_idx][1], 
                                                   game_params = self.game_files[self.env_idx][2], 
                                                   admissible_commands=False, 
                                                   split=split,
                                                   *args, **kwargs)
        elif env_name == "alfworld":
            self.train_envs, self.test_envs = get_alfworld_env_splits(**kwargs)
            if split == "train":
                self.game_files = self.train_envs
            else:
                self.game_files = self.test_envs
            self.env = alfworld_env.ALFWorldEnv(self.game_files[self.env_idx], 
                                                *args, **kwargs)
        else:
            raise ValueError(f"Unknown environment name: {env_name}, please choose from textworld, twx, or alfworld.")
        
    # Not sure if this is right, need to double check w/ Marc
    def reset(self, *, seed=None, options=None):
        return self.env.reset(seed=seed, options=options)

    def get_next_task(self, seed = None, options=None):
        # Move to the next env in the list.
        self.env_idx = (self.env_idx + 1) % len(self.game_files)
        if self.env is not None:
            self.env.close()
        if self.env_name == "textworld":
            self.env = textworld_env.TextWorldEnv(self.game_files[self.env_idx], *self.args, **self.kwargs)
        elif self.env_name == "twx":
            self.env = twx_env.TextWorldExpressEnv(game_name = self.game_files[self.env_idx][1], 
                                                   game_params = self.game_files[self.env_idx][2], 
                                                   *self.args, **self.kwargs)
        elif self.env_name == "alfworld":
            self.env = alfworld_env.ALFWorldEnv(self.game_files[self.env_idx], *self.args, **self.kwargs)
        else:
            raise ValueError(f"next_task not implemented for env {self.env_name}, only for textworld and alfworld.")
        return self.reset(seed = seed, options = options)

    def step(self, action):
        return self.env.step(action)


