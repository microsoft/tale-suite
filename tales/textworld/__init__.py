import gymnasium as gym

from .textworld_data import prepare_twcooking_data
from .textworld_env import TextWorldEnv, TWCookingEnv

environments = []
train_environments = []

# TWCookingEnv
for difficulty in range(1, 10 + 1):
    gamefiles = sorted(textworld_data.get_cooking_game(difficulty))
    train_gamefiles = gamefiles[1:]
    test_gamefiles = [gamefiles[0]]
    env_name = f"TWCookingLevel{difficulty}"
    environments.append([env_name, "v0"])

    gym.register(
        id=f"tales/{env_name}-v0",
        entry_point="tales.textworld:TWCookingEnv",
        kwargs={"all_gamefiles": test_gamefiles, "start_gamefile": test_gamefiles[0]},
    )

    train_env_name = env_name + "_train"
    train_environments.append([train_env_name, "v0"])
    gym.register(
        id=f"tales/{train_env_name}-v0",
        entry_point="tales.textworld:TWCookingEnv",
        kwargs={"all_gamefiles": train_gamefiles, "start_gamefile": train_gamefiles[0]},
    )


def download():
    prepare_twcooking_data()
