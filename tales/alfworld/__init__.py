import gymnasium as gym

from .alfworld_data import TASK_TYPES, prepare_alfworld_data
from .alfworld_env import ALFWorldTask

environments = []
train_environments = []

for split in ["seen", "unseen"]:
    for task_type in TASK_TYPES:
        gamefiles = sorted(alfworld_data.get_alfworld_game(task_type, split))
        train_gamefiles = gamefiles[1:]
        test_gamefiles = [gamefiles[0]]

        task_name = task_type.replace("_", " ").title().replace(" ", "")
        env_name = f"ALFWorld{task_name}{split.title()}"
        environments.append([env_name, "v0"])

        gym.register(
            id=f"tales/{env_name}-v0",
            entry_point="tales.alfworld:ALFWorldTask",
            kwargs={
                "all_gamefiles": test_gamefiles,
                "start_gamefile": test_gamefiles[0],
            },
        )

        train_env_name = env_name + "_train"
        train_environments.append([train_env_name, "v0"])
        gym.register(
            id=f"tales/{train_env_name}-v0",
            entry_point="tales.alfworld:ALFWorldTask",
            kwargs={
                "all_gamefiles": train_gamefiles,
                "start_gamefile": train_gamefiles[0],
            },
        )


def download():
    prepare_alfworld_data()
