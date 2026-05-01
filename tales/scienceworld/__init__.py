import gymnasium as gym

from .scienceworld_env import TASK_NAMES, ScienceWorldEnv

environments = []
train_environments = []

for task_name in TASK_NAMES:
    env_name = f"ScienceWorld{task_name.title().replace('-', '')}"
    environments.append([env_name, "v0"])

    gym.register(
        id=f"tales/{env_name}-v0",
        entry_point="tales.scienceworld:ScienceWorldEnv",
        kwargs={"task_name": task_name, "split": "test"},
    )

    train_env_name = env_name + "_train"
    train_environments.append([train_env_name, "v0"])
    gym.register(
        id=f"tales/{train_env_name}-v0",
        entry_point="tales.scienceworld:ScienceWorldEnv",
        kwargs={"task_name": task_name, "split": "train"},
    )


def download():
    pass
