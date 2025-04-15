import gymnasium as gym

from .alfworld_data import TASK_TYPES, prepare_alfworld_data
from .alfworld_env import ALFWorldTask

environments = []

for split in ["seen", "unseen"]:
    for task_type in TASK_TYPES:
        task_name = task_type.replace("_", " ").title().replace(" ", "")
        env_name = f"ALFWorld{task_name}{split.title()}"
        environments.append([env_name, "v0"])

        gym.register(
            id=f"tales/{env_name}-v0",
            entry_point="tales.alfworld:ALFWorldTask",
            kwargs={"task_type": task_type, "split": split},
        )


def download():
    prepare_alfworld_data()
