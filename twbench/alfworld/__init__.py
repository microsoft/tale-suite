import gymnasium as gym

from .alfworld_env import TASK_TYPES, ALFWorldTask

environments = []

for split in ["seen", "unseen"]:
    for task_type in TASK_TYPES:
        task_name = task_type.replace("_", " ").title().replace(" ", "")
        env_name = f"ALFWorld{task_name}{split.title()}"
        environments.append([env_name, "v0"])

        gym.register(
            id=f"twbench/{env_name}-v0",
            entry_point="twbench.alfworld:ALFWorldTask",
            kwargs={"task_type": task_type, "split": split},
        )
