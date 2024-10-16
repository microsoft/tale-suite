import gymnasium as gym

from .scienceworld_env import TASK_NAMES, ScienceWorldEnv

environments = []

for task_name in TASK_NAMES:
    env_name = f"ScienceWorld{task_name.title().replace('-', '')}"
    environments.append([env_name, "v0"])

    gym.register(
        id=f"twbench/{env_name}-v0",
        entry_point="twbench.scienceworld:ScienceWorldEnv",
        kwargs={"task_name": task_name},
    )
