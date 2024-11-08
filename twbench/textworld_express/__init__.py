import gymnasium as gym

from .twx_env import TASK_NAMES, TextWorldExpressEnv

environments = []

for task_name in TASK_NAMES:
    env_name = f"TWX{task_name.title()}"
    environments.append([env_name, "v0"])

    gym.register(
        id=f"twbench/{env_name}-v0",
        entry_point="twbench.textworld_express:TextWorldExpressEnv",
        kwargs={"task_name": task_name},
    )
