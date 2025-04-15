import gymnasium as gym

from .twx_env import TASKS, TextWorldExpressEnv

environments = []

for task_name, game_name, game_params in TASKS:
    env_name = f"TWX{task_name}"
    environments.append([env_name, "v0"])

    gym.register(
        id=f"tales/{env_name}-v0",
        entry_point="tales.textworld_express:TextWorldExpressEnv",
        kwargs={"game_name": game_name, "game_params": game_params},
    )


def download():
    pass
