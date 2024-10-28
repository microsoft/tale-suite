import gymnasium as gym
from textworld_express.constants import GAME_NAMES

from .tw_express_env import TextWorldExpressEnv

environments = []

for game_name in GAME_NAMES:
    env_name = f"TextWorldExpress{game_name.title().replace('-', '')}"
    environments.append([env_name, "v0"])

    gym.register(
        id=f"twbench/{env_name}-v0",
        entry_point="twbench.tw_express:TextWorldExpressEnv",
        kwargs={"task_name": game_name},
    )
