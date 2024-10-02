import gymnasium as gym

from .jericho_env import JerichoEnv
from .jericho_data import GAMES_INFOS

environments = []

for game, infos in GAMES_INFOS.items():
    env_name = f"JerichoEnv{game.title()}"
    environments.append([env_name, "v0"])

    gym.register(
        id=f'twbench/{env_name}-v0',
        entry_point='twbench.jericho:JerichoEnv',
        kwargs={'game': game},
    )
