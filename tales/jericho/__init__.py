import gymnasium as gym

from .jericho_data import GAMES_INFOS, prepare_jericho_data
from .jericho_env import JerichoEnv

environments = []

for game, infos in GAMES_INFOS.items():
    env_name = f"JerichoEnv{game.title()}"
    environments.append([env_name, "v0"])

    gym.register(
        id=f"tales/{env_name}-v0",
        entry_point="tales.jericho:JerichoEnv",
        kwargs={"game": game},
    )


def download():
    prepare_jericho_data()
