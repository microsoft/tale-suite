import gymnasium as gym

from .evaluation import (
    get_evaluation_games,
    get_evaluation_walkthrough,
    get_winning_walkthrough,
)
from .game_catalog import get_game_catalog, get_game_metadata
from .jericho_data import (
    CATALOG_REGISTRATION_KEYS,
    GAMES_INFOS,
    prepare_jericho_data,
)
from .jericho_env import JerichoEnv
from .walkthrough_catalog import get_walkthrough, list_walkthroughs

environments = []

for game, infos in GAMES_INFOS.items():
    env_name = f"JerichoEnv{game.title()}"
    environments.append([env_name, "v0"])

    gym.register(
        id=f"tales/{env_name}-v0",
        entry_point="tales.jericho:JerichoEnv",
        kwargs={"game": game},
    )

benchmark_splits = {
    "jericho-original": [
        record.environment
        for record in get_evaluation_games(
            new=False, registration_keys=CATALOG_REGISTRATION_KEYS
        )
    ],
    "jericho-new": [
        record.environment
        for record in get_evaluation_games(
            new=True, registration_keys=CATALOG_REGISTRATION_KEYS
        )
    ],
}


def download():
    prepare_jericho_data()
