import json
import os
from os.path import join as pjoin

from twbench.config import TWBENCH_CACHE_HOME, TWBENCH_FORCE_DOWNLOAD
from twbench.utils import download

GAMES_URLS = "https://github.com/BYU-PCCL/z-machine-games/raw/master/jericho-game-suite"
TWBENCH_CACHE_JERICHO = pjoin(TWBENCH_CACHE_HOME, "jericho")


with open(pjoin(os.path.dirname(__file__), "games.json")) as f:
    GAMES_INFOS = json.load(f)


def prepare_jericho_data(force=TWBENCH_FORCE_DOWNLOAD):
    # if os.path.exists(TWBENCH_CACHE_JERICHO) and not force:
    #     if os.listdir(TWBENCH_CACHE_JERICHO):
    #         return

    os.makedirs(TWBENCH_CACHE_JERICHO, exist_ok=True)

    for name, game_info in GAMES_INFOS.items():
        filename = game_info["filename"]

        game_file = pjoin(TWBENCH_CACHE_JERICHO, filename)
        if os.path.isfile(game_file) and not force:
            continue

        link = f"{GAMES_URLS}/{filename}"
        download(link, dst=TWBENCH_CACHE_JERICHO, force=force)


def get_game(game):
    prepare_jericho_data()  # make sure the data is ready

    game_info = GAMES_INFOS[game]
    game_file = pjoin(TWBENCH_CACHE_JERICHO, game_info["filename"])
    return game_file
