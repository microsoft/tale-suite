import json
import os
from importlib.resources import files as importlib_files
from os.path import join as pjoin

from tales.config import TALES_CACHE_HOME, TALES_FORCE_DOWNLOAD
from tales.utils import download

GAMES_URLS = "https://github.com/BYU-PCCL/z-machine-games/raw/master/jericho-game-suite"
TALES_CACHE_JERICHO = pjoin(TALES_CACHE_HOME, "jericho")


with open(importlib_files("tales") / "jericho" / "games.json") as f:
    GAMES_INFOS = json.load(f)

# Remove known games that are not working.
GAMES_INFOS.pop("hollywood", None)
GAMES_INFOS.pop("theatre", None)


def prepare_jericho_data(force=TALES_FORCE_DOWNLOAD):
    os.makedirs(TALES_CACHE_JERICHO, exist_ok=True)

    for name, game_info in GAMES_INFOS.items():
        filename = game_info["filename"]

        game_file = pjoin(TALES_CACHE_JERICHO, filename)
        if os.path.isfile(game_file) and not force:
            continue

        link = f"{GAMES_URLS}/{filename}"
        download(link, dst=TALES_CACHE_JERICHO, force=force)


def get_game(game):
    prepare_jericho_data()  # make sure the data is ready

    game_info = GAMES_INFOS[game]
    game_file = pjoin(TALES_CACHE_JERICHO, game_info["filename"])
    return game_file
