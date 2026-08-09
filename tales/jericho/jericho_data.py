import json
import os
import shutil
import zipfile
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


def prepare_jericho_data(force=TALES_FORCE_DOWNLOAD, games=None):
    os.makedirs(TALES_CACHE_JERICHO, exist_ok=True)

    selected = GAMES_INFOS.items()
    if games is not None:
        selected = ((name, GAMES_INFOS[name]) for name in games)

    for name, game_info in selected:
        filename = game_info["filename"]

        game_file = pjoin(TALES_CACHE_JERICHO, filename)
        if os.path.isfile(game_file) and not force:
            continue

        archive_url = game_info.get("archive_url")
        if archive_url is not None:
            archive_filename = game_info["archive_filename"]
            archive_file = download(
                archive_url,
                dst=TALES_CACHE_JERICHO,
                force=force,
                filename=archive_filename,
            )
            archive_member = game_info.get("archive_member", filename)
            temp_game_file = f"{game_file}.tmp"
            with zipfile.ZipFile(archive_file) as archive:
                with archive.open(archive_member) as src:
                    with open(temp_game_file, "wb") as dst:
                        shutil.copyfileobj(src, dst)
            os.replace(temp_game_file, game_file)
            continue

        link = game_info.get("download_url", f"{GAMES_URLS}/{filename}")
        download(
            link,
            dst=TALES_CACHE_JERICHO,
            force=force,
            filename=filename,
        )


def get_game(game):
    prepare_jericho_data(games=[game])  # make sure the requested game is ready

    game_info = GAMES_INFOS[game]
    game_file = pjoin(TALES_CACHE_JERICHO, game_info["filename"])
    return game_file
