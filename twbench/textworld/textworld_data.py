import glob
import os
import zipfile
from os.path import join as pjoin

from twbench.config import TWBENCH_CACHE_HOME, TWBENCH_FORCE_DOWNLOAD
from twbench.utils import download

TW_COOKING_URL = (
    "https://github.com/xingdi-eric-yuan/GATA-public/releases/download/data/rl.0.2.zip"
)
TWBENCH_CACHE_TEXTWORLD = pjoin(TWBENCH_CACHE_HOME, "textworld")
TWBENCH_CACHE_TWCOOKING = pjoin(TWBENCH_CACHE_TEXTWORLD, "tw-cooking")
TWBENCH_CACHE_TWCOOKING_TEST = pjoin(TWBENCH_CACHE_TWCOOKING, "test")


def prepare_twcooking_data(force=TWBENCH_FORCE_DOWNLOAD):
    os.makedirs(TWBENCH_CACHE_TWCOOKING, exist_ok=True)
    if os.path.exists(TWBENCH_CACHE_TWCOOKING_TEST) and not force:
        return

    zip_file = pjoin(TWBENCH_CACHE_TWCOOKING, "rl.0.2.zip")
    if not os.path.exists(zip_file) or force:
        download(
            TW_COOKING_URL,
            dst=TWBENCH_CACHE_TWCOOKING,
            desc="Downloading TWCooking",
            force=force,
        )

    # Extract the content of the folder test from the downloaded file
    with zipfile.ZipFile(zip_file, "r") as zip_ref:
        # Only extract the test folder
        for member in zip_ref.namelist():
            if "test" in member:
                zip_ref.extract(member, TWBENCH_CACHE_TWCOOKING)


def get_cooking_game(difficulty):
    prepare_twcooking_data()  # make sure the data is ready

    cooking_dir = pjoin(TWBENCH_CACHE_TWCOOKING_TEST, f"difficulty_level_{difficulty}")
    game_files = glob.glob(pjoin(cooking_dir, "*.z8"))
    return game_files
