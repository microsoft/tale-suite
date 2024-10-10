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


def prepare_twcooking_data(force=TWBENCH_FORCE_DOWNLOAD):
    if os.path.exists(TWBENCH_CACHE_TWCOOKING) and not force:
        return

    os.makedirs(TWBENCH_CACHE_TWCOOKING, exist_ok=True)
    download(
        TW_COOKING_URL,
        dst=TWBENCH_CACHE_TWCOOKING,
        desc="Downloading TWCooking",
        force=force,
    )

    # Extract the content of the folder train_20 from the downloaded file
    zip_file = pjoin(TWBENCH_CACHE_TWCOOKING, "rl.0.2.zip")
    with zipfile.ZipFile(zip_file, "r") as zip_ref:
        # Only extract the train_20 folder
        for member in zip_ref.namelist():
            if "train_20" in member:
                zip_ref.extract(member, TWBENCH_CACHE_TWCOOKING)


def get_cooking_game(difficulty):
    prepare_twcooking_data()  # make sure the data is ready

    cooking_dir = pjoin(
        TWBENCH_CACHE_TWCOOKING, "train_20", f"difficulty_level_{difficulty}"
    )
    game_files = glob.glob(pjoin(cooking_dir, "*.z8"))
    return game_files
