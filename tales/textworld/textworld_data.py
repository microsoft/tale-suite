import glob
import os
import zipfile
from os.path import join as pjoin

from tales.config import TALES_CACHE_HOME, TALES_FORCE_DOWNLOAD
from tales.utils import download

TW_COOKING_URL = (
    "https://github.com/xingdi-eric-yuan/GATA-public/releases/download/data/rl.0.2.zip"
)
TALES_CACHE_TEXTWORLD = pjoin(TALES_CACHE_HOME, "textworld")
TALES_CACHE_TWCOOKING = pjoin(TALES_CACHE_TEXTWORLD, "tw-cooking")
TALES_CACHE_TWCOOKING_TEST = pjoin(TALES_CACHE_TWCOOKING, "test")
TALES_CACHE_TWCOOKING_TRAIN = pjoin(TALES_CACHE_TWCOOKING, "train")


def prepare_twcooking_data(force=TALES_FORCE_DOWNLOAD):
    os.makedirs(TALES_CACHE_TWCOOKING, exist_ok=True)
    if os.path.exists(TALES_CACHE_TWCOOKING_TEST) and not force:
        return
    if os.path.exists(TALES_CACHE_TWCOOKING_TRAIN) and not force:
        return

    zip_file = pjoin(TALES_CACHE_TWCOOKING, "rl.0.2.zip")
    if not os.path.exists(zip_file) or force:
        download(
            TW_COOKING_URL,
            dst=TALES_CACHE_TWCOOKING,
            desc="Downloading TWCooking",
            force=force,
        )

    # Extract the content of the folder test from the downloaded file
    with zipfile.ZipFile(zip_file, "r") as zip_ref:
        # Only extract the test folder
        for member in zip_ref.namelist():
            if "train" in member or "test" in member:
                zip_ref.extract(member, TALES_CACHE_TWCOOKING)


def get_cooking_game(difficulty, split="test"):
    prepare_twcooking_data()  # make sure the data is ready
    if split == "train":
        cooking_dir = pjoin(TALES_CACHE_TWCOOKING_TRAIN, f"difficulty_level_{difficulty}")
    elif split == "test":
        cooking_dir = pjoin(TALES_CACHE_TWCOOKING_TEST, f"difficulty_level_{difficulty}")
        
    game_files = glob.glob(pjoin(cooking_dir, "*.z8"))
    return game_files
