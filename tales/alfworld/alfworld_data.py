import glob
import os
import zipfile
from os.path import join as pjoin

from tales.config import TALES_CACHE_HOME, TALES_FORCE_DOWNLOAD
from tales.utils import download

TASK_TYPES = [
    "pick_and_place_simple",
    "look_at_obj_in_light",
    "pick_clean_then_place_in_recep",
    "pick_heat_then_place_in_recep",
    "pick_cool_then_place_in_recep",
    "pick_two_obj_and_place",
]

ALFWORLD_DATA_URL = "https://github.com/alfworld/alfworld/releases/download/0.4.2/json_2.1.3_tw-pddl.zip"
TALES_CACHE_ALFWORLD = pjoin(TALES_CACHE_HOME, "alfworld")
TALES_CACHE_ALFWORLD_DATA_ZIP = pjoin(TALES_CACHE_ALFWORLD, "json_2.1.3_tw-pddl.zip")
TALES_CACHE_ALFWORLD_VALID_SEEN = pjoin(
    TALES_CACHE_ALFWORLD, "json_2.1.1", "valid_seen"
)
TALES_CACHE_ALFWORLD_VALID_UNSEEN = pjoin(
    TALES_CACHE_ALFWORLD, "json_2.1.1", "valid_unseen"
)


def prepare_alfworld_data(force=TALES_FORCE_DOWNLOAD):
    os.makedirs(TALES_CACHE_ALFWORLD, exist_ok=True)
    data_exists = os.path.exists(TALES_CACHE_ALFWORLD_VALID_SEEN) and os.path.exists(
        TALES_CACHE_ALFWORLD_VALID_UNSEEN
    )
    if data_exists and not force:
        return

    if not os.path.exists(TALES_CACHE_ALFWORLD_DATA_ZIP) or force:
        download(
            ALFWORLD_DATA_URL,
            dst=TALES_CACHE_ALFWORLD,
            desc="Downloading ALFWorld data",
            force=force,
        )

    # Extract the content of the folder test from the downloaded file
    with zipfile.ZipFile(TALES_CACHE_ALFWORLD_DATA_ZIP, "r") as zip_ref:
        # Only extract the test folder
        for member in zip_ref.namelist():
            if "valid_seen" in member or "valid_unseen" in member:
                zip_ref.extract(member, TALES_CACHE_ALFWORLD)


def get_alfworld_game(task_type, split="seen"):
    prepare_alfworld_data()  # make sure the data is ready

    if split == "seen":
        root = TALES_CACHE_ALFWORLD_VALID_SEEN
    elif split == "unseen":
        root = TALES_CACHE_ALFWORLD_VALID_UNSEEN
    else:
        raise ValueError(f"Unknown split: {split}")

    game_files = sorted(glob.glob(pjoin(root, f"{task_type}*", "**", "*.tw-pddl")))
    return game_files
