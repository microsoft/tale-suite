import importlib
import traceback
import warnings

from termcolor import colored
from tqdm import tqdm

from twbench import tasks


def download():
    for task in tqdm(tasks, desc="Downloading data for TW-Bench"):
        try:
            module = importlib.import_module(f".{task}", package="twbench")
            module.download()
        except Exception as e:
            warnings.warn(
                "Failed to download data for `{task}`.",
                UserWarning,
            )
            warnings.warn(colored(f"{e}", "red"), UserWarning)
            warnings.warn(colored(f"{traceback.format_exc()}", "red"), UserWarning)
            continue


if __name__ == "__main__":
    download()
