import importlib
import traceback
import warnings

from termcolor import colored
from tqdm import tqdm

from tales import tasks


def download():
    for task in tqdm(tasks, desc="Downloading data for TALES"):
        try:
            module = importlib.import_module(f".{task}", package="tales")
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
