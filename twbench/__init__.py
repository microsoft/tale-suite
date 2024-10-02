import importlib
import os
import traceback
import warnings

import yaml
from termcolor import colored

from twbench.agent import Agent
from twbench.version import __version__

root_dir = os.path.dirname(os.path.abspath(__file__))
tasks = []
env_list = []


_exclude_path = ["__pycache__", "utils", "tests"]
_module_dir = os.path.dirname(__file__)


for dirname in os.listdir(root_dir):
    if not os.path.isdir(os.path.join(root_dir, dirname)):
        continue

    if dirname in _exclude_path:
        continue

    if "skip" in os.listdir(os.path.join(root_dir, dirname)):
        continue

    if "__init__.py" in os.listdir(os.path.join(root_dir, dirname)):
        tasks.append(dirname)


for task in tasks:
    try:
        # Load environments
        module = importlib.import_module(f".{task}", package="twbench")
        environments = getattr(module, "environments", None)
        if environments:
            for env_name, version in environments:
                # env_list.append('{}-{}'.format(env_name, version))
                env_list.append(env_name)
        else:
            warnings.warn(
                "Failed to load `{}.environments`. Skipping the task.".format(task),
                UserWarning,
            )
            continue

    except Exception as e:
        warnings.warn(
            "Failed to import `{}`. Skipping the task.".format(task), UserWarning
        )
        warnings.warn(colored(f"{e}", "red"), UserWarning)
        # Add stacktrace
        warnings.warn(colored(f"{traceback.format_exc()}", "red"), UserWarning)
        continue


# from twbench.eval import *
