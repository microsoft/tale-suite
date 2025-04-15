import importlib
import os
import traceback
import warnings
from collections import defaultdict

from termcolor import colored

from tales.agent import Agent
from tales.version import __version__

root_dir = os.path.dirname(os.path.abspath(__file__))
tasks = []
envs = []
envs_per_task = defaultdict(list)

_exclude_path = ["__pycache__", "tests"]

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
        module = importlib.import_module(f".{task}", package="tales")
        environments = getattr(module, "environments", None)
        if environments:
            for env_name, version in environments:
                envs.append(env_name)
                envs_per_task[task].append(env_name)
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
        warnings.warn(colored(f"{traceback.format_exc()}", "red"), UserWarning)
        continue

envs_per_task = dict(envs_per_task)
env2task = {env: task for task, envs in envs_per_task.items() for env in envs}

__all__ = ["Agent", "__version__", "envs", "envs_per_task", "tasks"]
