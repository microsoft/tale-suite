"""Qualified Jericho games and environment construction.

``tales`` is imported lazily so that the pure helpers (and their tests) do not
need the game dependencies installed.
"""

from __future__ import annotations

import fcntl
import os
import sys
import warnings
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator, Sequence


@dataclass(frozen=True)
class GameSpec:
    env_name: str
    registration_key: str
    md5: str
    title: str
    walkthrough_id: str | None = None
    walkthrough_seed: int | None = None

    @property
    def env_id(self) -> str:
        return f"tales/{self.env_name}-v0"

    def to_dict(self) -> dict[str, Any]:
        return {
            "env_name": self.env_name,
            "registration_key": self.registration_key,
            "md5": self.md5,
            "title": self.title,
            "walkthrough_id": self.walkthrough_id,
            "walkthrough_seed": self.walkthrough_seed,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "GameSpec":
        return cls(
            env_name=data["env_name"],
            registration_key=data["registration_key"],
            md5=data.get("md5", ""),
            title=data.get("title", ""),
            walkthrough_id=data.get("walkthrough_id"),
            walkthrough_seed=data.get("walkthrough_seed"),
        )


@contextmanager
def _working_directory(path: Path) -> Iterator[None]:
    path.mkdir(parents=True, exist_ok=True)
    previous = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(previous)


@contextmanager
def _silence_native_output() -> Iterator[None]:
    """Keep native interpreter diagnostics out of supervisor output."""
    sys.stdout.flush()
    sys.stderr.flush()
    saved_stdout = os.dup(1)
    saved_stderr = os.dup(2)
    sink = os.open(os.devnull, os.O_WRONLY)
    try:
        os.dup2(sink, 1)
        os.dup2(sink, 2)
        yield
    finally:
        sys.stdout.flush()
        sys.stderr.flush()
        os.dup2(saved_stdout, 1)
        os.dup2(saved_stderr, 2)
        os.close(saved_stdout)
        os.close(saved_stderr)
        os.close(sink)


class IsolatedGameEnv:
    """Run game I/O in a private directory so save files cannot cross jobs."""

    def __init__(self, env: Any, working_directory: Path | str) -> None:
        self.env = env
        self.working_directory = Path(working_directory).resolve()

    def reset(self, *args: Any, **kwargs: Any):
        with _working_directory(self.working_directory), _silence_native_output():
            return self.env.reset(*args, **kwargs)

    def step(self, *args: Any, **kwargs: Any):
        with _working_directory(self.working_directory), _silence_native_output():
            return self.env.step(*args, **kwargs)

    def close(self) -> None:
        with _working_directory(self.working_directory), _silence_native_output():
            self.env.close()

    def __getattr__(self, name: str) -> Any:
        return getattr(self.env, name)


def list_split_games(split: str = "jericho-new") -> list[GameSpec]:
    """Games in a Jericho split.

    ``jericho-original`` is the historical TALES roster marked ``new: false``
    in the game catalog. ``jericho-new`` remains limited to games with verified
    winning walkthroughs, matching the qualified 959-game expansion.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        from tales.jericho import (
            CATALOG_REGISTRATION_KEYS,
            GAMES_INFOS,
            get_evaluation_games,
            get_game_catalog,
        )

    if split not in {"jericho-new", "jericho-original", "jericho-all"}:
        raise ValueError(f"Unknown split: {split!r}")

    games: list[GameSpec] = []
    if split in {"jericho-original", "jericho-all"}:
        for metadata in get_game_catalog().games:
            if metadata.new:
                continue
            registration_key = metadata.id
            runtime_info = GAMES_INFOS[registration_key]
            games.append(
                GameSpec(
                    env_name=f"JerichoEnv{registration_key.title()}",
                    registration_key=registration_key,
                    md5=runtime_info["md5"].lower(),
                    title=metadata.title,
                )
            )
    if split in {"jericho-new", "jericho-all"}:
        records = get_evaluation_games(
            new=True,
            registration_keys=CATALOG_REGISTRATION_KEYS,
        )
        games.extend(
            GameSpec(
                env_name=record.environment,
                registration_key=record.registration_key,
                md5=record.game.md5,
                title=record.game.title,
                walkthrough_id=record.walkthrough.id,
                walkthrough_seed=record.walkthrough.seed,
            )
            for record in records
        )

    games.sort(key=lambda game: game.env_name)
    return games


def filter_games(games: Sequence[GameSpec], wanted: Iterable[str] | None):
    if not wanted:
        return list(games)
    names = {name for name in wanted}
    selected = [game for game in games if game.env_name in names]
    missing = names - {game.env_name for game in selected}
    if missing:
        raise ValueError(f"Unknown game(s) for this split: {sorted(missing)}")
    return selected


@contextmanager
def game_data_lock() -> Iterator[None]:
    """Serialise ROM downloads across workers.

    ``tales.utils.download`` stages downloads through a *shared* temp path, so
    two workers fetching the same archive corrupt each other. Creating envs
    under this lock keeps the shared game cache consistent.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        from tales.config import TALES_CACHE_HOME

    lock_dir = Path(TALES_CACHE_HOME) / "jericho"
    lock_dir.mkdir(parents=True, exist_ok=True)
    with open(lock_dir / ".tales-download.lock", "w") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def download_games(games: Sequence[GameSpec], *, log=print) -> int:
    """Pre-fetch every ROM once, before any concurrent worker starts."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        from tales.jericho import jericho_data

    downloaded = 0
    with game_data_lock():
        for index, game in enumerate(games, start=1):
            jericho_data.prepare_jericho_data(games=[game.registration_key])
            downloaded += 1
            if index % 50 == 0 or index == len(games):
                log(f"  prepared {index}/{len(games)} games")
    return downloaded


def make_env(
    env_name: str,
    *,
    admissible_commands: bool = False,
    lock: bool = True,
    working_directory: Path | str | None = None,
):
    """Create the gym environment used by the TALES benchmark."""
    import gymnasium as gym

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        import tales  # noqa: F401 - registers the tales/* gym environments.

    def _make():
        from jericho.jericho import UnsupportedGameWarning

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UnsupportedGameWarning)
            if working_directory is None:
                return gym.make(
                    f"tales/{env_name}-v0",
                    disable_env_checker=True,
                    admissible_commands=admissible_commands,
                )
            directory = Path(working_directory).resolve()
            with _working_directory(directory), _silence_native_output():
                env = gym.make(
                    f"tales/{env_name}-v0",
                    disable_env_checker=True,
                    admissible_commands=admissible_commands,
                )
            return IsolatedGameEnv(env, directory)

    if not lock:
        return _make()
    with game_data_lock():
        return _make()
