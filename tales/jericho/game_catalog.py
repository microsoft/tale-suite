from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache
from importlib.resources import files as importlib_files
from typing import Any


@dataclass(frozen=True)
class GameMetadata:
    id: str
    title: str
    filename: str
    aliases: tuple[str, ...]
    md5: str
    new: bool
    source_repository: str
    source_commit: str
    source_collection: str
    download_url: str | None
    archive_url: str | None
    archive_filename: str | None
    archive_member: str | None

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "GameMetadata":
        return cls(
            id=data["id"],
            title=data["title"],
            filename=data["filename"],
            aliases=tuple(data.get("aliases", [])),
            md5=data["md5"].lower(),
            new=data["new"],
            source_repository=data["source_repository"],
            source_commit=data["source_commit"],
            source_collection=data["source_collection"],
            download_url=data.get("download_url"),
            archive_url=data.get("archive_url"),
            archive_filename=data.get("archive_filename"),
            archive_member=data.get("archive_member"),
        )


class GameCatalog:
    def __init__(self, games: tuple[GameMetadata, ...]):
        self.games = games
        self._by_id: dict[str, GameMetadata] = {}
        self._by_md5: dict[str, GameMetadata] = {}
        self._by_filename: dict[str, list[GameMetadata]] = {}
        self._by_alias: dict[str, list[GameMetadata]] = {}

        for game in games:
            if game.id in self._by_id:
                raise ValueError(f"Duplicate game id: {game.id}")
            if game.md5 in self._by_md5:
                raise ValueError(f"Duplicate game MD5: {game.md5}")
            self._by_id[game.id] = game
            self._by_md5[game.md5] = game
            self._by_filename.setdefault(game.filename, []).append(game)
            for alias in game.aliases:
                self._by_alias.setdefault(alias, []).append(game)

    def get(self, game: str) -> GameMetadata:
        metadata = self._by_id.get(game) or self._by_md5.get(game.lower())
        if metadata is not None:
            return metadata

        matches = self._by_filename.get(game) or self._by_alias.get(game)
        if not matches:
            raise KeyError(f"Unknown game: {game}")
        unique_matches = {match.md5: match for match in matches}
        if len(unique_matches) > 1:
            ids = ", ".join(sorted(match.id for match in unique_matches.values()))
            raise ValueError(f"Ambiguous game name {game!r}; use one of: {ids}")
        return next(iter(unique_matches.values()))

    @property
    def new_games(self) -> tuple[GameMetadata, ...]:
        return tuple(game for game in self.games if game.new)


@lru_cache(maxsize=1)
def get_game_catalog() -> GameCatalog:
    resource = importlib_files("tales.jericho").joinpath("game_catalog.jsonl")
    games = []
    with resource.open(encoding="utf-8") as catalog_file:
        for line_number, line in enumerate(catalog_file, start=1):
            if not line.strip():
                continue
            try:
                data = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid game catalog JSON on line {line_number}"
                ) from exc
            games.append(GameMetadata.from_dict(data))
    return GameCatalog(tuple(games))


def get_game_metadata(game: str) -> GameMetadata:
    return get_game_catalog().get(game)


def catalog_registration_key(
    game: GameMetadata, registered: dict[str, dict[str, Any]]
) -> str:
    existing = registered.get(game.id)
    if existing is None or existing.get("md5", "").lower() == game.md5:
        return game.id
    return f"{game.id}_{game.md5[:8]}"


def catalog_game_info(game: GameMetadata) -> dict[str, Any]:
    return {
        "filename": game.filename,
        "md5": game.md5,
        "catalog_id": game.id,
        "title": game.title,
        "new": game.new,
        "download_url": game.download_url,
        "archive_url": game.archive_url,
        "archive_filename": game.archive_filename,
        "archive_member": game.archive_member,
    }
