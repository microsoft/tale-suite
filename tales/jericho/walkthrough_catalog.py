from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache
from importlib.resources import files as importlib_files
from typing import Any


@dataclass(frozen=True)
class Walkthrough:
    id: str
    game_id: str
    rom: str
    rom_md5: str
    seed: int | None
    prelude: tuple[str, ...]
    commands: tuple[str, ...]
    outcome: str | None
    replay_outcome: str | None
    verified: bool
    jericho_compatible: bool
    default: bool
    source: str
    notes: str
    transcript_sha256: str | None

    @property
    def actions(self) -> tuple[str, ...]:
        return self.prelude + self.commands

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Walkthrough":
        return cls(
            id=data["id"],
            game_id=data["game_id"],
            rom=data["rom"],
            rom_md5=data["rom_md5"].lower(),
            seed=data.get("seed"),
            prelude=tuple(data.get("prelude", [])),
            commands=tuple(data["commands"]),
            outcome=data.get("outcome"),
            replay_outcome=data.get("replay_outcome"),
            verified=data.get("verified", False),
            jericho_compatible=data.get("jericho_compatible", True),
            default=data.get("default", False),
            source=data["source"],
            notes=data.get("notes", ""),
            transcript_sha256=data.get("transcript_sha256"),
        )

    def metadata(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "game_id": self.game_id,
            "rom": self.rom,
            "rom_md5": self.rom_md5,
            "seed": self.seed,
            "outcome": self.outcome,
            "replay_outcome": self.replay_outcome,
            "verified": self.verified,
            "jericho_compatible": self.jericho_compatible,
            "default": self.default,
            "source": self.source,
            "transcript_sha256": self.transcript_sha256,
        }


class WalkthroughCatalog:
    def __init__(self, walkthroughs: tuple[Walkthrough, ...]):
        self.walkthroughs = walkthroughs
        self._by_id: dict[str, Walkthrough] = {}
        self._by_game: dict[str, list[Walkthrough]] = {}
        self._by_rom: dict[str, list[Walkthrough]] = {}
        self._by_md5: dict[str, list[Walkthrough]] = {}

        for walkthrough in walkthroughs:
            if walkthrough.id in self._by_id:
                raise ValueError(f"Duplicate walkthrough id: {walkthrough.id}")
            self._by_id[walkthrough.id] = walkthrough
            self._by_game.setdefault(walkthrough.game_id, []).append(walkthrough)
            self._by_rom.setdefault(walkthrough.rom, []).append(walkthrough)
            self._by_md5.setdefault(walkthrough.rom_md5, []).append(walkthrough)

        for game_id, routes in self._by_game.items():
            defaults = [route for route in routes if route.default]
            if len(defaults) > 1:
                raise ValueError(
                    f"Multiple default walkthroughs for {game_id}: "
                    + ", ".join(route.id for route in defaults)
                )
            if defaults and (
                not defaults[0].verified or not defaults[0].jericho_compatible
            ):
                raise ValueError(
                    f"Default walkthrough is not Jericho-verified: {defaults[0].id}"
                )

    def get(self, walkthrough_id: str) -> Walkthrough:
        try:
            return self._by_id[walkthrough_id]
        except KeyError as exc:
            raise KeyError(f"Unknown walkthrough: {walkthrough_id}") from exc

    def list(self, game: str) -> tuple[Walkthrough, ...]:
        routes = self._by_game.get(game)
        if routes is None:
            routes = self._by_md5.get(game.lower())
        if routes is None:
            routes = self._by_rom.get(game)
        return tuple(routes or ())

    def default(self, game: str) -> Walkthrough | None:
        return next((route for route in self.list(game) if route.default), None)


@lru_cache(maxsize=1)
def get_catalog() -> WalkthroughCatalog:
    resource = importlib_files("tales.jericho").joinpath("walkthroughs.jsonl")
    walkthroughs = []
    with resource.open(encoding="utf-8") as catalog_file:
        for line_number, line in enumerate(catalog_file, start=1):
            if not line.strip():
                continue
            try:
                data = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid walkthrough catalog JSON on line {line_number}"
                ) from exc
            walkthroughs.append(Walkthrough.from_dict(data))
    return WalkthroughCatalog(tuple(walkthroughs))


def list_walkthroughs(game: str) -> tuple[Walkthrough, ...]:
    return get_catalog().list(game)


def get_walkthrough(
    game: str, walkthrough_id: str | None = None
) -> Walkthrough | None:
    catalog = get_catalog()
    if walkthrough_id is not None:
        walkthrough = catalog.get(walkthrough_id)
        if walkthrough not in catalog.list(game):
            raise ValueError(
                f"Walkthrough {walkthrough_id!r} does not belong to game {game!r}"
            )
        return walkthrough
    return catalog.default(game)
