from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

from .game_catalog import GameMetadata, get_game_catalog
from .walkthrough_catalog import Walkthrough, get_catalog


@dataclass(frozen=True)
class EvaluationGame:
    game: GameMetadata
    walkthrough: Walkthrough
    registration_key: str

    @property
    def environment(self) -> str:
        return f"JerichoEnv{self.registration_key.title()}"


def _qualifies(route: Walkthrough) -> bool:
    return (
        route.verified
        and route.jericho_compatible
        and route.replay_outcome == "won"
        and route.seed is not None
        and route.seed > 0
    )


def _preferred_route(routes: list[Walkthrough]) -> Walkthrough:
    return min(
        routes,
        key=lambda route: (
            not route.default,
            len(route.actions),
            route.id,
        ),
    )


@lru_cache(maxsize=1)
def _qualified_by_md5() -> dict[str, Walkthrough]:
    routes_by_md5: dict[str, list[Walkthrough]] = {}
    for route in get_catalog().walkthroughs:
        if _qualifies(route):
            routes_by_md5.setdefault(route.rom_md5, []).append(route)
    return {md5: _preferred_route(routes) for md5, routes in routes_by_md5.items()}


@lru_cache(maxsize=1)
def _winning_by_md5() -> dict[str, Walkthrough]:
    routes_by_md5: dict[str, list[Walkthrough]] = {}
    for route in get_catalog().walkthroughs:
        if (
            route.verified
            and route.jericho_compatible
            and route.replay_outcome == "won"
        ):
            routes_by_md5.setdefault(route.rom_md5, []).append(route)
    return {md5: _preferred_route(routes) for md5, routes in routes_by_md5.items()}


def get_evaluation_games(
    *,
    new: bool | None = None,
    registration_keys: dict[str, str] | None = None,
) -> tuple[EvaluationGame, ...]:
    qualified = _qualified_by_md5()
    records = []
    for game in get_game_catalog().games:
        if new is not None and game.new is not new:
            continue
        walkthrough = qualified.get(game.md5)
        if walkthrough is None:
            continue
        registration_key = (
            registration_keys.get(game.md5, game.id)
            if registration_keys is not None
            else game.id
        )
        records.append(EvaluationGame(game, walkthrough, registration_key))
    return tuple(records)


def get_evaluation_walkthrough(game: str) -> Walkthrough | None:
    walkthrough = _qualified_by_md5().get(game.lower())
    if walkthrough is not None:
        return walkthrough
    metadata = get_game_catalog().get(game)
    return _qualified_by_md5().get(metadata.md5)


def get_winning_walkthrough(game: str) -> Walkthrough | None:
    walkthrough = _winning_by_md5().get(game.lower())
    if walkthrough is not None:
        return walkthrough
    try:
        metadata = get_game_catalog().get(game)
    except KeyError:
        return None
    return _winning_by_md5().get(metadata.md5)
