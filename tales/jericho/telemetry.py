from __future__ import annotations

import re

from .walkthrough_catalog import Walkthrough, list_walkthroughs


def _normalize_action(action: str) -> str:
    return " ".join(action.casefold().split())


def _catalog_routes(game: str) -> tuple[Walkthrough, ...]:
    return tuple(
        route
        for route in list_walkthroughs(game)
        if route.verified
        and route.jericho_compatible
        and route.replay_outcome in {"won", "ended", "incomplete"}
    )


class CatalogTelemetry:
    def __init__(self, game: str, supported: bool):
        self.supported = supported
        self.routes = _catalog_routes(game)
        self.reset()

    def reset(self) -> None:
        self.progress: dict[str, int | None] = {
            route.id: 0 for route in self.routes
        }
        self.terminal_route_id = None
        self.terminal_route_outcome = None
        self.winning_route_id = None
        self.moves = 0
        self.binary_score = False
        for route in self.routes:
            if route.actions:
                continue
            self.terminal_route_id = route.id
            self.terminal_route_outcome = route.replay_outcome
            if route.replay_outcome == "won":
                self.winning_route_id = route.id
            break

    def step(self, action: str) -> None:
        self.moves += 1
        normalized = _normalize_action(action)
        for route in self.routes:
            index = self.progress[route.id]
            if index is None or index >= len(route.actions):
                continue
            if normalized == _normalize_action(route.actions[index]):
                index += 1
                self.progress[route.id] = index
                if index == len(route.actions):
                    self.terminal_route_id = route.id
                    self.terminal_route_outcome = route.replay_outcome
                    if route.replay_outcome == "won":
                        self.winning_route_id = route.id
            else:
                self.progress[route.id] = None

    @property
    def following_catalog_route(self) -> bool:
        return any(
            index is not None and 0 < index < len(route.actions)
            for route in self.routes
            if (index := self.progress[route.id]) is not None
        )

    def adapt(self, observation: str, done: bool, info: dict) -> tuple[bool, dict]:
        reported_score = _reported_score(observation)
        won_from_text = _looks_like_win(observation)
        won_from_score = _has_full_score(info, reported_score)
        route_outcome = self.terminal_route_outcome
        if route_outcome is not None:
            won = route_outcome == "won"
            lost = route_outcome == "ended" and (
                bool(info.get("lost")) or _looks_like_loss(observation)
            )
            ended = route_outcome == "ended" and not lost
        elif self.following_catalog_route:
            won = bool(info.get("won"))
            lost = not won and bool(info.get("lost"))
            ended = done and not won and not lost
        else:
            won = bool(info.get("won")) or won_from_text or won_from_score
            lost = not won and (
                bool(info.get("lost")) or _looks_like_loss(observation)
            )
            ended = not won and not lost and _looks_like_end(observation)
        adapted = dict(info)
        adapted["won"] = won
        adapted["lost"] = lost

        if self.supported:
            self.binary_score = (
                self.binary_score
                or info.get("max_score") == 0
                or (reported_score is not None and reported_score[1] == 0)
            )
            binary_score = self.binary_score
            if binary_score:
                adapted["score"] = int(won)
                adapted["max_score"] = 1
            elif reported_score is not None:
                adapted["score"], adapted["max_score"] = reported_score
            if not (won or lost or ended):
                return (
                    done,
                    adapted if binary_score or reported_score is not None else info,
                )
            if won or lost or ended:
                adapted["extra.catalog_telemetry"] = {
                    "synthetic": False,
                    "synthetic_score": binary_score,
                    "terminal": ("won" if won else "lost" if lost else "ended"),
                    "win_source": (
                        "native"
                        if info.get("won")
                        else (
                            "walkthrough" if route_outcome == "won"
                            else (
                                "terminal"
                                if won_from_text
                                else "score" if won_from_score else None
                            )
                        )
                    ),
                    "winning_walkthrough_id": self.winning_route_id,
                    "terminal_walkthrough_id": self.terminal_route_id,
                }
            return done or won or lost or ended, adapted

        adapted["score"] = int(won)
        adapted["max_score"] = 1
        adapted["moves"] = self.moves
        adapted["extra.catalog_telemetry"] = {
            "synthetic": True,
            "terminal": (
                "won" if won else "lost" if lost else "ended" if ended else None
            ),
            "win_source": (
                ("walkthrough" if route_outcome == "won" else "terminal")
                if won
                else None
            ),
            "winning_walkthrough_id": self.winning_route_id,
            "terminal_walkthrough_id": self.terminal_route_id,
        }
        return done or won or lost or ended, adapted


def _looks_like_win(observation: str) -> bool:
    normalized = " ".join(observation.casefold().split())
    if (
        "well, okay, not really" in normalized
        and "exit point in the current vr session" in normalized
    ):
        return False
    if re.search(
        r"\*{3}\s*(?:you have won\b|you win\b|you are victorious\b)[^*]*\*{3}",
        normalized,
    ):
        return True
    return (
        "congratulations" in normalized
        and any(
            marker in normalized
            for marker in (
                "adventure is over",
                "completed the game",
                "restart, restore",
                "restart or quit",
            )
        )
        or any(
            marker in normalized
            for marker in (
                "your descendants will be as countless as the stars of the sky",
                "you caught the murderer!",
            )
        )
    )


def _looks_like_loss(observation: str) -> bool:
    normalized = " ".join(observation.casefold().split())
    if "you have died" in normalized and "just kidding" in normalized:
        return False
    if re.search(
        r"\*{3}\s*(?:you have died\b|you are dead\b|you have lost\b|game over\b)"
        r"[^*]*\*{3}",
        normalized,
    ):
        return True
    if any(
        marker in normalized
        for marker in (
            "now you are dead you stupid player",
            "you are dead before you hit the ground",
        )
    ):
        return True
    return _has_restart_prompt(normalized) and bool(
        re.search(
            r"\b(?:you (?:have died|are dead|have lost)|game over)\b",
            normalized,
        )
    )


def _looks_like_end(observation: str) -> bool:
    normalized = " ".join(observation.casefold().split())
    return bool(
        re.search(r"\*{3}\s*the end[.!?]?\s*\*{3}", normalized)
        or _has_restart_prompt(normalized)
        or "press enter to quit the game" in normalized
        or "this adventure is over" in normalized
    )


def _has_restart_prompt(normalized: str) -> bool:
    return "would you like to restart, restore" in normalized


def _has_full_score(info: dict, reported_score: tuple[int, int] | None) -> bool:
    if reported_score is not None:
        score, max_score = reported_score
        if max_score > 0 and score >= max_score:
            return True

    score = info.get("score")
    max_score = info.get("max_score")
    if (
        isinstance(score, (int, float))
        and isinstance(max_score, (int, float))
        and max_score > 0
        and score >= max_score
    ):
        return True
    return False


def _reported_score(observation: str) -> tuple[int, int] | None:
    normalized = " ".join(observation.casefold().split())
    patterns = (
        r"scored\s+(-?\d+)\s+(?:points\s+)?out of "
        r"(?:a\s+)?(?:possible|maximum)(?:\s+of)?\s+(-?\d+)",
        r"(?:your\s+)?score\s+is\s+(-?\d+)\s+(?:out\s+)?of\s+"
        r"(?:a\s+)?possible\s+(-?\d+)",
    )
    for pattern in patterns:
        match = re.search(pattern, normalized)
        if match:
            return int(match.group(1)), int(match.group(2))
    return None
