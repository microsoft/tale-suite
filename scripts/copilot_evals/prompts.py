"""Prompts for the tool-calling Jericho agent.

Every model — including recent Claude releases that will not reliably emit a
bare action line — acts through the structured ``game_action`` tool, so the
prompt only has to deliver observations.
"""

from __future__ import annotations

from typing import Sequence

SYSTEM_PROMPT = (
    "You are playing a text-based game and your goal is to finish it with the"
    " highest score. A native function tool named `game_action` is available in"
    " your tool list. Read each observation carefully, think step by step about"
    " the best next move, then invoke that native tool exactly once with a single"
    " short command, e.g. `get lamp`, `north` or `inventory`. Never print or"
    " serialize a proposed tool call as text, JSON, XML, Markdown, or a code"
    " block; use the model's native tool-calling interface. Never call the tool"
    " more than once per observation or pass more than one command in a call."
    " For a single-key prompt, pass one literal space for SPACE or any-key"
    " screens, pass `<ENTER>` for the Enter or Return key, and pass"
    " `<BACKSPACE>` for Backspace; invoke the tool instead of describing the"
    " keypress."
    " After the tool call you will receive the next game observation. When stuck,"
    " try the `help` command to see what commands are available."
)

NUDGE_PROMPT = (
    "Your previous response did not invoke the native `game_action` function."
    " Invoke it now with exactly one short command for the observation above."
    " Do not print JSON, Markdown, or any plain-text answer."
)

RECAP_HEADER = (
    "// The previous session was interrupted and has been restored. Complete"
    " gameplay history:\n"
)


def observation_prompt(observation: str) -> str:
    """Mirror the TALES agent turn format: the observation then a prompt caret."""
    return f"{observation}\n> "


def recap_prompt(history: Sequence[tuple[str, str]], observation: str) -> str:
    """Restore complete TALES observation/action history in a new session."""
    if not history:
        return observation_prompt(observation)

    lines = [RECAP_HEADER]
    for past_observation, action in history:
        lines.append(f"{past_observation}\n> {action}\n")
    lines.append("// End of restored history. Continue playing.\n")
    lines.append(observation_prompt(observation))
    return "\n".join(lines)


def avoid_crashed_actions_prompt(prompt: str, actions: Sequence[str]) -> str:
    rendered = ", ".join(repr(action) for action in actions)
    guidance = ""
    if len(actions) >= 3:
        guidance += (
            " Do not submit a spelling, synonym, or phrasing variant of these "
            "commands; choose a semantically different action."
        )
    if any(
        action.casefold().split(maxsplit=1)[0] in {"choose", "click", "select", "type"}
        for action in actions
        if action.split()
    ):
        guidance += (
            " Avoid menu-selection verbs such as choose, click, select, or type "
            "at this state; inspect something else, move away, or use another "
            "ordinary game command."
        )
    return (
        f"{prompt}\n"
        "// Native-runtime safety: the following action(s) crashed or corrupted "
        f"this exact game state and must not be retried: {rendered}. Choose a "
        f"different game command.{guidance}"
    )


def crashed_action_nudge(action: str) -> str:
    return (
        f"The action {action!r} previously crashed or corrupted the native game "
        "at this exact state, so it was not executed. Invoke `game_action` once "
        "with a different command. Do not retry that action."
    )
