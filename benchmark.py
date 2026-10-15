import argparse
import datetime
import glob
import importlib
import json
import logging
import os
import re
import sys
import time
from functools import partial
from os.path import join as pjoin

import gymnasium as gym
import pandas as pd
import wandb
from termcolor import colored
from tqdm import tqdm

import tales
from tales.logger import log, setup_logging
from tales.observation_rewriter import apply_keyword_aliases
from tales.utils import NumpyEncoder
from tales.wandb_utils import WANDB_PROJECT, fetch_run_trajectory, find_matching_run

os.environ["WANDB_MODE"] = "disabled"

BLOCKED_STATE_COMMANDS = frozenset(
    {"save", "load", "restore", "undo", "script", "transcript"}
)
BLOCKED_LEADING_INTERPRETER_COMMANDS = frozenset(
    {
        "abstract",
        "actions",
        "changes",
        "gonear",
        "goto",
        "predictable",
        "purloin",
        "random",
        "recording",
        "replay",
        "routines",
        "scope",
        "showobj",
        "showverb",
        "timers",
        "trace",
        "tree",
    }
)
INVALID_STATE_COMMAND_ACTION = "talesblockedcommand"
INVALID_UNSAFE_ACTION = "talesunsafeaction"
ENTER_GAME_ACTION = "<ENTER>"
BACKSPACE_GAME_ACTION = "<BACKSPACE>"
WASTED_STEP_FEEDBACK = "The rejected action was discarded; this step was wasted."
_ACTION_WORD_RE = re.compile(r"[a-z]+")
OBSERVATION_REWRITE_STAT_KEYS = (
    "rewritten_observation",
    "observation_rewrite_prompt",
    "observation_rewrite_model",
    "observation_rewrite_nb_tokens",
    "observation_rewrite_nb_tokens_prompt",
    "observation_rewrite_nb_tokens_response",
    "observation_rewrite_nb_tokens_thinking",
    "observation_rewrite_cache_read_tokens",
    "observation_rewrite_cache_write_tokens",
    "observation_rewrite_duration",
    "pre_alias_rewritten_observation",
    "observation_keyword_alias_profile",
    "observation_keyword_alias_replacements",
    "observation_keyword_alias_replacement_count",
)


def _blocked_state_command(action):
    """Return the first disabled interpreter command in an action."""
    words = _ACTION_WORD_RE.findall(action.casefold())
    for word in words:
        # Z-machine versions 1-3 compare only the first six dictionary
        # characters, so "restoring" and "restorx" invoke RESTORE in Zork I.
        if word.startswith("restor"):
            return "restore"
        if word in BLOCKED_STATE_COMMANDS:
            return word
    if words and words[0] in BLOCKED_LEADING_INTERPRETER_COMMANDS:
        return words[0]
    return None


def _resolve_action(action, disable_state_commands):
    """Resolve a proposed action to the text that may reach the environment."""
    if action.strip().casefold() in {"<enter>", "<return>"}:
        return "", None
    if action.strip().casefold() == "<backspace>":
        return "\x08", None
    if not disable_state_commands:
        return action, None
    blocked_command = _blocked_state_command(action)
    if blocked_command is None:
        return action, None
    return INVALID_STATE_COMMAND_ACTION, blocked_command


def _make_state(obs, info):
    """Create mutable game state dict."""
    return {
        "step": 0,
        "score": 0,
        "moves": 0,
        "highscore": 0,
        "max_score": info["max_score"],
        "nb_wins": 0,
        "nb_losts": 0,
        "nb_resets": 0,
        "nb_invalid_actions": 0,
        "nb_blocked_state_commands": 0,
        "nb_wasted_steps": 0,
        "nb_translated_actions": 0,
        "nb_actions_with_canonical_keywords": 0,
        "nb_environment_steps": 0,
        "obs": obs,
        "done": False,
        "info": info,
        "results": [],
    }


def _step_env(env, state, action):
    """Execute an action and update state. Returns (prev_obs, feedback)."""
    prev_obs = state["obs"]

    if "\n" in action.strip():
        state["obs"] = "The game only allows one action per step."
    else:
        obs, _, done, info = env.step(action)
        state["obs"] = obs
        state["done"] = done
        state["info"] = info

    state["score"] = state["info"]["score"]
    state["moves"] = state["info"]["moves"]
    state["highscore"] = max(state["score"], state["highscore"])

    return prev_obs, state["info"]["feedback"]


def _check_invalid(state, action, admissible_commands):
    """Track invalid actions when admissible commands are enabled."""
    if (
        admissible_commands
        and state["info"]["admissible_commands"]
        and action not in state["info"]["admissible_commands"]
    ):
        state["nb_invalid_actions"] += 1


def _handle_done(env, agent, state, env_name, break_on_max=True):
    """Handle game-over (win/loss/reset). Returns True if loop should break."""
    if not state["done"]:
        return False

    if state["info"]["won"]:
        state["nb_wins"] += 1
        if state["highscore"] == state["max_score"]:
            log.debug(state["obs"])
            if break_on_max:
                return True  # Break: no reason to play more.
            else:
                return False  # Replay: don't break or reset.
    elif state["info"]["lost"]:
        state["nb_losts"] += 1

    # Restart the game to try for a better score.
    last_obs = state["obs"]
    obs, info = env.reset()
    state["obs"] = last_obs + "\n\n-= Restarting =-\n" + obs
    state["info"] = info
    state["done"] = False
    agent.reset(state["obs"], info, env_name)
    state["nb_resets"] += 1
    log.debug(state["obs"])
    return False


def _record_step(
    state,
    prev_obs,
    action,
    feedback,
    token_stats,
    wandb_run,
    *,
    proposed_action,
    blocked_state_command,
    action_was_wasted=False,
    unsafe_proposed_action=None,
    unsafe_action_reason=None,
    unsafe_action_source=None,
    translated_action=None,
    action_keyword_alias_replacements=None,
    action_canonical_keyword_matches=None,
    executed_action=None,
    environment_step_called=False,
):
    """Append step to results and log to wandb."""
    s = state
    norm_score = s["score"] / s["max_score"]
    norm_highscore = s["highscore"] / s["max_score"]

    step_metrics = {
        "episode/moves": s["moves"],
        "episode/score": s["score"],
        "episode/highscore": s["highscore"],
        "episode/normalized_score": norm_score,
        "episode/normalized_highscore": norm_highscore,
        "episode/token_usage": token_stats["nb_tokens"],
        "episode/token_usage_thinking": token_stats.get("nb_tokens_thinking", 0),
        "episode/blocked_state_commands": s["nb_blocked_state_commands"],
        "episode/wasted_steps": s["nb_wasted_steps"],
        "episode/translated_actions": s["nb_translated_actions"],
        "episode/actions_with_canonical_keywords": s[
            "nb_actions_with_canonical_keywords"
        ],
        "episode/environment_steps": s["nb_environment_steps"],
    }
    if token_stats.get("rewritten_observation") is not None:
        step_metrics["episode/observation_rewrite_token_usage"] = token_stats.get(
            "observation_rewrite_nb_tokens", 0
        )
    wandb_run.log(step_metrics, step=s["step"])

    # fmt: off
    s["results"].append([
        s["step"], s["score"], s["max_score"], norm_score, s["moves"],
        prev_obs, action, feedback,
        token_stats["prompt"], token_stats["response"], token_stats.get("thinking"),
        token_stats.get("thinking_is_summary"),
        token_stats["nb_tokens"], token_stats["nb_tokens_prompt"],
        token_stats["nb_tokens_response"], token_stats.get("nb_tokens_thinking", 0),
        proposed_action, blocked_state_command is not None, blocked_state_command,
        action_was_wasted, unsafe_proposed_action, unsafe_action_reason,
        unsafe_action_source,
        token_stats.get("rewritten_observation"),
        token_stats.get("observation_rewrite_prompt"),
        token_stats.get("observation_rewrite_model"),
        token_stats.get("observation_rewrite_nb_tokens"),
        token_stats.get("observation_rewrite_nb_tokens_prompt"),
        token_stats.get("observation_rewrite_nb_tokens_response"),
        token_stats.get("observation_rewrite_nb_tokens_thinking"),
        token_stats.get("observation_rewrite_cache_read_tokens"),
        token_stats.get("observation_rewrite_cache_write_tokens"),
        token_stats.get("observation_rewrite_duration"),
        token_stats.get("pre_alias_rewritten_observation"),
        token_stats.get("observation_keyword_alias_profile"),
        token_stats.get("observation_keyword_alias_replacements"),
        token_stats.get("observation_keyword_alias_replacement_count"),
        translated_action if translated_action is not None else proposed_action,
        translated_action is not None and translated_action != proposed_action,
        action_keyword_alias_replacements or [],
        sum(
            item.get("count", 0)
            for item in (action_keyword_alias_replacements or [])
        ),
        action_canonical_keyword_matches or [],
        sum(
            item.get("count", 0)
            for item in (action_canonical_keyword_matches or [])
        ),
        executed_action,
        environment_step_called,
    ])
    # fmt: on

    rewrite_file = s.get("observation_rewrite_file")
    if rewrite_file and token_stats.get("rewritten_observation") is not None:
        rewrite_record = {
            "Step": s["step"],
            "Observation": prev_obs,
            "Rewritten Observation": token_stats["rewritten_observation"],
            "Observation Rewrite Prompt": token_stats.get("observation_rewrite_prompt"),
            "Observation Rewrite Model": token_stats.get("observation_rewrite_model"),
            "Observation Rewrite Token Usage": token_stats.get(
                "observation_rewrite_nb_tokens"
            ),
            "Observation Rewrite Prompt Tokens": token_stats.get(
                "observation_rewrite_nb_tokens_prompt"
            ),
            "Observation Rewrite Response Tokens": token_stats.get(
                "observation_rewrite_nb_tokens_response"
            ),
            "Observation Rewrite Thinking Tokens": token_stats.get(
                "observation_rewrite_nb_tokens_thinking"
            ),
            "Observation Rewrite Cache Read Tokens": token_stats.get(
                "observation_rewrite_cache_read_tokens"
            ),
            "Observation Rewrite Cache Write Tokens": token_stats.get(
                "observation_rewrite_cache_write_tokens"
            ),
            "Observation Rewrite Duration": token_stats.get(
                "observation_rewrite_duration"
            ),
            "Pre-Alias Rewritten Observation": token_stats.get(
                "pre_alias_rewritten_observation"
            ),
            "Observation Keyword Alias Profile": token_stats.get(
                "observation_keyword_alias_profile"
            ),
            "Observation Keyword Alias Replacements": token_stats.get(
                "observation_keyword_alias_replacements"
            ),
            "Observation Keyword Alias Replacement Count": token_stats.get(
                "observation_keyword_alias_replacement_count"
            ),
            "Proposed Action": proposed_action,
            "Translated Action": (
                translated_action if translated_action is not None else proposed_action
            ),
            "Action Was Translated": (
                translated_action is not None and translated_action != proposed_action
            ),
            "Action Keyword Alias Replacements": (
                action_keyword_alias_replacements or []
            ),
            "Action Keyword Alias Replacement Count": sum(
                item.get("count", 0)
                for item in (action_keyword_alias_replacements or [])
            ),
            "Action Canonical Keyword Matches": (
                action_canonical_keyword_matches or []
            ),
            "Action Canonical Keyword Match Count": sum(
                item.get("count", 0)
                for item in (action_canonical_keyword_matches or [])
            ),
            "Executed Action": executed_action,
            "Environment Step Called": environment_step_called,
            "Action": action,
        }
        with open(rewrite_file, "a", encoding="utf-8") as writer:
            json.dump(rewrite_record, writer, cls=NumpyEncoder)
            writer.write("\n")
            writer.flush()


def replay_trajectory(
    env, agent, trajectory_df, state, wandb_run, args, env_name, start_time
):
    """Replay recorded actions through the environment (no LLM calls).

    Feeds each action from the trajectory to the environment, verifies
    observations match, and builds agent history for subsequent LLM play.
    """
    replay_steps = len(trajectory_df)
    log.info(colored(f"Replaying {replay_steps} steps...", "cyan"))

    replay_pbar = tqdm(
        trajectory_df.iterrows(),
        total=args.nb_steps,
        desc=f"  {env_name} (replay)",
        unit="steps",
        leave=False,
    )
    for _, row in replay_pbar:
        state["step"] = int(row["Step"])
        recorded_action = str(row["Action"])
        proposed_value = row.get("Proposed Action")
        proposed_action = (
            proposed_value if isinstance(proposed_value, str) else recorded_action
        )
        translated_value = row.get("Translated Action")
        translated_action = (
            translated_value if isinstance(translated_value, str) else recorded_action
        )
        action_keyword_alias_replacements = row.get("Action Keyword Alias Replacements")
        if not isinstance(action_keyword_alias_replacements, list):
            action_keyword_alias_replacements = []
        action_canonical_keyword_matches = row.get("Action Canonical Keyword Matches")
        if not isinstance(action_canonical_keyword_matches, list):
            action_canonical_keyword_matches = []
        if translated_action != proposed_action:
            state["nb_translated_actions"] += 1
        if action_canonical_keyword_matches:
            state["nb_actions_with_canonical_keywords"] += 1
        action, detected_block = _resolve_action(
            recorded_action,
            getattr(args, "disable_state_commands", False),
        )
        recorded_block = row.get("Blocked State Command")
        blocked_state_command = (
            recorded_block.strip()
            if isinstance(recorded_block, str) and recorded_block.strip()
            else detected_block
        )
        action_was_wasted = row.get("Action Was Wasted") is True
        if blocked_state_command is not None and not action_was_wasted:
            raise ValueError(
                "Cannot replay a blocked state command that was not recorded "
                "as a wasted step."
            )
        unsafe_proposed_action = row.get("Unsafe Proposed Action")
        unsafe_action_reason = row.get("Unsafe Action Reason")
        unsafe_action_source = row.get("Unsafe Action Source")
        if blocked_state_command is not None:
            state["nb_blocked_state_commands"] += 1
        if action_was_wasted:
            state["nb_wasted_steps"] += 1

        replay_pbar.set_postfix_str(
            f"Score: {state['info']['score']}/{state['info']['max_score']}"
            f" ({state['info']['score']/state['info']['max_score']:.1%})"
        )

        if action_was_wasted:
            prev_obs = state["obs"]
            feedback = row.get("Feedback") or WASTED_STEP_FEEDBACK
            executed_action = None
            environment_step_called = False
        else:
            prev_obs, feedback = _step_env(env, state, action)
            state["nb_environment_steps"] += 1
            executed_action = action
            environment_step_called = True
            _check_invalid(state, action, args.admissible_commands)

        # Verify replay fidelity.
        logged_obs = row.get("Observation")
        if logged_obs is not None and isinstance(logged_obs, str):
            if prev_obs.strip() != logged_obs.strip():
                log.warning(
                    f"Replay divergence at step {state['step']}:\n"
                    f"  Expected: {logged_obs[:200]!r}\n"
                    f"  Got:      {prev_obs[:200]!r}"
                )

        # Build agent history for subsequent LLM calls.
        rewritten_value = row.get("Rewritten Observation")
        history_observation = (
            rewritten_value if isinstance(rewritten_value, str) else prev_obs
        )
        agent.history.append((f"{history_observation}\n> ", f"{proposed_action}\n"))

        msg = "{:5d}. Time: {:9.2f}\tScore: {:3d}\tMove: {:5d}\tAction: {:20s} (replay)"
        msg = msg.format(
            state["step"],
            time.time() - start_time,
            state["score"],
            state["moves"],
            action,
        )
        log.info(msg)

        # Use original token stats from the trajectory.
        token_stats = {
            "prompt": row.get("Prompt", "") or "",
            "response": row.get("Response", "") or "",
            "thinking": row.get("Thinking"),
            "thinking_is_summary": row.get("Thinking Is Summary"),
            "nb_tokens": row.get("Token Usage", 0) or 0,
            "nb_tokens_prompt": row.get("Prompt Tokens", 0) or 0,
            "nb_tokens_response": row.get("Response Tokens", 0) or 0,
            "nb_tokens_thinking": row.get("Thinking Tokens", 0) or 0,
        }
        for key, column in zip(
            OBSERVATION_REWRITE_STAT_KEYS,
            (
                "Rewritten Observation",
                "Observation Rewrite Prompt",
                "Observation Rewrite Model",
                "Observation Rewrite Token Usage",
                "Observation Rewrite Prompt Tokens",
                "Observation Rewrite Response Tokens",
                "Observation Rewrite Thinking Tokens",
                "Observation Rewrite Cache Read Tokens",
                "Observation Rewrite Cache Write Tokens",
                "Observation Rewrite Duration",
                "Pre-Alias Rewritten Observation",
                "Observation Keyword Alias Profile",
                "Observation Keyword Alias Replacements",
                "Observation Keyword Alias Replacement Count",
            ),
        ):
            token_stats[key] = row.get(column)
        _record_step(
            state,
            prev_obs,
            action,
            feedback,
            token_stats,
            wandb_run,
            proposed_action=proposed_action,
            blocked_state_command=blocked_state_command,
            action_was_wasted=action_was_wasted,
            unsafe_proposed_action=unsafe_proposed_action,
            unsafe_action_reason=unsafe_action_reason,
            unsafe_action_source=unsafe_action_source,
            translated_action=translated_action,
            action_keyword_alias_replacements=(action_keyword_alias_replacements),
            action_canonical_keyword_matches=(action_canonical_keyword_matches),
            executed_action=executed_action,
            environment_step_called=environment_step_called,
        )

        if not state["done"]:
            log.debug(state["obs"])

        _handle_done(env, agent, state, env_name, break_on_max=False)

    replay_pbar.close()

    if state["highscore"] == state["max_score"]:
        log.info(
            colored(
                f"Replay complete: game already won with max score "
                f"({state['highscore']}/{state['max_score']}). No further steps needed.",
                "green",
            )
        )
        return args.nb_steps  # Signal: skip the play loop.
    else:
        log.info(
            colored(
                f"Replay complete: {replay_steps} steps, score={state['score']}, "
                f"highscore={state['highscore']}. "
                f"LLM takes over from step {replay_steps + 1}.",
                "cyan",
            )
        )
        return replay_steps


def play_with_agent(
    env, agent, state, wandb_run, args, env_name, start_time, start_step
):
    """Play the game with the LLM agent from start_step to nb_steps."""
    pbar = tqdm(
        range(start_step, args.nb_steps + 1),
        initial=start_step - 1,
        total=args.nb_steps,
        desc=f"  {env_name}",
        unit="steps",
        leave=False,
    )
    for step in pbar:
        state["step"] = step
        environment = getattr(env, "unwrapped", env)
        refresh_context = getattr(
            environment,
            "set_submitted_step_context",
            None,
        )
        if callable(refresh_context):
            state["obs"] = refresh_context(state["obs"], step)
        pbar.set_postfix_str(
            f"Score: {state['info']['score']}/{state['info']['max_score']}"
            f" ({state['info']['score']/state['info']['max_score']:.1%})"
        )

        proposed_action, stats = agent.act(
            state["obs"], state["score"], state["done"], state["info"]
        )
        action_was_unsafe = stats.get("action_was_unsafe") is True
        unsafe_proposed_action = stats.get("unsafe_action")
        unsafe_action_reason = stats.get("unsafe_action_reason")
        unsafe_action_source = stats.get("unsafe_action_source")
        if action_was_unsafe and proposed_action != INVALID_UNSAFE_ACTION:
            raise ValueError("Unsafe action did not use the canonical sentinel.")
        alias_profile = getattr(args, "observation_keyword_alias_profile", None)
        canonical_action_result = apply_keyword_aliases(
            proposed_action,
            alias_profile,
        )
        action_canonical_keyword_matches = [
            item.as_dict() for item in canonical_action_result.replacements
        ]
        if action_canonical_keyword_matches:
            state["nb_actions_with_canonical_keywords"] += 1
        action_alias_result = apply_keyword_aliases(
            proposed_action,
            alias_profile,
            reverse=True,
        )
        translated_action = action_alias_result.text
        action_keyword_alias_replacements = [
            item.as_dict() for item in action_alias_result.replacements
        ]
        if translated_action != proposed_action:
            state["nb_translated_actions"] += 1
        action, blocked_state_command = _resolve_action(
            translated_action,
            getattr(args, "disable_state_commands", False),
        )
        if blocked_state_command is not None:
            state["nb_blocked_state_commands"] += 1
            log.warning(
                "Blocked state command %r in proposed action %r; executing %r.",
                blocked_state_command,
                proposed_action,
                action,
            )
        action_was_wasted = action_was_unsafe or blocked_state_command is not None
        if action_was_wasted:
            state["nb_wasted_steps"] += 1
        logged_proposal = (
            unsafe_proposed_action if action_was_unsafe else proposed_action
        )
        log.debug(colored(f"> {logged_proposal!r}", "green"))

        if args.debug:
            breakpoint()

        if action_was_wasted:
            prev_obs = state["obs"]
            feedback = WASTED_STEP_FEEDBACK
            executed_action = None
            environment_step_called = False
            log.warning(
                "Discarded action before env.step; step %d was wasted "
                "(blocked_state_command=%r, unsafe_reason=%r, proposal=%r).",
                step,
                blocked_state_command,
                unsafe_action_reason,
                logged_proposal,
            )
        else:
            prev_obs, feedback = _step_env(env, state, action)
            state["nb_environment_steps"] += 1
            executed_action = action
            environment_step_called = True
            _check_invalid(state, action, args.admissible_commands)

        msg = "{:5d}. Time: {:9.2f}\tScore: {:3d}\tMove: {:5d}" "\tAction: {:20s}{}"
        msg = msg.format(
            step,
            time.time() - start_time,
            state["score"],
            state["moves"],
            action,
            " (wasted)" if action_was_wasted else "",
        )
        log.info(msg)

        token_stats = {
            "prompt": stats["prompt"],
            "response": stats["response"],
            "thinking": stats.get("thinking"),
            "thinking_is_summary": stats.get("thinking_is_summary"),
            "nb_tokens": stats["nb_tokens"],
            "nb_tokens_prompt": stats["nb_tokens_prompt"],
            "nb_tokens_response": stats["nb_tokens_response"],
            "nb_tokens_thinking": stats.get("nb_tokens_thinking", 0),
        }
        token_stats.update(
            {key: stats.get(key) for key in OBSERVATION_REWRITE_STAT_KEYS}
        )
        _record_step(
            state,
            prev_obs,
            action,
            feedback,
            token_stats,
            wandb_run,
            proposed_action=proposed_action,
            blocked_state_command=blocked_state_command,
            action_was_wasted=action_was_wasted,
            unsafe_proposed_action=unsafe_proposed_action,
            unsafe_action_reason=unsafe_action_reason,
            unsafe_action_source=unsafe_action_source,
            translated_action=translated_action,
            action_keyword_alias_replacements=(action_keyword_alias_replacements),
            action_canonical_keyword_matches=(action_canonical_keyword_matches),
            executed_action=executed_action,
            environment_step_called=environment_step_called,
        )

        if not state["done"]:
            log.debug(state["obs"])

        if _handle_done(env, agent, state, env_name, break_on_max=True):
            break


def evaluate(agent, env_name, args):
    # Fetch trajectory if continuing from a previous run.
    trajectory_df = None
    continue_from = getattr(args, "continue_from", None)
    if continue_from:
        # Auto-find matching run if no explicit run ID was provided.
        if continue_from == "auto":
            continue_from = find_matching_run(env_name, agent.params, args.game_seed)
            if continue_from is None:
                log.info(
                    colored(
                        f"No matching previous run found for {env_name}. "
                        f"Running from scratch.",
                        "yellow",
                    )
                )

    if continue_from:
        original_config, trajectory_df = fetch_run_trajectory(continue_from)

        # Validate that the game matches.
        original_game = original_config.get("game")
        if original_game and original_game != env_name:
            raise ValueError(
                f"Environment mismatch: --continue-from run played '{original_game}' "
                f"but current run targets '{env_name}'."
            )

        # Override game_seed from original run to ensure deterministic replay.
        original_seed = original_config.get("game_seed")
        if original_seed is not None and original_seed != args.game_seed:
            log.info(
                f"Overriding --game-seed from {args.game_seed} to {original_seed} "
                f"(from original run)."
            )
            args.game_seed = original_seed

        # Truncate trajectory if it has more steps than the target.
        if len(trajectory_df) > args.nb_steps:
            log.info(
                f"Trajectory has {len(trajectory_df)} steps but target is {args.nb_steps}. "
                f"Truncating replay to {args.nb_steps} steps."
            )
            trajectory_df = trajectory_df.iloc[: args.nb_steps]

        replay_steps = len(trajectory_df)
        log.info(
            colored(
                f"Continuing from run {continue_from}: "
                f"replaying {replay_steps} steps, then LLM takes over up to {args.nb_steps}.",
                "cyan",
            )
        )

    state_policy = "_nostate1" if getattr(args, "disable_state_commands", False) else ""
    rewriter_model = getattr(args, "observation_rewriter_model", None)
    rewrite_policy = ""
    if rewriter_model:
        safe_rewriter_model = re.sub(r"[^a-zA-Z0-9.-]+", "-", rewriter_model)
        rewrite_effort = getattr(args, "observation_rewriter_reasoning_effort", "max")
        rewrite_policy = f"_rewrite-{safe_rewriter_model}-{rewrite_effort}"
        rewrite_condition_sha256 = agent.params.get(
            "observation_rewrite_condition_sha256"
        )
        if rewrite_condition_sha256:
            rewrite_policy += f"-{rewrite_condition_sha256[:12]}"
    alias_profile = getattr(args, "observation_keyword_alias_profile", None)
    alias_policy = ""
    if alias_profile:
        safe_alias_profile = re.sub(r"[^a-zA-Z0-9.-]+", "-", alias_profile)
        alias_policy = f"_aliases-{safe_alias_profile}"
    env_params = (
        f"a{int(args.admissible_commands)}{state_policy}{rewrite_policy}"
        f"{alias_policy}"
        f"_s{args.game_seed}_steps{args.nb_steps}"
    )
    logdir = pjoin(args.log_dir, f"{env_name}")
    os.makedirs(logdir, exist_ok=True)
    summary_file = pjoin(logdir, f"{env_params}.json")
    rollouts_file = pjoin(logdir, f"{env_params}.jsonl")
    log_file = pjoin(logdir, f"{env_params}.log")
    rewrite_file = (
        pjoin(logdir, f"{env_params}.observation_rewrites.jsonl")
        if rewriter_model
        else None
    )

    # Create new file handler for this env evaluation.
    fh = log.add_new_file_handler(log_file)

    # Check if the game has already been evaluated.
    if not args.force_all and os.path.exists(summary_file):
        log.info(f"Previous evaluation found: {summary_file}")
        with open(summary_file) as reader:
            summary = json.load(reader)

        log.info(f"Previous evaluation status: {summary['status']}")
        if not args.force_failed or summary["status"] == "finished":
            log.info(colored("Skipped, already done.", "yellow"))
            log.removeHandler(fh)
            return summary

    run_name = f"{env_name} - {agent.uid}"
    if args.wandb and not args.force_all:
        # Check if there already exists a run with the same name using Wandb API.
        wandb_api = wandb.Api()
        wandb_runs = wandb_api.runs(filters={"display_name": run_name})
        if wandb_runs:
            wandb_run = wandb_runs[0]
            log.info(f"Previous evaluation found: {wandb_run.url} ({wandb_run.state})")
            if wandb_run.state in ("finished", "running"):
                log.info(colored("Skipped, already exists.", "yellow"))
                log.removeHandler(fh)
                summary = {
                    "status": wandb_run.state,
                    "env_name": env_name,
                    "env_params": env_params,
                    "wandb_run_id": wandb_run.id,
                    "wandb_url": wandb_run.url,
                    "nb_steps": wandb_run.summary["total/Env. Steps"],
                    "nb_moves": wandb_run.summary["total/Game Moves"],
                    "nb_invalid_actions": wandb_run.summary["total/Invalid Actions"],
                    "nb_losts": wandb_run.summary["total/Losts"],
                    "nb_wins": wandb_run.summary["total/Wins"],
                    "nb_resets": wandb_run.summary["total/Resets"],
                    "highscore": wandb_run.summary["final/Highscore"],
                    "max_score": wandb_run.summary["final/Game Max Score"],
                    "norm_score": wandb_run.summary["final/Normalized Score"],
                    "duration": wandb_run.summary["final/Duration"],
                }
                return summary

    # initialize wandb
    wandb_config = {
        "version": tales.__version__,
        "game": env_name,
        "framework": tales.env2task[env_name],
        "agent": agent.uid,
        "max_steps": args.nb_steps,
        "game_seed": args.game_seed,
        "admissible_commands": args.admissible_commands,
        "disable_state_commands": getattr(args, "disable_state_commands", False),
        **agent.params,
    }
    if trajectory_df is not None:
        wandb_config["continued_from_run_id"] = original_config["_run_id"]
        wandb_config["continued_from_run_url"] = original_config["_run_url"]
        wandb_config["replay_steps"] = len(trajectory_df)
    wandb_run = wandb.init(
        project=WANDB_PROJECT,
        config=wandb_config,
        reinit=True,
        name=run_name,
    )
    if rewrite_file:
        with open(rewrite_file, "w", encoding="utf-8"):
            pass

    env = gym.make(
        f"tales/{env_name}-v0",
        disable_env_checker=True,
        admissible_commands=args.admissible_commands,
    )

    log.debug(f"Using {env.__class__.__name__}")
    log.debug(f"Playing {env_name}")

    start_time = time.time()
    obs, info = env.reset(seed=args.game_seed)

    agent = agent.new()
    agent.reset(obs, info, env_name)

    log.debug(f"Environment reset.\n{obs}\n")

    state = _make_state(obs, info)
    state["observation_rewrite_file"] = rewrite_file

    wandb_run.log(
        {
            "episode/moves": 0,
            "episode/score": 0,
            "episode/highscore": 0,
            "episode/normalized_score": 0,
            "episode/normalized_highscore": 0,
            "episode/token_usage": 0,
            "episode/translated_actions": 0,
            "episode/actions_with_canonical_keywords": 0,
            "episode/environment_steps": 0,
        },
        step=0,
    )

    # Replay phase (if continuing from a previous run).
    replay_steps = 0
    if trajectory_df is not None:
        replay_steps = replay_trajectory(
            env, agent, trajectory_df, state, wandb_run, args, env_name, start_time
        )

    # Play phase (LLM-driven).
    status = "running"
    try:
        play_with_agent(
            env,
            agent,
            state,
            wandb_run,
            args,
            env_name,
            start_time,
            start_step=replay_steps + 1,
        )
        status = "finished"

    except KeyboardInterrupt as e:
        status = "killed"
        log.critical(colored(f"{env_name} (killed)", "red"))
        log.error(str(e))
        time.sleep(1)
        if args.debug:
            raise

    except Exception as e:
        status = "failed"
        log.critical(colored(f"{env_name} ({e!r})", "red"))
        log.error(str(e), exc_info=True)
        if args.debug:
            raise

    env.close()

    final_stats = {
        "nb_steps": state["step"],
        "nb_moves": state["moves"],
        "nb_invalid_actions": state["nb_invalid_actions"],
        "nb_blocked_state_commands": state["nb_blocked_state_commands"],
        "nb_wasted_steps": state["nb_wasted_steps"],
        "nb_translated_actions": state["nb_translated_actions"],
        "nb_actions_with_canonical_keywords": state[
            "nb_actions_with_canonical_keywords"
        ],
        "nb_environment_steps": state["nb_environment_steps"],
        "nb_losts": state["nb_losts"],
        "nb_wins": state["nb_wins"],
        "nb_resets": state["nb_resets"],
        "highscore": state["highscore"],
        "max_score": state["max_score"],
        "norm_score": state["highscore"] / state["max_score"],
        "duration": time.time() - start_time,
    }

    # fmt: off
    columns = [
        "Step", "Score", "Max Score", "Normalized Score", "Moves",
        "Observation", "Action", "Feedback",
        "Prompt", "Response", "Thinking", "Thinking Is Summary",
        "Token Usage", "Prompt Tokens", "Response Tokens", "Thinking Tokens",
        "Proposed Action", "Action Was Blocked", "Blocked State Command",
        "Action Was Wasted", "Unsafe Proposed Action", "Unsafe Action Reason",
        "Unsafe Action Source",
        "Rewritten Observation", "Observation Rewrite Prompt",
        "Observation Rewrite Model", "Observation Rewrite Token Usage",
        "Observation Rewrite Prompt Tokens", "Observation Rewrite Response Tokens",
        "Observation Rewrite Thinking Tokens",
        "Observation Rewrite Cache Read Tokens",
        "Observation Rewrite Cache Write Tokens",
        "Observation Rewrite Duration",
        "Pre-Alias Rewritten Observation",
        "Observation Keyword Alias Profile",
        "Observation Keyword Alias Replacements",
        "Observation Keyword Alias Replacement Count",
        "Translated Action", "Action Was Translated",
        "Action Keyword Alias Replacements",
        "Action Keyword Alias Replacement Count",
        "Action Canonical Keyword Matches",
        "Action Canonical Keyword Match Count",
        "Executed Action", "Environment Step Called",
    ]
    # fmt: on
    df = pd.DataFrame(state["results"], columns=columns)
    df.to_json(rollouts_file, orient="records", lines=True)

    wandb_stats = {
        "total/Env. Steps": final_stats["nb_steps"],
        "total/Game Moves": final_stats["nb_moves"],
        "total/Invalid Actions": final_stats["nb_invalid_actions"],
        "total/Blocked State Commands": final_stats["nb_blocked_state_commands"],
        "total/Wasted Steps": final_stats["nb_wasted_steps"],
        "total/Translated Actions": final_stats["nb_translated_actions"],
        "total/Actions With Canonical Keywords": final_stats[
            "nb_actions_with_canonical_keywords"
        ],
        "total/Environment Steps": final_stats["nb_environment_steps"],
        "total/Losts": final_stats["nb_losts"],
        "total/Wins": final_stats["nb_wins"],
        "total/Resets": final_stats["nb_resets"],
        "total/Tokens": df["Token Usage"].sum(),
        "total/Prompt Tokens": df["Prompt Tokens"].sum(),
        "total/Response Tokens": df["Response Tokens"].sum(),
        "total/Thinking Tokens": df["Thinking Tokens"].sum(),
        "final/Highscore": final_stats["highscore"],
        "final/Game Max Score": final_stats["max_score"],
        "final/Normalized Score": final_stats["norm_score"],
        "final/Duration": final_stats["duration"],
    }
    if rewriter_model:
        wandb_stats.update(
            {
                "total/Observation Rewrite Tokens": df[
                    "Observation Rewrite Token Usage"
                ].sum(),
                "total/Observation Rewrite Prompt Tokens": df[
                    "Observation Rewrite Prompt Tokens"
                ].sum(),
                "total/Observation Rewrite Response Tokens": df[
                    "Observation Rewrite Response Tokens"
                ].sum(),
                "total/Observation Rewrite Thinking Tokens": df[
                    "Observation Rewrite Thinking Tokens"
                ].sum(),
            }
        )
    wandb_run.log(
        {"episode/rollout": wandb.Table(dataframe=df), **wandb_stats},
        step=final_stats["nb_steps"],
    )

    # Save summary.
    summary = {
        "status": status,
        "env_name": env_name,
        "env_params": env_params,
        "wandb_run_id": wandb_run.id,
        "wandb_url": wandb_run.url,
        **final_stats,
        **wandb_stats,
        "agent_params": agent.params,
    }
    if rewrite_file:
        summary["observation_rewrites_file"] = rewrite_file

    with open(summary_file, "w") as f:
        json.dump(summary, f, indent=2, sort_keys=True, cls=NumpyEncoder)

    wandb.save(rollouts_file)
    wandb.save(log_file)
    wandb.save(summary_file)
    if rewrite_file:
        wandb.save(rewrite_file)

    wandb_run.finish(exit_code=int(status != "finished"))

    log.removeHandler(fh)
    return summary


def benchmark(agent, args):
    # Log how many envs we are about to evaluate.
    log.critical("Evaluating {} interactive text environments:".format(len(args.envs)))

    mean_score = 0
    total_time = 0.0
    total_steps = 0
    total_invalid = 0

    nb_envs = 0
    max_env_name = max(map(len, args.envs))
    for env in tqdm(args.envs, desc="Benchmarking", unit="game", leave=False):
        summary = evaluate(agent, env, args)

        nb_envs += 1

        total_time += summary["duration"]  # In seconds
        total_steps += summary["nb_steps"]
        total_invalid += summary["nb_invalid_actions"]

        msg = (
            f"{env.ljust(max_env_name)}"
            f"  Steps: {summary['nb_steps']:4d}/{args.nb_steps:4d}"
            f"  Time: {datetime.timedelta(seconds=int(summary['duration']))}"
            f"{summary['nb_resets']:4d} resets"
            f"  Score: {summary['highscore']:3d}/{summary['max_score']:3d} ({summary['norm_score']:6.2%})"
        )

        log.critical(msg)

        mean_score += summary["norm_score"]

    if nb_envs > 0 and total_time > 0:
        log.critical(
            f"Mean score (over {nb_envs} games) = {mean_score / nb_envs:8.2%} of total possible"
        )
        log.critical(f"Total time {total_time:9.2f} seconds")
        log.critical(f"Total {total_invalid} invalid actions")
        log.critical(f"Avg. speed: {total_steps / total_time:8.2f} steps per second")


def pretty_print_tasks(num_cols: int = 3, disable_print: bool = False):
    output = []

    max_justify = max(
        len(env_name) for task in tales.envs_per_task.values() for env_name in task
    )

    for task in sorted(tales.envs_per_task):
        task_output = f"{'=' * 5} {task} {'=' * 5}\n"

        # Reference: https://stackoverflow.com/a/33464001
        for count, env_id in enumerate(sorted(tales.envs_per_task[task]), 1):
            # Print column with justification.
            task_output += env_id.ljust(max_justify) + " "

            # Once all rows printed, switch to new column.
            if count % num_cols == 0:
                task_output = task_output.rstrip(" ")

                if count != len(tales.envs_per_task[task]):
                    task_output += "\n"

        output.append(task_output.rstrip(" "))

    if disable_print:
        return "\n".join(output)
    else:
        print("\n".join(output))


def exit_listing_agents(agent=None):
    msg = ""
    if agent is not None:
        msg += "Unknown agent: {}\n\n".format(agent)

    msg += "Available agents:\n  "
    msg += "\n  ".join(sorted(tales.agent.AGENTS))
    print(msg)
    sys.exit(1)


def _maybe_load_agent_module():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument(
        "--agent",
        default="agents/*.py",
        help="Load external python file(s). Useful to register custom agent on-the-fly. Default: %(default)s",
    )
    args, _ = parser.parse_known_args()
    if args.agent:
        print(f"Importing agent(s) from {args.agent}.")
        for agent_file in glob.glob(args.agent):
            print(f"Importing {agent_file}...")
            agent_dirname = os.path.dirname(agent_file)
            agent_filename, _ = os.path.splitext(os.path.basename(agent_file))
            module_name = f"{agent_dirname}.{agent_filename}"
            if module_name in sys.modules:
                continue

            spec = importlib.util.spec_from_file_location(module_name, agent_file)
            module = importlib.util.module_from_spec(spec)
            sys.modules[module_name] = module
            try:
                spec.loader.exec_module(module)
            except BaseException:
                sys.modules.pop(module_name, None)
                raise


def parse_args():
    # fmt: off

    description = "Benchmark some agent on interactive text environments."
    general_parser = argparse.ArgumentParser(add_help=False, description=description)
    general_parser.add_argument("--agent", default="./agents/*",
                                help="Load external python file(s). Useful to register custom agent on-the-fly. Default: %(default)s")

    parser = argparse.ArgumentParser(parents=[general_parser])
    subparsers = parser.add_subparsers(dest="subcommand", title='Available agents to benchmark')

    def _add_general_settings(parser):

        parser.formatter_class = argparse.RawTextHelpFormatter
        general_group = parser.add_argument_group('General settings')

        general_group.add_argument("--envs", metavar="env", nargs="+",
                            choices=tales.envs + tales.tasks + list(tales.benchmark_splits),
                            help="Interactive text environments to evaluate the agent(s)."
                                f" Available:\n{pretty_print_tasks(disable_print=True)}")
        general_group.add_argument("--game-seed", type=int,
                            help="Seed for the game. Default: game-specific one.")
        general_group.add_argument("--nb-steps", type=int, default=100,
                            help="Maximum number of steps per game.")
        general_group.add_argument("--admissible-commands", action="store_true",
                            help="Enable admissible commands.")
        general_group.add_argument("--disable-state-commands", action="store_true",
                            help="Replace save/load/restore/undo actions with an invalid command.")

        general_group.add_argument("--log-dir", default="logs",
                            help="Folder where to save verbose log information.")
        general_group.add_argument("--work-dir",
                            help="Working directory used by the game interpreter.")

        general_group.add_argument("--wandb", action="store_true",
                            help="Log to wandb")
        general_group.add_argument("-ff", "--force-all", action="store_true",
                            help="Force overwriting existing log files.")
        general_group.add_argument("-f", "--force-failed", action="store_true",
                            help="Force overwriting only log files that have failed.")
        general_group.add_argument("--debug", action="store_true",
                            help="Debug mode.")
        general_group.add_argument("--continue-from", dest="continue_from",
                            nargs="?", const="auto",
                            help="Continue from a previous wandb run. "
                                 "Pass a run ID to replay a specific run, or use without a value to auto-find "
                                 "a matching run based on the current config (game, agent, seed).")

        subgroup = general_group.add_mutually_exclusive_group()
        subgroup.add_argument(
            "-v", "--verbose", dest="logging_level",
            action="store_const", const=logging.INFO, default=logging.CRITICAL,
            help="Display actions taken.",
        )
        subgroup.add_argument(
            "-vv", "--very-verbose", dest="logging_level",
            action="store_const", const=logging.DEBUG, default=logging.CRITICAL,
            help="Display actions and game observations.",
        )
        subgroup.add_argument(
            "--logging-level", dest="logging_level", default=logging.CRITICAL,
            choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
            help="Set a specific logging level",
        )

    _add_general_settings(parser)

    agent_parsers = []
    for challenge_name, (desc, _, add_agent_arguments) in tales.agent.AGENTS.items():
        agent_parser = subparsers.add_parser(challenge_name, help=desc)
        add_agent_arguments(agent_parser)
        _add_general_settings(agent_parser)
        agent_parsers.append(agent_parser)

    return parser.parse_args()
    # fmt: on


def main():
    _maybe_load_agent_module()
    args = parse_args()

    if args.subcommand is None:
        print("Need to specify which type of agent to benchmark.")
        exit_listing_agents(args.subcommand)

    # Instanciate the agent.
    _, Agent, _ = tales.agent.AGENTS[args.subcommand]
    agent = Agent(**vars(args))
    agent.new = partial(Agent, **vars(args))

    # Create logging directory.
    args.log_dir = os.path.abspath(os.path.expanduser(args.log_dir))
    args.log_dir = pjoin(args.log_dir, f"tales_{agent.uid.replace('/', '-')}")
    os.makedirs(args.log_dir, exist_ok=True)
    setup_logging(args)
    log.critical(
        colored(f"Logs will be saved in {os.path.abspath(args.log_dir)}", "magenta")
    )

    if args.wandb:
        os.environ["WANDB_MODE"] = "online"
        os.environ.pop("WANDB_RUN_ID", None)

    if args.work_dir:
        args.work_dir = os.path.abspath(os.path.expanduser(args.work_dir))
        os.makedirs(args.work_dir, exist_ok=True)
        os.environ["WANDB_DISABLE_GIT"] = "true"
        os.chdir(args.work_dir)

    args.envs = args.envs or tales.envs
    args.envs = [  # Expand tasks into their respective environments.
        env
        for task in args.envs
        for env in (
            tales.envs_per_task[task]
            if task in tales.tasks
            else tales.benchmark_splits.get(task, [task])
        )
    ]

    benchmark(agent, args)


if __name__ == "__main__":
    main()
