"""Validate a fully supported Jericho game through its canonical walkthrough."""

import argparse
import hashlib
import json
from importlib.resources import files as importlib_files
from pathlib import Path

import gymnasium as gym
from jericho import FrotzEnv

import tales  # noqa: F401 - importing registers TALE Suite environments.
from tales.jericho import jericho_data


EXPECTED = {
    "kiiwii": {
        "commands": 7,
        "moves": 3,
        "score": 1,
        "max_score": 1,
        "player_object": 42,
        "world_objects": 45,
        "inventory_objects": [45],
        "named_objects": {45: "bird"},
        "state_change_command": "take bird",
        "victory_text": "You have made a friend for life.",
        "loss_commands": None,
    },
    "stars": {
        "commands": 1,
        "moves": 1,
        "score": 1,
        "max_score": 1,
        "player_object": 20,
        "world_objects": 38,
        "inventory_objects": [],
        "named_objects": {20: "(self object)", 29: "Bedroom"},
        "state_change_command": None,
        "victory_text": "Christmas has won",
        "loss_commands": ["down", "out", "attack zrblm"],
        "loss_moves": 3,
        "loss_text": "You have died",
        "loss_world_change": True,
    },
    "lily": {
        "commands": 2,
        "moves": 1,
        "score": 0,
        "max_score": 10,
        "player_object": 31,
        "world_objects": 244,
        "inventory_objects": None,
        "named_objects": {30: "Bathroom", 31: "you"},
        "state_change_command": None,
        "victory_text": "I have won",
        "loss_commands": None,
    },
    "internal": {
        "commands": 2,
        "moves": 1,
        "score": 1,
        "max_score": 1,
        "player_object": 20,
        "world_objects": 333,
        "inventory_objects": None,
        "named_objects": {20: "you", 53: "Township of Sebastian"},
        "state_change_command": None,
        "victory_text": "You have won!",
        "loss_commands": None,
    },
    "noroom": {
        "commands": 2,
        "moves": 1,
        "score": 0,
        "max_score": 1,
        "player_object": 20,
        "world_objects": 37,
        "inventory_objects": None,
        "named_objects": {20: "you", 22: "Darkness"},
        "state_change_command": None,
        "victory_text": "You have won",
        "loss_commands": None,
    },
    "pancakedetectives": {
        "commands": 1,
        "moves": 1,
        "score": 1,
        "max_score": 1,
        "player_object": 41,
        "world_objects": 58,
        "inventory_objects": None,
        "named_objects": {41: "you", 42: "Kitchen"},
        "state_change_command": None,
        "victory_text": "Ah-ha! Caught blue-mouthed!",
        "loss_commands": None,
    },
    "banana": {
        "commands": 2,
        "moves": 1,
        "score": -1,
        "max_score": 10,
        "player_object": 20,
        "world_objects": 57,
        "inventory_objects": None,
        "named_objects": {20: "you", 27: "Happy Parrot Bar"},
        "state_change_command": None,
        "victory_text": "won the drinking contest",
        "loss_commands": None,
    },
    "mrp": {
        "commands": 2,
        "moves": 1,
        "score": 1,
        "max_score": 1,
        "player_object": 20,
        "world_objects": 183,
        "inventory_objects": None,
        "named_objects": {20: "you", 33: "bed"},
        "state_change_command": None,
        "victory_text": "You have won",
        "loss_commands": None,
    },
    "putpbaa": {
        "commands": 2,
        "moves": 1,
        "score": 1,
        "max_score": 1,
        "player_object": 20,
        "world_objects": 27,
        "inventory_objects": None,
        "named_objects": {20: "you", 25: "The Town Square"},
        "state_change_command": None,
        "victory_text": "You win",
        "loss_commands": None,
    },
    "annoy": {
        "commands": 1,
        "moves": 1,
        "score": 1,
        "max_score": 1,
        "player_object": 20,
        "world_objects": 28,
        "inventory_objects": None,
        "named_objects": {20: "you", 24: "West End"},
        "state_change_command": None,
        "victory_text": "You have won",
        "loss_commands": None,
    },
    "theemptyroom": {
        "commands": 2,
        "moves": 2,
        "score": 1,
        "max_score": 1,
        "player_object": 41,
        "world_objects": 85,
        "inventory_objects": None,
        "named_objects": {41: "you", 44: "White Room"},
        "state_change_command": None,
        "victory_text": "You win",
        "loss_commands": None,
    },
    "paranoia": {
        "commands": 2,
        "moves": 1,
        "score": 1,
        "max_score": 1,
        "player_object": 41,
        "world_objects": 64,
        "inventory_objects": None,
        "named_objects": {41: "you", 42: "Playroom"},
        "state_change_command": None,
        "victory_text": "You have won",
        "loss_commands": None,
    },
    "ludite": {
        "commands": 2,
        "moves": 1,
        "score": 1,
        "max_score": 1,
        "player_object": 20,
        "world_objects": 31,
        "inventory_objects": None,
        "named_objects": {20: "you", 25: "The Oven"},
        "state_change_command": None,
        "victory_text": "You have won",
        "loss_commands": None,
    },
    "minimalist": {
        "commands": 2,
        "moves": 1,
        "score": 1,
        "max_score": 1,
        "player_object": 41,
        "world_objects": 42,
        "inventory_objects": None,
        "named_objects": {41: "you", 42: "Minimalist prompt"},
        "state_change_command": None,
        "victory_text": "You have won",
        "loss_commands": None,
    },
    "passthemilk": {
        "commands": 1,
        "moves": 1,
        "score": 1,
        "max_score": 1,
        "player_object": 46,
        "world_objects": 59,
        "inventory_objects": None,
        "named_objects": {45: "chair", 46: "you"},
        "state_change_command": None,
        "victory_text": "You passed the milk.",
        "loss_commands": None,
    },
    "forms": {
        "commands": 2,
        "moves": 1,
        "score": 1,
        "max_score": 1,
        "player_object": 20,
        "world_objects": 178,
        "inventory_objects": None,
        "named_objects": {20: "you", 30: "In Your Room"},
        "state_change_command": None,
        "victory_text": "You have won",
        "loss_commands": None,
    },
    "i0": {
        "commands": 2,
        "moves": 1,
        "score": 1,
        "max_score": 1,
        "player_object": 29,
        "world_objects": 309,
        "inventory_objects": None,
        "named_objects": {29: "you", 43: "In your car"},
        "state_change_command": None,
        "victory_text": "You have won",
        "loss_commands": None,
    },
    "peacock": {
        "commands": 1,
        "moves": 1,
        "score": 1,
        "max_score": 1,
        "player_object": 20,
        "world_objects": 138,
        "inventory_objects": None,
        "named_objects": {20: "you", 86: "Peacock Chamber"},
        "state_change_command": "enter opening",
        "victory_text": "You have won",
        "loss_commands": None,
    },
    "putpbad": {
        "commands": 1,
        "moves": 1,
        "score": 1,
        "max_score": 1,
        "player_object": 41,
        "world_objects": 44,
        "inventory_objects": None,
        "named_objects": {41: "you", 42: "Lowell Prison yard"},
        "state_change_command": None,
        "victory_text": "escaped Lowell Prison dead in a pine box",
        "loss_commands": ["take box"],
        "loss_moves": 1,
        "loss_text": "You have died",
        "loss_world_change": False,
    },
    "service": {
        "commands": 2,
        "moves": 1,
        "score": 42,
        "max_score": 42,
        "player_object": 20,
        "world_objects": 28,
        "inventory_objects": None,
        "named_objects": {20: "you", 26: "Chinese Restaurant"},
        "state_change_command": None,
        "victory_text": "You have won",
        "loss_commands": None,
    },
    "spot": {
        "commands": 2,
        "moves": 1,
        "score": 1000,
        "max_score": 1000,
        "player_object": 20,
        "world_objects": 25,
        "inventory_objects": None,
        "named_objects": {20: "you", 24: "In a room with the spot"},
        "state_change_command": None,
        "victory_text": "You have won",
        "loss_commands": None,
    },
    "dayinlife": {
        "commands": 1,
        "moves": 1,
        "score": 0,
        "max_score": 57,
        "player_object": 45,
        "world_objects": 70,
        "inventory_objects": None,
        "named_objects": {45: "you", 46: "Parking lot"},
        "state_change_command": None,
        "victory_text": "YOU WON! Way to clean up. Victory is yours!",
        "loss_commands": None,
    },
}

with (importlib_files("jericho") / "froggy_bindings.json").open() as manifest_file:
    for key, game in json.load(manifest_file).items():
        expected = game["validation"]
        expected["named_objects"] = {
            int(obj_num): name
            for obj_num, name in expected["named_objects"].items()
        }
        if not game["native"].get("victory_moves"):
            expected["victory_text"] = game["native"]["victory_text"]
        EXPECTED[key] = expected


def validate(game: str) -> None:
    game_info = jericho_data.GAMES_INFOS[game]

    game_path = Path(jericho_data.get_game(game))
    digest = hashlib.md5(game_path.read_bytes()).hexdigest()
    if digest != game_info["md5"]:
        raise AssertionError(
            f"{game}: expected MD5 {game_info['md5']}, downloaded {digest}"
        )

    expected = EXPECTED[game]
    env = FrotzEnv(str(game_path))
    try:
        if not env.is_fully_supported:
            raise AssertionError(f"{game}: Jericho did not load the native adapter")

        commands = env.get_walkthrough()
        if len(commands) != expected["commands"]:
            raise AssertionError(
                f"{game}: expected {expected['commands']} commands, got {len(commands)}"
            )

        for replay in range(2):
            env.reset()
            initial_hash = env.get_world_state_hash()
            changed_hash = None
            final_observation = ""
            for command in commands:
                final_observation, _, _, _ = env.step(command)
                if command == expected["state_change_command"]:
                    changed_hash = env.get_world_state_hash()

            if not env.victory() or env.game_over():
                raise AssertionError(
                    f"{game}: replay {replay + 1} is not a clean victory"
                )
            if env.get_score() != expected["score"]:
                raise AssertionError(f"{game}: incorrect final score")
            if env.get_max_score() != expected["max_score"]:
                raise AssertionError(f"{game}: incorrect maximum score")
            if env.get_moves() != expected["moves"]:
                raise AssertionError(
                    f"{game}: expected {expected['moves']} moves, got {env.get_moves()}"
                )
            if env.player_obj_num != expected["player_object"]:
                raise AssertionError(f"{game}: incorrect player object")
            if len(env.get_world_objects()) - 1 != expected["world_objects"]:
                raise AssertionError(f"{game}: incorrect world object count")

            world = env.get_world_objects(clean=True)
            inventory_objects = [obj.num for obj in env.get_inventory()]
            if (
                expected["inventory_objects"] is not None
                and inventory_objects != expected["inventory_objects"]
            ):
                raise AssertionError(
                    f"{game}: expected inventory objects "
                    f"{expected['inventory_objects']}, got {inventory_objects}"
                )
            for obj_num, name in expected["named_objects"].items():
                if world[obj_num].name != name:
                    raise AssertionError(
                        f"{game}: object {obj_num} should be named {name!r}"
                    )
            if (
                expected["state_change_command"] is not None
                and (changed_hash is None or changed_hash == initial_hash)
            ):
                raise AssertionError(
                    f"{game}: {expected['state_change_command']!r} did not "
                    "change world state"
                )
            normalized_observation = " ".join(final_observation.split())
            normalized_victory_text = " ".join(expected["victory_text"].split())
            if normalized_victory_text not in normalized_observation:
                raise AssertionError(f"{game}: expected victory text is missing")

        if expected["loss_commands"] is not None:
            env.reset()
            initial_hash = env.get_world_state_hash()
            for command in expected["loss_commands"]:
                loss_observation, _, _, _ = env.step(command)
            if env.victory() or not env.game_over():
                raise AssertionError(f"{game}: authored loss was not detected")
            if env.get_score() != 0:
                raise AssertionError(f"{game}: loss should not award score")
            if env.get_moves() != expected["loss_moves"]:
                raise AssertionError(f"{game}: incorrect move count after loss")
            if (
                expected["loss_world_change"]
                and env.get_world_state_hash() == initial_hash
            ):
                raise AssertionError(f"{game}: loss path did not change world state")
            if expected["loss_text"] not in loss_observation:
                raise AssertionError(f"{game}: expected loss text is missing")
    finally:
        env.close()

    tale_observations = []
    for replay in range(2):
        tale_env = gym.make(
            f"tales/JerichoEnv{game.title()}-v0",
            disable_env_checker=True,
            admissible_commands=False,
        )
        try:
            tale_env.reset(seed=1)
            final_observation = ""
            final_info = {}
            for command in commands:
                final_observation, _, _, final_info = tale_env.step(command)
            if not final_info["won"] or final_info["lost"]:
                raise AssertionError(
                    f"{game}: TALE replay {replay + 1} is not a clean victory"
                )
            if final_info["score"] != expected["score"]:
                raise AssertionError(f"{game}: incorrect TALE final score")
            if final_info["max_score"] != expected["max_score"]:
                raise AssertionError(f"{game}: incorrect TALE maximum score")
            if final_info["moves"] != expected["moves"]:
                raise AssertionError(f"{game}: incorrect TALE move count")
            tale_observations.append(final_observation)
        finally:
            tale_env.close()
    if tale_observations[0] != tale_observations[1]:
        raise AssertionError(f"{game}: TALE replays were not deterministic")

    print(
        f"{game}: validated {len(commands)} commands, "
        f"{expected['world_objects']} objects, score {expected['score']}/"
        f"{expected['max_score']}"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("game", choices=sorted(EXPECTED))
    args = parser.parse_args()
    validate(args.game)


if __name__ == "__main__":
    main()
