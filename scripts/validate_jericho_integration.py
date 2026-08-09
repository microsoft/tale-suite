"""Validate a fully supported Jericho game through its canonical walkthrough."""

import argparse
import hashlib
import json
from importlib.resources import files as importlib_files
from pathlib import Path

from jericho import FrotzEnv

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
}


def validate(game: str) -> None:
    games_path = importlib_files("tales") / "jericho" / "games.json"
    with games_path.open() as games_file:
        game_info = json.load(games_file)[game]

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
            if expected["victory_text"] not in final_observation:
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
            if env.get_world_state_hash() == initial_hash:
                raise AssertionError(f"{game}: loss path did not change world state")
            if expected["loss_text"] not in loss_observation:
                raise AssertionError(f"{game}: expected loss text is missing")
    finally:
        env.close()

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
