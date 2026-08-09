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
        "inventory_object": 45,
    }
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
            bird_hash = None
            final_observation = ""
            for command in commands:
                final_observation, _, _, _ = env.step(command)
                if command == "take bird":
                    bird_hash = env.get_world_state_hash()

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
            if inventory_objects != [expected["inventory_object"]]:
                raise AssertionError(
                    f"{game}: expected inventory object "
                    f"{expected['inventory_object']}, got {inventory_objects}"
                )
            if world[expected["inventory_object"]].name != "bird":
                raise AssertionError(f"{game}: cleaned bird object name is missing")
            if bird_hash is None or bird_hash == initial_hash:
                raise AssertionError(
                    f"{game}: taking the bird did not change world state"
                )
            if "You have made a friend for life." not in final_observation:
                raise AssertionError(f"{game}: expected victory text is missing")
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
