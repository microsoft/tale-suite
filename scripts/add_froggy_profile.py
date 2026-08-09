#!/usr/bin/env python3
"""Add one analyzed Froggy game to the manifest-backed Jericho registry."""

import argparse
import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "vendor" / "jericho" / "jericho" / "froggy_bindings.json"
GENERATOR = (
    ROOT
    / "vendor"
    / "jericho"
    / "scripts"
    / "generate_froggy_registry.py"
)
ARCHIVE_URL = (
    "https://raw.githubusercontent.com/BYU-PCCL/z-machine-games/"
    "bfd7544aa1f50549fc327bd4eb3bc8144bfd80ea/"
    "the-large-game-collection.zip"
)


def convert(profile):
    player_object = profile["player_object"]
    location_object = profile["location_object"]
    location_name = profile["location_name"]
    if not location_name or len(location_name) >= 64:
        location_name = profile["title"]
    named_objects = {str(player_object): profile["player_display_name"]}
    if location_object and location_object != player_object:
        named_objects[str(location_object)] = location_name

    return {
        "key": profile["key"],
        "title": profile["title"],
        "rom": profile["rom_filename"],
        "md5": profile["md5"],
        "seed": 1,
        "walkthrough": profile["walkthrough"],
        "native": {
            "victory_text": profile["victory_substring"],
            "loss_text": profile["loss_substring"],
            "player_object": player_object,
            "moves_address": profile["moves_ram_address"],
            "moves_base": profile["moves_base"],
            "score_address": profile["score_ram_address"],
            "score_base": profile["score_base"],
            "max_score": profile["max_score"],
            "world_objects": profile["world_object_count"],
            "player_name": profile["player_display_name"],
            "location_object": location_object,
            "location_name": location_name,
        },
        "tale": {
            "filename": profile["rom_filename"],
            "info": profile["title"],
            "md5": profile["md5"],
            "archive_url": ARCHIVE_URL,
            "archive_filename": "the-large-game-collection.zip",
            "archive_member": profile["archive_member_path"],
        },
        "validation": {
            "commands": len(profile["walkthrough"]),
            "moves": profile["final_moves"],
            "score": profile["final_score"],
            "max_score": profile["max_score"],
            "player_object": player_object,
            "world_objects": profile["world_object_count"],
            "inventory_objects": None,
            "named_objects": named_objects,
            "state_change_command": None,
            "victory_text": profile["terminal_prose_substring"],
            "loss_commands": None,
        },
        "caveats": profile.get("caveats", ""),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("profiles", type=Path)
    parser.add_argument("key")
    args = parser.parse_args()

    analyzed = json.loads(args.profiles.read_text())
    matching = [
        profile for profile in analyzed["selected"] if profile["key"] == args.key
    ]
    if len(matching) != 1:
        raise ValueError(f"Expected one analyzed profile for {args.key!r}")

    games = json.loads(MANIFEST.read_text())
    if args.key in games:
        raise ValueError(f"{args.key!r} is already integrated")
    if any(game["md5"] == matching[0]["md5"] for game in games.values()):
        raise ValueError(f"MD5 for {args.key!r} is already integrated")

    games[args.key] = convert(matching[0])
    MANIFEST.write_text(json.dumps(games, indent=2) + "\n")
    subprocess.check_call([sys.executable, str(GENERATOR)])


if __name__ == "__main__":
    main()
