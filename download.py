import io
import os
import json
import shutil
import zipfile
import tempfile
import argparse
import urllib.request
import subprocess
from os.path import join as pjoin


def download(link):
    filename = link.split("/")[-1]
    with urllib.request.urlopen(link) as response:
        data = response.read()

    return data, filename


def extract_file_from_zip(link):
    splits = link.split(":")
    downloadable_link = ":".join(splits[:-1])
    filename_to_extract = splits[-1]
    data, filename = download(downloadable_link)
    zipped_file = zipfile.ZipFile(io.BytesIO(data))
    data = zipped_file.read(filename_to_extract)
    return data, filename_to_extract


def extract_zfile_from_zblorb(data):
    temp_dir = tempfile.mkdtemp()
    try:
        src = pjoin(temp_dir, "src.zblorb")
        with open(src, "wb") as f:
            f.write(data)

        dest = pjoin(temp_dir, "dest.z8")
        cmd = ["python", "blorbtool.py", src, "export", "ZCOD", dest]
        subprocess.check_call(cmd)
        with open(dest, "rb") as f:
            new_data = f.read()

    finally:
        shutil.rmtree(temp_dir)

    return new_data


def download_game(game_info):
    if ".zip:" in game_info["link"].lower():
        data, filename = extract_file_from_zip(game_info["link"])
    else:
        data, filename = download(game_info["link"])

    if filename.endswith(".zblorb"):
        data = extract_zfile_from_zblorb(data)

    return data


def parse_args():
    parser = argparse.ArgumentParser(description="Retrieves all games from the curated list.")
    parser.add_argument("--out", default="./games",
                        help="Output directory where to save the games. Default: '%(default)s'.")
    return parser.parse_args()


def main():
    args = parse_args()

    with open("games.json") as f:
        game_infos = json.load(f)

    if not os.path.isdir(args.out):
        os.makedirs(args.out)

    print("Downloading...")

    # If available, show progress bar.
    try:
        import tqdm
        pbar = tqdm.tqdm(game_infos.items(), leave=False)
        using_tqdm = True
    except ImportError:
        pbar = game_infos.items()
        using_tqdm = False

    nb_downloaded = 0
    for name, game_info in pbar:
        if using_tqdm:
            pbar.set_postfix_str(game_info["filename"])

        game_file = pjoin(args.out, game_info["filename"])
        if not os.path.isfile(game_file):
            nb_downloaded += 1
            data = download_game(game_info)
            with open(game_file, "wb") as f:
                f.write(data)

    print("{} games ({} new)".format(len(game_infos), nb_downloaded))


if __name__ == "__main__":
    main()
