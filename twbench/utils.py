import io
import os
import shutil
import zipfile
import requests
import tempfile

from os.path import join as pjoin

from tqdm import tqdm


def mkdirs(dirpath: str) -> str:
    """ Create a directory and all its parents.

    If the folder already exists, its path is returned without raising any exceptions.

    Arguments:
        dirpath: Path where a folder need to be created.

    Returns:
        Path to the (created) folder.
    """
    try:
        os.makedirs(dirpath)
    except FileExistsError:
        pass

    return dirpath


def download(url, dst, desc=None, force=False):
    """ Download a remote file using HTTP get request.

    Args:
        url (str): URL where to get the file.
        dst (str): Destination folder where to save the file.
        force (bool, optional):
            Download again if it exists]. Defaults to False.

    Returns:
        str: Path to the downloaded file.

    Notes:
        This code is inspired by
        https://github.com/huggingface/transformers/blob/v4.0.0/src/transformers/file_utils.py#L1069
    """
    filename = dst_filename or url.split('/')[-1]
    path = pjoin(mkdirs(dst), filename)

    if os.path.isfile(path) and not force:
        return path

    # Download to a temp folder first to avoid corrupting the cache
    # with incomplete downloads.
    temp_dir = mkdirs(pjoin(tempfile.gettempdir(), "twbench"))
    temp_path = pjoin(temp_dir, filename)
    with open(temp_path, 'ab') as temp_file:
        headers = {}
        resume_size = temp_file.tell()
        if resume_size:
            headers['Range'] = f'bytes={resume_size}-'
            headers['x-ms-version'] = "2020-04-08"  # Needed for Range support.

        r = requests.get(url, stream=True, headers=headers)
        if r.headers.get("x-ms-error-code") == "InvalidRange" and r.headers["Content-Range"].rsplit("/", 1)[-1] == str(resume_size):
            shutil.move(temp_path, path)
            return path

        r.raise_for_status()  # Bad request.
        content_length = r.headers.get("Content-Length")
        total = resume_size + int(content_length)
        pbar = tqdm(
            unit="B",
            initial=resume_size,
            unit_scale=True,
            total=total,
            desc=desc or "Downloading {}".format(filename),
        )

        for chunk in r.iter_content(chunk_size=1024):
            if chunk:  # filter out keep-alive new chunks
                pbar.update(len(chunk))
                temp_file.write(chunk)

    shutil.move(temp_path, path)

    pbar.close()
    return path


# def download(link):
#     filename = link.split("/")[-1]
#     response = requests.get(link)
#     data = response.content
#     return data, filename


# def extract_file_from_zip(link):
#     splits = link.split(":")
#     downloadable_link = ":".join(splits[:-1])
#     filename_to_extract = splits[-1]
#     data, filename = download(downloadable_link)
#     zipped_file = zipfile.ZipFile(io.BytesIO(data))
#     data = zipped_file.read(filename_to_extract)
#     return data, filename_to_extract


# def download_game(game_info):
#     if ".zip:" in game_info["link"].lower():
#         data, filename = extract_file_from_zip(game_info["link"])
#     else:
#         data, filename = download(game_info["link"])

#     return data
