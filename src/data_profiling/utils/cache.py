"""Dataset cache utility functions"""
import zipfile
from pathlib import Path

from functools import partial
from requests import get as _get_file

get_file = partial(_get_file, timeout=30)

from data_profiling.utils.paths import get_data_path


def cache_file(file_name: str, url: str) -> Path:
    """Check if file_name already is in the data path, otherwise download it from url.

    Args:
        file_name: the file name
        url: the URL of the dataset

    Returns:
        The relative path to the dataset
    """

    data_path = get_data_path()
    data_path.mkdir(exist_ok=True)

    file_path = data_path / file_name

    # If not exists, download and create file
    if not file_path.exists():
        response = get_file(url, allow_redirects=True)
        response.raise_for_status()

        file_path.write_bytes(response.content)

    return file_path


def cache_zipped_file(file_name: str, url: str) -> Path:
    """Check if file_name already is in the data path, otherwise download it from url.

    Args:
        file_name: the file name
        url: the URL of the dataset

    Returns:
        The relative path to the dataset
    """

    data_path = get_data_path()
    file_path = data_path / file_name

    # If not exists, download and create file
    if not file_path.exists():
        tmp_path = cache_file("tmp.zip", url)

        with zipfile.ZipFile(tmp_path, "r") as zip_file:
            zip_file.extract(file_path.name, data_path)

        tmp_path.unlink()

    return file_path
