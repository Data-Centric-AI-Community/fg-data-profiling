import concurrent.futures
from functools import partial
from gzip import decompress
from typing import Tuple

import numpy as np
import pandas as pd
import pytest
import requests

from data_profiling.model.summary import describe_1d


def mock_multiprocess_1d(args, config, summarizer, typeset) -> Tuple[str, dict]:
    """Wrapper to process series in parallel.

    Args:
        args: Tuple containing (column name, series).
        config: Profiling configuration.
        summarizer: The summarizer instance.
        typeset: The data typeset.

    Returns:
        A tuple with column name and its profile description.
    """
    column, series = args
    return column, describe_1d(config, series, summarizer, typeset)


@pytest.mark.network
@pytest.mark.skip("This test is no longer valid")
def test_multiprocessing_describe1d(config, summarizer, typeset):
    """
    This test ensures that parallelized describe1d operations do not cause a ValueError due to
    race conditions when modifying series data in place.
    """

    def download_and_process_data():
        """Downloads and processes the dataset into a Pandas DataFrame."""
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/122.0.0.0 Safari/537.36"
        }
        response = requests.get(
            "https://ndownloader.figshare.com/files/5976042", headers=headers
        )

        response.raise_for_status()  # Ensure successful download

        file = decompress(response.content)
        text = file.decode()
        split_text = [row.split(",") for row in text.split("\n") if row]

        dtype_mapping = [
            ("duration", int),
            ("protocol_type", "S4"),
            ("service", "S11"),
            ("flag", "S6"),
            ("src_bytes", int),
            ("dst_bytes", int),
            ("land", int),
            ("wrong_fragment", int),
            ("urgent", int),
            ("hot", int),
            ("num_failed_logins", int),
            ("logged_in", int),
            ("num_compromised", int),
            ("root_shell", int),
            ("su_attempted", int),
            ("num_root", int),
            ("num_file_creations", int),
            ("num_shells", int),
            ("num_access_files", int),
            ("num_outbound_cmds", int),
            ("is_host_login", int),
            ("is_guest_login", int),
            ("count", int),
            ("srv_count", int),
            ("serror_rate", float),
            ("srv_serror_rate", float),
            ("rerror_rate", float),
            ("srv_rerror_rate", float),
            ("same_srv_rate", float),
            ("diff_srv_rate", float),
            ("srv_diff_host_rate", float),
            ("dst_host_count", int),
            ("dst_host_srv_count", int),
            ("dst_host_same_srv_rate", float),
            ("dst_host_diff_srv_rate", float),
            ("dst_host_same_src_port_rate", float),
            ("dst_host_srv_diff_host_rate", float),
            ("dst_host_serror_rate", float),
            ("dst_host_srv_serror_rate", float),
            ("dst_host_rerror_rate", float),
            ("dst_host_srv_rerror_rate", float),
            ("labels", "S16"),
        ]

        split_text = np.array(split_text, dtype=object)

        # Convert each column to its appropriate type
        for j, (_, col_dtype) in enumerate(dtype_mapping):
            split_text[:, j] = split_text[:, j].astype(col_dtype)

        return pd.DataFrame(split_text)

    def run_multiprocess(config, df):
        """Runs describe1d in parallel using ThreadPoolExecutor."""
        args = [(column, series) for column, series in df.items()]
        process_func = partial(
            mock_multiprocess_1d, config=config, summarizer=summarizer, typeset=typeset
        )

        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            results = list(executor.map(lambda x: process_func(x), args))

        return results

    try:
        df = download_and_process_data()
        assert not df.empty, "Dataset failed to download or process correctly."

        run_multiprocess(config, df)

    except ValueError as ex:
        raise RuntimeError(
            "Parallel describe1d execution raised ValueError unexpectedly!"
        ) from ex
