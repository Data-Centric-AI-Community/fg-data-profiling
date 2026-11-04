"""Reference series for the default typeset tests.

Each entry exercises one dtype or value pattern. The expected type of every
series is listed in ``test_typeset_default.py``.
"""
import datetime
import os
import uuid
from collections import namedtuple
from ipaddress import IPv4Address, IPv6Address
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Dict
from urllib.parse import urlparse

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
IMAGES = REPO_ROOT / "docs" / "_static" / "img"

EmailAddress = namedtuple("EmailAddress", ["local", "domain"])


def _python_sequences() -> Dict[str, list]:
    py_files = [
        REPO_ROOT / "tests" / "unit" / "test_utils.py",
        REPO_ROOT / "tests" / "unit" / "__init__.py",
        REPO_ROOT / "tests" / "conftest.py",
    ]
    images = [
        IMAGES / "cli.png",
        IMAGES / "warnings_section.png",
        IMAGES / "univariate_profiling.png",
    ]
    return {
        "int_series": [4, 8, 15],
        "int_range": list(range(12)),
        "int_series_boolean": [0, 1, 1, 0, 1, 0, 0, 1, 1, 1, 0, 1],
        "float_series": [0.5, 2.25, 7.0],
        "float_series2": [3.0, 6.0, 9.0, 12.0],
        "string_series": ["Ada Lovelace", "Grace Hopper"],
        "string_unicode_series": ["olá", "mundo", "café"],
        "string_num": ["4.0", "5.0", "6.0"],
        "string_flt": ["0.5", "12.75", "8.25"],
        "string_bool_nan": ["False", "True", None],
        "str_url": [
            "https://www.python.org/downloads/",
            "https://github.com/Data-Centric-AI-Community",
        ],
        "path_series_windows_str": [
            r"C:\\data\\input.csv",
            r"C:\\data\\output.csv",
        ],
        "path_series_linux_str": ["/data/input.csv", "/data/output.csv"],
        "str_int_leading_zeros": ["0042", "17"],
        "str_float_non_leading_zeros": ["0.0", "0.25", "0"],
        "str_int_zeros": ["0.00", "0.0", "0", "3"],
        "bool_series": [False, True],
        "bool_nan_series": [False, True, None],
        "str_complex": ["(1+2j)", "(3-1j)", "(0+5j)"],
        "str_complex_nan": ["(1+2j)", "(3-1j)", "(0+5j)", "NaN"],
        "complex_series_py": [complex(1, 1), complex(0, 2), complex(-2, 3)],
        "complex_series_py_float": [complex(2, 0), complex(4, 0), complex(-1, 0)],
        "string_date": ["1999-12-31", "15/6/2021"],
        "timestamp_string_series": ["2000-01-01", "31/12/2010"],
        "py_datetime_str": ["2020-02-29 10:30:00", "2021-03-01 23:59:59"],
        "date": [
            datetime.date(2020, 2, 29),
            datetime.date(2021, 3, 1),
            datetime.date(2022, 4, 2),
        ],
        "time": [
            datetime.time(7, 0, 0),
            datetime.time(12, 30, 15),
            datetime.time(23, 59, 59),
        ],
        "path_series_linux": [
            PurePosixPath("/data/input.csv"),
            PurePosixPath("/data/output.csv"),
        ],
        "path_series_linux_missing": [
            PurePosixPath("/data/input.csv"),
            None,
            PurePosixPath("/data/output.csv"),
        ],
        "path_series_windows": [
            PureWindowsPath("C:\\data\\input.csv"),
            PureWindowsPath("C:\\data\\output.csv"),
        ],
        "url_series": [
            urlparse("https://www.python.org/downloads/"),
            urlparse("https://github.com/Data-Centric-AI-Community"),
        ],
        "url_none_series": [
            urlparse("https://www.python.org/downloads/"),
            None,
            urlparse("https://github.com/Data-Centric-AI-Community"),
        ],
        "uuid_series": [
            uuid.UUID("1b4e28ba-2fa1-11d2-883f-0016d3cca427"),
            uuid.UUID("6fa459ea-ee8a-3ca4-894e-db77e160355e"),
            uuid.UUID("00000000-0000-0000-0000-000000000000"),
        ],
        "uuid_series_missing": [
            uuid.UUID("1b4e28ba-2fa1-11d2-883f-0016d3cca427"),
            None,
            uuid.UUID("6fa459ea-ee8a-3ca4-894e-db77e160355e"),
        ],
        "uuid_series_str": [
            "1b4e28ba-2fa1-11d2-883f-0016d3cca427",
            "6fa459ea-ee8a-3ca4-894e-db77e160355e",
            "00000000-0000-0000-0000-000000000000",
        ],
        "mixed_list[str,int]": [[1, "a"], [2, "b"], [3, "c"]],
        "mixed_dict": [{"k": "v"}, {"x": "y"}, {"p": "q"}],
        "callable": [os.getcwd, os.listdir, len],
        "module": [os, uuid],
        "textual_float": ["2.5", "7"],
        "textual_float_nan": ["2.5", "7", "NAN"],
        "mixed_integer": ["x", 7],
        "mixed_list": [[True], [False], [True]],
        "ip_str": ["192.168.0.1", "192.168.0.1"],
        "empty": [],
        "ip": [IPv4Address("192.168.0.1"), IPv4Address("192.168.0.1")],
        "ip_missing": [IPv4Address("192.168.0.1"), None, IPv4Address("10.0.0.1")],
        "ip_mixed_v4andv6": [IPv6Address("::1"), IPv4Address("192.168.0.1")],
        "file_test_py": py_files,
        "file_mixed_ext": [
            REPO_ROOT / "pyproject.toml",
            REPO_ROOT / "README.md",
            REPO_ROOT / "tests" / "conftest.py",
        ],
        "file_test_py_missing": [py_files[0], None, py_files[1], None, py_files[2]],
        "image_png": images,
        "image_png_missing": [images[0], None, images[1], None, images[2]],
        "email_address": [
            EmailAddress("ada", "example.com"),
            EmailAddress("grace", "example.org"),
        ],
        "email_address_missing": [
            EmailAddress("ada", "example.com"),
            None,
            EmailAddress("grace", "example.org"),
        ],
        "email_address_str": ["ada@example.com", "grace@example.org"],
    }


def _pandas_sequences() -> Dict[str, pd.Series]:
    return {
        "float_series6": pd.Series([np.nan, 2.5], dtype=np.single),
        "bool_series2": pd.Series([False, True, True, False], dtype=bool),
        "nullable_bool_series": pd.Series([False, True, None], dtype="boolean"),
        "int_str_range": pd.Series(range(20)).astype("str"),
        "Int64_int_series": pd.Series([4, 8, 15], dtype="Int64"),
        "Int64_int_nan_series": pd.Series([4, 8, 15, np.nan], dtype="Int64"),
        "pd_uint32": pd.Series([4, 8, 15, 16], dtype="UInt32"),
        "categorical_int_series": pd.Series([4, 8, 15], dtype="category"),
        "categorical_char": pd.Series(
            pd.Categorical(["x", "y", "z", "z", "y", "x"], ordered=False)
        ),
        "categorical_float_series": pd.Series([0.5, 1.5, 2.25], dtype="category"),
        "categorical_string_series": pd.Series(["Lisbon", "Porto"], dtype="category"),
        "ordinal": pd.Series(
            pd.Categorical(
                ["low", "mid", "high", "high", "mid", "low"],
                categories=["low", "mid", "high"],
                ordered=True,
            )
        ),
        "timestamp_series": pd.Series(
            pd.to_datetime(
                [datetime.datetime(2020, 2, 29, 10, 30), datetime.datetime(2021, 3, 1)]
            )
        ),
        "timestamp_series_nat": pd.Series(
            pd.to_datetime(
                [
                    datetime.datetime(2020, 2, 29),
                    datetime.datetime(2021, 3, 1, 23, 59),
                    pd.NaT,
                ]
            )
        ),
        "date_series_nat": pd.Series(
            pd.to_datetime(
                [datetime.datetime(2020, 2, 29), datetime.datetime(2021, 3, 1), pd.NaT]
            )
        ),
        "timestamp_aware_series": pd.Series(
            pd.date_range(
                start="2024-05-18 12:00:00", periods=3, freq="h", tz="Europe/Lisbon"
            )
        ),
        "datetime": pd.Series(
            pd.to_datetime(
                [
                    datetime.date(2020, 2, 29),
                    datetime.date(2021, 3, 1),
                    datetime.date(2022, 4, 2),
                ]
            )
        ),
        "timedelta_series": pd.Series([datetime.timedelta(hours=h) for h in range(3)]),
        "timedelta_series_nat": pd.Series(
            [datetime.timedelta(hours=h) for h in range(3)] + [pd.NaT]
        ),
        "timedelta_negative": pd.Series(
            [
                datetime.timedelta(days=2, hours=1),
                datetime.timedelta(days=-1, hours=3, minutes=15),
            ]
        ),
        "empty_float": pd.Series([], dtype=float),
        "empty_int64": pd.Series([], dtype="Int64"),
        "empty_object": pd.Series([], dtype="object"),
        "empty_bool": pd.Series([], dtype=bool),
        "float_series4": pd.Series([2, 4, 6.5, 8], dtype=np.float64),
        "all_null_none": pd.Series([None, None]),
        "all_null_nan": pd.Series([np.nan, np.nan]),
        "all_null_nat": pd.Series([pd.NaT, pd.NaT]),
        "all_null_empty_str": pd.Series(["", ""]),
        "string_dtype_series": pd.Series(
            ["Ada Lovelace", "Grace Hopper"], dtype="string"
        ),
    }


def get_reference_series() -> Dict[str, pd.Series]:
    series = {name: pd.Series(values) for name, values in _python_sequences().items()}
    series.update(_pandas_sequences())
    return series
