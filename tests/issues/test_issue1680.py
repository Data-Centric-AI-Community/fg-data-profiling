"""
Test for issue 1680.

https://github.com/Data-Centric-AI-Community/fg-data-profiling/issues/1680
"""
import json

import numpy as np
import pandas as pd
import pytest
from typeguard import TypeCheckError

from data_profiling import ProfileReport


@pytest.fixture
def data():
    """Create a dataframe."""
    np.random.seed(42)

    return pd.DataFrame(np.random.rand(7, 3), columns=["a", "b", "c"])


def test_empty_df(data):
    """Check if the data frame is empty."""
    if not len(data):
        raise ValueError("Check the dataframe it is empty")


# Checking excluded fields
def test_excluded_fields(data):
    """Check if there are excluded fields in the json."""
    test_empty_df(data)

    excluded_fields = [
        "value_counts_without_nan",
        "value_counts_index_sorted",
        "n_var",
        "mean",
        "sum",
        "mad",
    ]

    report = ProfileReport(data, excluded_fields=excluded_fields)
    report_json = report.to_json()
    data_json = json.loads(report_json)

    def check_no_key(dict_recursion):
        for key, value in dict_recursion.items():
            if key in excluded_fields:
                description = f"The field: {key} " f"that was excluded is present."
                raise ValueError(description)
            if isinstance(value, dict):
                check_no_key(value)

    check_no_key(data_json)


@pytest.mark.parametrize("invalid_value", [12345, "abcde", {}, True])
def test_invalid_excluded_fields(data, invalid_value):
    """
    Validate excluded_fields types.

    Checks that passing invalid types
    to excluded_fields causes a type error.
    """
    with pytest.raises(TypeCheckError):
        ProfileReport(data, excluded_fields=invalid_value)  # type: ignore # noqa
