"""
Test for issue 377:
https://github.com/Data-Centric-AI-Community/data-profiling/issues/377
"""
import sys

import numpy as np
import pandas as pd
import pytest

from data_profiling import ProfileReport


@pytest.fixture()
def df(tmp_path):
    """Shaped like UCI Bank Marketing ``bank-full.csv``: 17 mixed-type
    columns in a non-alphabetical order, ``;``-separated."""
    rng = np.random.default_rng(377)
    n = 2000
    yes_no = ["yes", "no"]
    data = pd.DataFrame(
        {
            "age": rng.integers(18, 95, n),
            "job": rng.choice(["admin.", "technician", "services", "retired"], n),
            "marital": rng.choice(["married", "single", "divorced"], n),
            "education": rng.choice(["primary", "secondary", "tertiary", "unknown"], n),
            "default": rng.choice(yes_no, n, p=[0.02, 0.98]),
            "balance": rng.integers(-8000, 100000, n),
            "housing": rng.choice(yes_no, n),
            "loan": rng.choice(yes_no, n, p=[0.16, 0.84]),
            "contact": rng.choice(["cellular", "telephone", "unknown"], n),
            "day": rng.integers(1, 32, n),
            "month": rng.choice(["jan", "feb", "mar", "may", "jun", "jul"], n),
            "duration": rng.integers(0, 4900, n),
            "campaign": rng.integers(1, 63, n),
            "pdays": rng.choice([-1, 92, 181, 370], n, p=[0.8, 0.1, 0.05, 0.05]),
            "previous": rng.integers(0, 10, n),
            "poutcome": rng.choice(["unknown", "failure", "success", "other"], n),
            "y": rng.choice(yes_no, n, p=[0.12, 0.88]),
        }
    )
    path = tmp_path / "bank-full.csv"
    data.to_csv(path, sep=";", index=False)
    return pd.read_csv(path, sep=";")


@pytest.mark.skipif(sys.version_info < (3, 6), reason="requires python3.6 or higher")
def test_issue377(df):
    original_order = tuple(df.columns.values)

    profile = ProfileReport(df, sort=None, pool_size=1, progress_bar=False)
    new_order = tuple(profile.get_description().variables.keys())
    assert original_order == new_order
