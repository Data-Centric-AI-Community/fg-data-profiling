"""
Test for issue 147:
https://github.com/Data-Centric-AI-Community/data-profiling/issues/147
"""
import numpy as np
import pandas as pd
import pytest

from data_profiling import ProfileReport


def _userdata(n: int = 1000, seed: int = 147) -> pd.DataFrame:
    """Shaped like Kylo's sample ``userdata*.parquet`` files."""
    rng = np.random.default_rng(seed)
    salary = np.round(rng.uniform(12000, 290000, n), 2)
    salary[rng.random(n) < 0.07] = np.nan
    birth = pd.to_datetime("1955-01-01") + pd.to_timedelta(
        rng.integers(0, 46 * 365, n), unit="D"
    )
    return pd.DataFrame(
        {
            "registration_dttm": pd.Timestamp("2016-02-03")
            + pd.to_timedelta(rng.integers(0, 86400, n), unit="s"),
            "id": np.arange(1, n + 1, dtype=float),
            "first_name": [f"First{i % 300}" for i in range(n)],
            "last_name": [f"Last{i % 700}" for i in range(n)],
            "email": [f"user{i}@example.com" for i in range(n)],
            "gender": rng.choice(["Female", "Male", ""], n, p=[0.47, 0.47, 0.06]),
            "ip_address": [
                ".".join(map(str, rng.integers(1, 255, 4))) for _ in range(n)
            ],
            "cc": [str(rng.integers(10**15, 10**16)) for _ in range(n)],
            "country": rng.choice(["China", "Indonesia", "Russia", "Brazil"], n),
            "birthdate": [f"{d.month}/{d.day}/{d.year}" for d in birth],
            "salary": salary,
            "title": rng.choice(["Engineer", "Analyst", "Nurse", ""], n),
            "comments": rng.choice(["", "great", "1E+02", "\u00e9t\u00e9"], n),
        }
    )


def test_issue147(tmp_path):
    pytest.importorskip("pyarrow")
    file_name = tmp_path / "userdata1.parquet"
    _userdata().to_parquet(file_name, engine="pyarrow")

    df = pd.read_parquet(str(file_name), engine="pyarrow")
    report = ProfileReport(
        df, title="PyArrow with Pandas Parquet Backend", progress_bar=False, pool_size=1
    )
    html = report.to_html()
    assert type(html) == str
    assert "Dataset statistics" in html
