from dataclasses import asdict

import numpy as np
import pandas as pd
import pytest

from data_profiling import ProfileReport


@pytest.fixture
def tdf(meteorites_csv):
    df = pd.read_csv(meteorites_csv)

    # Note: Pandas does not support dates before 1880, so we ignore these for this analysis
    df["year"] = pd.to_datetime(df["year"], errors="coerce")

    # Example: Constant variable
    df["source"] = "NASA"

    # Example: Boolean variable
    df["boolean"] = np.random.choice([True, False], df.shape[0])

    # Example: Mixed with base types
    df["mixed"] = np.random.choice([1, "A"], df.shape[0])

    # Example: Highly correlated variables
    df["reclat_city"] = df["reclat"] + np.random.normal(scale=5, size=(len(df)))

    # Example: Duplicate observations
    duplicates_to_add = pd.DataFrame(df.iloc[0:10])
    df = pd.concat([df, duplicates_to_add], ignore_index=True)
    return df


def test_modular_description_set(tdf):
    profile = ProfileReport(
        tdf,
        title="Modular test",
        duplicates=None,
        samples={"head": 0, "tail": 0},
        correlations=None,
        interactions=None,
        missing_diagrams={
            "matrix": False,
            "bar": False,
            "heatmap": False,
        },
        pool_size=1,
    )

    description = profile.get_description()
    assert len(asdict(description)) > 0


def test_modular_absent(tdf):
    profile = ProfileReport(
        tdf,
        title="Modular test",
        duplicates={"head": 0},
        samples={"head": 0, "tail": 0},
        interactions=None,
        correlations=None,
        missing_diagrams=None,
    )

    html = profile.to_html()
    assert "Correlations</h1>" not in html
    assert "Duplicate rows</h1>" not in html
    assert "Sample</h1>" not in html
    assert "Missing values</h1>" not in html


def test_modular_present(tdf):
    profile = ProfileReport(
        tdf,
        title="Modular test",
        duplicates={"head": 10},
        samples={"head": 10, "tail": 10},
        interactions={"targets": ["mass (g)"], "continuous": True},
        correlations={
            "pearson": {"calculate": True},
            "spearman": {"calculate": True},
            "kendall": {"calculate": True},
            "phi_k": {"calculate": True},
            "cramers": {"calculate": True},
        },
        missing_diagrams={
            "matrix": True,
            "bar": True,
            "heatmap": True,
        },
        pool_size=1,
    )

    html = profile.to_html()
    assert "Correlations</h1>" in html
    assert "Duplicate rows</h1>" in html
    assert "Sample</h1>" in html
    assert "Missing values</h1>" in html
