"""
Test for issue 51:
https://github.com/Data-Centric-AI-Community/data-profiling/issues/51
"""
import numpy as np
import pandas as pd

from data_profiling import ProfileReport


def test_issue51(tmp_path):
    # Categorical has empty ('') value
    rng = np.random.default_rng(51)
    n = 100
    buggy = pd.DataFrame(
        {
            "id": np.arange(n),
            "category": pd.Categorical(rng.choice(["", "a", "b", "c"], n)),
            "label": rng.choice(["", "yes", "no"], n),
            "value": rng.normal(size=n),
        }
    )
    file_name = tmp_path / "buggy1.pkl"
    buggy.to_pickle(file_name)

    df = pd.read_pickle(str(file_name))

    report = ProfileReport(
        df, title="YData Profiling Report", progress_bar=False, explorative=True
    )
    assert (
        "<title>YData Profiling Report</title>" in report.to_html()
    ), "Profile report should be generated."


def test_issue51_similar():
    df = pd.DataFrame(
        {
            "test": ["", "hoi", None],
            "blest": [None, "", "geert"],
            "bert": ["snor", "", None],
        }
    )

    report = ProfileReport(
        df, title="YData Profiling Report", progress_bar=False, explorative=True
    )
    report.config.vars.num.low_categorical_threshold = 0
    # FIXME: assert correlation values (report.description_set["correlations"])

    assert (
        "<title>YData Profiling Report</title>" in report.to_html()
    ), "Profile report should be generated."


def test_issue51_empty():
    df = pd.DataFrame(
        {
            "test": ["", "", "", "", ""],
            "blest": ["", "", "", "", ""],
            "bert": ["", "", "", "", ""],
        }
    )

    report = ProfileReport(
        df,
        title="YData Profiling Report",
        progress_bar=False,
        explorative=True,
    )
    report.config.vars.num.low_categorical_threshold = 0

    assert (
        "cramers" not in report.get_description().correlations
        or (
            report.get_description().correlations["cramers"].values == np.ones((3, 3))
        ).all()
    )


def test_issue51_identical():
    df = pd.DataFrame(
        {
            "test": ["v1", "v1", "v1"],
            "blest": ["v1", "v1", "v1"],
            "bert": ["v1", "v1", "v1"],
        }
    )

    report = ProfileReport(
        df, title="YData Profiling Report", progress_bar=False, explorative=True
    )
    report.config.vars.num.low_categorical_threshold = 0
    # this should not return any correlation value as the variables are identical constants
    assert report.get_description().correlations == {}
