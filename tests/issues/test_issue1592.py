"""
Test for issue 1592:
https://github.com/Data-Centric-AI-Community/fg-data-profiling/issues/1592
"""
import pandas as pd

from data_profiling import ProfileReport, compare
from data_profiling.report.structure.variables.render_common import render_common


def _rows(table):
    return [(row["label"], row["count"], row["percentage"], row["n"]) for row in table]


def test_issue1592():
    df1 = pd.DataFrame([1] * 50 + [0] * 50, columns=["point"])
    df2 = pd.DataFrame([1] * 9 + [0] * 1, columns=["point"])
    report1 = ProfileReport(df1, minimal=True)
    report2 = ProfileReport(df2, minimal=True)
    comparison = compare([report1, report2], compute=True)

    summary = comparison.get_description().variables["point"]
    template_variables = render_common(comparison.config, summary)

    # the first table belongs to the first dataset and the second to the second
    # one, each one with its own total number of observations
    assert [_rows(t) for t in template_variables["firstn_expanded"]] == [
        [(0, 50, 0.5, 100), (1, 50, 0.5, 100)],
        [(0, 1, 0.1, 10), (1, 9, 0.9, 10)],
    ]
    # the largest values are listed first
    assert [_rows(t) for t in template_variables["lastn_expanded"]] == [
        [(1, 50, 0.5, 100), (0, 50, 0.5, 100)],
        [(1, 9, 0.9, 10), (0, 1, 0.1, 10)],
    ]


def test_issue1592_single_report():
    df = pd.DataFrame([1] * 9 + [0] * 1, columns=["point"])
    report = ProfileReport(df, minimal=True)

    summary = report.get_description().variables["point"]
    template_variables = render_common(report.config, summary)

    assert [_rows(t) for t in template_variables["lastn_expanded"]] == [
        [(1, 9, 0.9, 10), (0, 1, 0.1, 10)]
    ]
