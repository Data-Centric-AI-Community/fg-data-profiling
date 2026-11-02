"""
Test for issue 1529:
https://github.com/Data-Centric-AI-Community/data-profiling/issues/1529
"""

import json

import pandas as pd
import pytest

from data_profiling import ProfileReport


def test_issue1529():
    previous_dataset = pd.DataFrame(
        data=[(1000, 42), (900, 30), (1500, 40), (1800, 38)],
        columns=["rent_per_month", "total_area"],
    )
    current_dataset = pd.DataFrame(
        data=[(5000, 350), (9000, 600), (5000, 400), (3500, 500), (6000, 600)],
        columns=["rent_per_month", "total_area"],
    )
    previous_dataset_report = ProfileReport(
        previous_dataset, title="Previous dataset report"
    )
    current_dataset_report = ProfileReport(
        current_dataset, title="Current dataset report"
    )
    comparison_report = previous_dataset_report.compare(current_dataset_report)
    json_str = comparison_report.to_json()
    compare_dict = json.loads(json_str)
    assert compare_dict["kind"] == "comparison"
    assert compare_dict["labels"] == [
        "Previous dataset report",
        "Current dataset report",
    ]
    assert [r["dataset"]["title"] for r in compare_dict["reports"]] == [
        "Previous dataset report",
        "Current dataset report",
    ]
    assert compare_dict["delta"][0]["rows"] == [4, 5]

    with pytest.warns(FutureWarning):
        legacy = json.loads(comparison_report.to_json(legacy=True))
    assert (
        legacy["analysis"]["title"]
        == "<em>Comparing</em> Previous dataset report <em>and</em> Current dataset report"
    )
