import json

import pandas as pd
import pytest

from data_profiling import ProfileReport


@pytest.fixture
def data():
    return pd.DataFrame(data={"col1": [1, 2], "col2": [3, 4]})


def test_json(data):
    report = ProfileReport(data)
    data = json.loads(report.to_json())
    assert data["schema"] == "data-profiling.agent/1"
    assert set(data.keys()) == {
        "schema",
        "detail",
        "dataset",
        "findings",
        "columns",
        "correlations",
        "omitted",
    }
    assert set(data["columns"]) == {"col1", "col2"}


def test_json_legacy(data):
    report = ProfileReport(data)
    with pytest.warns(FutureWarning):
        report_json = report.to_json(legacy=True)
    data = json.loads(report_json)
    assert set(data.keys()) == {
        "analysis",
        "time_index_analysis",
        "correlations",
        "duplicates",
        "alerts",
        "missing",
        "package",
        "sample",
        "scatter",
        "table",
        "variables",
    }


def test_repr(data):
    report = ProfileReport(data)
    assert repr(report) == ""
