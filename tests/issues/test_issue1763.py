"""
Test for issue 1763:
https://github.com/Data-Centric-AI-Community/fg-data-profiling/issues/1763
"""
import pandas as pd
import pytest

from data_profiling import ProfileReport


def test_issue1763():
    ipywidgets = pytest.importorskip("ipywidgets")

    df = pd.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})
    profile = ProfileReport(df, title="Test", minimal=True)

    profile.to_widgets()

    assert isinstance(profile.widgets, ipywidgets.Widget)
