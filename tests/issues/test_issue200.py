"""
Test for issue 200:
https://github.com/Data-Centric-AI-Community/data-profiling/issues/200
"""
import pandas as pd

from data_profiling import ProfileReport


def test_issue200():
    df = pd.DataFrame([0, 1, 2], columns=["a"], index=["0", "1", "2"])

    assert pd.api.types.is_string_dtype(df.index.dtype)
    report = ProfileReport(df, title="String indices", progress_bar=False, pool_size=1)
    assert (
        "<title>String indices</title>" in report.to_html()
    ), "Profile report should be generated."
