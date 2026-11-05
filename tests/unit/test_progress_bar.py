import pandas as pd

from data_profiling import ProfileReport


def test_progress_bar_disabled_has_no_stderr_output(capsys):
    df = pd.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})
    profile = ProfileReport(df, progress_bar=False, minimal=True)
    profile.to_html()

    assert capsys.readouterr().err == ""
