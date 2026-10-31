"""
Test for issue 388:
https://github.com/Data-Centric-AI-Community/data-profiling/issues/388
"""
import pytest

from data_profiling.controller import console


def test_issue388(titanic_csv, test_output_dir):
    console_data = titanic_csv

    report = test_output_dir / "test_unsupported_extension.foobar"
    with pytest.warns(UserWarning) as warning:
        console.main(["-s", "--pool_size", "1", str(console_data), str(report)])

    assert any("Extension .foobar not supported." in w.message.args[0] for w in warning)

    assert not report.exists(), "Report with .foobar extension shouldn't exist"
    assert report.with_suffix(
        ".html"
    ).exists(), "Report with .html extension should exist"
