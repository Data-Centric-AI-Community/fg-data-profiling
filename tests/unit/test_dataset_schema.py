import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def auto_dta(tmp_path):
    """Stata file shaped like Stata's ``auto2.dta`` example dataset."""
    rng = np.random.default_rng(1978)
    n = 74
    df = pd.DataFrame(
        {
            "make": [f"Make {i}" for i in range(n)],
            "price": rng.integers(3291, 15906, n),
            "mpg": rng.integers(12, 41, n),
            "rep78": pd.Categorical(
                rng.choice(["Poor", "Fair", "Average", "Good", "Excellent"], n)
            ),
            "weight": rng.integers(1760, 4840, n),
            "length": rng.integers(142, 233, n),
            "foreign": pd.Categorical(rng.choice(["Domestic", "Foreign"], n)),
        }
    )
    path = tmp_path / "auto2.dta"
    df.to_stata(path, write_index=False)
    return path


def test_dataset_schema(auto_dta):
    df = pd.read_stata(auto_dta)

    metadata = {
        "creator": "Firstname Lastname",
        "author": "Firstname Lastname",
        "description": "This profiling report was generated using a sample of 5% of the original dataset.",
        "copyright_holder": "RandoCorp LLC",
        "copyright_year": "2020",
        "url": "http://www.dataset-sources.com/data/dataset.dat",
    }

    # Length left out due to correlation with weight.
    report = df.profile_report(
        title="Dataset schema",
        dataset=metadata,
        minimal=True,
    )

    html = report.to_html()

    assert ">Dataset<" in html
    for key in metadata.keys():
        if not key.startswith("copyright_") and key != "url":
            assert f"<th>{key.capitalize()}<td" in html
    assert "<tr><th>Copyright</th><td>(c) RandoCorp LLC 2020</td></tr>"
    assert '<tr><th>URL</th><td><a href="http://www.dataset-sources.com/data/dataset.dat">http://www.dataset-sources.com/data/dataset.dat</a></td></tr>'
    assert ">Reproduction<" in html


def test_dataset_schema_empty(auto_dta):
    df = pd.read_stata(auto_dta)

    # Length left out due to correlation with weight.
    report = df.profile_report(
        title="Dataset schema empty",
        minimal=True,
        dataset=None,
    )

    html = report.to_html()

    assert ">Dataset<" not in html
    assert ">Reproduction<" in html
