import numpy as np
import pandas as pd

from data_profiling import ProfileReport


def test_urls(tmp_path):
    # Same layout as openeventdata's whitelist_urls.csv (no header row)
    rng = np.random.default_rng(42)
    n = 200
    sites = [f"news{i}" for i in range(40)]
    rows = pd.DataFrame(
        {
            "source": rng.choice(sites, n),
            "url": [
                f"http://www.{s}.com/feed/{i}"
                for i, s in enumerate(rng.choice(sites, n))
            ],
            "reach": rng.choice(["international", "national", "local"], n),
            "language": rng.choice(["english", "spanish", "arabic"], n),
        }
    )
    file_name = tmp_path / "whitelist_urls.csv"
    rows.to_csv(file_name, header=False, index=False)

    df = pd.read_csv(
        file_name, header=None, names=["source", "url", "reach", "language"]
    )

    # Add ~10% missing values
    df = df.mask(np.random.random(df.shape) < 0.1)

    profile = ProfileReport(
        df,
        title="DataFrame with URL column",
        samples={"head": 0, "tail": 0},
        explorative=True,
    )

    assert "URL</span>" in profile.to_html(), "URL not detected"
    assert "<th>URL<td" in profile.to_html(), "URL not detected"
