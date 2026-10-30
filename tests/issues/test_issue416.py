"""
Test for issue 416:
https://github.com/Data-Centric-AI-Community/data-profiling/issues/416
"""
import numpy as np
import pandas as pd

import data_profiling


def test_issue416(tmp_path):
    # Shaped like clickstream-pandas' products.tsv: a category and a product URL
    rng = np.random.default_rng(416)
    categories = ["Sports", "Books", "Music", "Garden", "Toys"]
    category = rng.choice(categories, 200)
    products = pd.DataFrame(
        {
            "category": category,
            "url": [
                f"http://www.acme.com/{c.lower()}/product{i}"
                for i, c in enumerate(category)
            ],
        }
    )
    file_name = tmp_path / "products.tsv"
    products.to_csv(file_name, sep="\t", index=False)

    df = pd.read_csv(file_name, sep="\t")
    df["path"] = df["url"].str.replace("http://www.acme.com", "", regex=False)

    profile = data_profiling.ProfileReport(
        df,
        title="YData Profiling Report",
        html={"style": {"full_width": True}},
        explorative=True,
        progress_bar=False,
    )
    data = profile.get_description()

    assert data.table["types"]["Categorical"] == 1
    assert data.table["types"]["Path"] == 1
    assert data.table["types"]["URL"] == 1
    assert data.variables["path"]["common_prefix"] == "/"
