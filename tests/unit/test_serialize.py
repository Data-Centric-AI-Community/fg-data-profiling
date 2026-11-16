import numpy as np
import pandas as pd
import pytest

from data_profiling import ProfileReport
from data_profiling.model import BaseDescription


def test_load(meteorites_csv, test_output_dir):

    # For reproducibility
    np.random.seed(7331)

    df = pd.read_csv(meteorites_csv)

    # Note: Pandas does not support dates before 1880, so we ignore these for this analysis
    df["year"] = pd.to_datetime(df["year"], errors="coerce")

    # Example: Constant variable
    df["source"] = "NASA"

    # Example: Boolean variable
    df["boolean"] = np.random.choice([True, False], df.shape[0])

    # Example: Mixed with base types
    df["mixed"] = np.random.choice([1, "A"], df.shape[0])

    # Example: Highly correlated variables
    df["reclat_city"] = df["reclat"] + np.random.normal(scale=5, size=(len(df)))

    # Example: Duplicate observations
    duplicates_to_add = pd.DataFrame(df.iloc[0:10].copy())

    df = pd.concat([df, duplicates_to_add], ignore_index=True)

    profile1 = ProfileReport(
        df,
        title="NASA Meteorites",
        samples={"head": 5, "tail": 5},
        duplicates={"head": 10},
        minimal=True,
        progress_bar=False,
    )

    test_output_path = test_output_dir / "NASA-Meteorites.pp"
    json1 = profile1.to_json()
    profile1.dump(test_output_path)
    _ = profile1.to_html()

    assert test_output_path.exists(), "Output file does not exist"

    profile2 = ProfileReport(df, progress_bar=False).load(test_output_path)
    # json1 are compute before dumps, so _description_set should be the same
    assert isinstance(profile2._description_set, BaseDescription)

    # profile1 is lazy, html1 are compute after dumps, so report should be None
    assert profile2._report is None

    json2 = profile2.to_json()

    # both profile should generate same output
    assert json1 == json2


def test_load_error():
    import warnings

    warnings.filterwarnings("error")

    df = pd.DataFrame(
        {"a": [1, 2, 3, 1, 1, 1], "b": [1, 2, 3, 4, 5, 6], "c": [1, 2, 3, 4, 5, 6]}
    )
    profile1 = ProfileReport(df, minimal=True, progress_bar=False)

    data = profile1.dumps()

    # config not match but ignore_config
    ProfileReport(df, minimal=False, progress_bar=False).loads(data)

    # df not match
    with pytest.raises(ValueError) as e:
        ProfileReport(df=df[["a", "b"]][:], minimal=True, progress_bar=False).loads(
            data
        )

    assert str(e.value) == "DataFrame does not match with the current ProfileReport."
