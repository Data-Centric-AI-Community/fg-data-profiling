import numpy as np
import pandas as pd
import pytest

from data_profiling import ProfileReport
from data_profiling.model.pandas.describe_timeseries_pandas import compute_gap_stats


@pytest.fixture
def html_profile() -> str:
    size = 1000
    time_steps = np.arange(size)

    df = pd.DataFrame(
        {
            "ascending_sequence": time_steps,
            "descending_sequence": time_steps[::-1],
            "ascending_sequence_with_noise": [
                x + e for x, e in zip(time_steps, np.random.normal(5, 30, size))
            ],
            "descending_sequence_with_noise": [
                x + e for x, e in zip(time_steps[::-1], np.random.normal(5, 30, size))
            ],
            "ts_with_nan": [x if x % 4 != 0 else None for x in time_steps],
            "ts_negatives": -1 * time_steps,
            "constant": np.ones(size),
            "sin": [round(np.sin(x * np.pi / 180), 2) for x in time_steps],
            "cos": [round(np.cos(x * np.pi / 180), 2) for x in time_steps],
            "uniform": [round(x, 2) for x in np.random.uniform(0, 10, size)],
            "gaussian": [round(x, 2) for x in np.random.normal(0, 1, size)],
        }
    )

    profile = ProfileReport(df, tsmode=True)
    return profile.to_html()


@pytest.fixture
def sample_ts_df():
    dates = pd.date_range(start="2023-01-01", periods=100, freq="D")
    return pd.DataFrame(
        {
            "date": dates,
            "value": np.sin(np.arange(100) * np.pi / 180)
            + np.random.normal(0, 0.1, 100),
            "trend": np.arange(100) * 0.1,
            "category": ["A", "B"] * 50,
        }
    )


def test_timeseries_identification(html_profile: str):

    assert "<th>TimeSeries<td" in html_profile, "TimeSeries not detected"
    assert (
        'TimeSeries<td style="white-space: nowrap;">8' in html_profile
    ), "TimeSeries incorrectly identified"


def test_timeseries_autocorrelation_tab(html_profile: str):
    assert ">Autocorrelation<" in html_profile, "TimeSeries not detected"
    assert (
        html_profile.count(">Autocorrelation<") == 8
    ), "TimeSeries autocorrelation tabs incorrectly generated"


def test_timeseries_seasonality(html_profile: str):
    assert ">Seasonal<" in html_profile, "Seasonality incorrectly identified"
    assert (
        html_profile.count(">Seasonal<") == 4
    ), "Seasonality warning incorrectly identified"


def test_timeseries_with_sortby(sample_ts_df):
    # Test time series with explicit sort column
    profile = ProfileReport(sample_ts_df, tsmode=True, sortby="date")
    html = profile.to_html()
    assert "date" in html
    assert profile.config.vars.timeseries.sortby == "date"


def test_timeseries_gap_stats_with_string_date_index():
    # GH#1773
    dates = pd.date_range("2023-01-01", periods=10, freq="D").tolist()
    dates.append(pd.Timestamp("2023-01-20"))
    series = pd.Series(
        np.arange(11, dtype=float),
        index=pd.Index([date.strftime("%Y-%m-%d") for date in dates], name="Date Local"),
    )

    gap_stats = compute_gap_stats(series)

    assert gap_stats["n_gaps"] == 1
    assert gap_stats["min"] == pd.Timedelta(days=10)


def test_timeseries_without_sortby(sample_ts_df):
    # Test time series without explicit sort column
    profile = ProfileReport(sample_ts_df, tsmode=True)
    html = profile.to_html()
    assert profile.config.vars.timeseries.sortby is None
    assert "TimeSeries" in html


def test_invalid_sortby(sample_ts_df):
    # Test with non-existent sort column
    with pytest.raises(KeyError):
        profile = ProfileReport(sample_ts_df, tsmode=True, sortby="nonexistent")
        profile.to_html()


def test_timeseries_with_missing_values(sample_ts_df):
    # Introduce missing values
    df_with_missing = sample_ts_df.copy()
    df_with_missing.loc[10:20, "value"] = np.nan
    profile = ProfileReport(df_with_missing, tsmode=True)
    html = profile.to_html()
    assert "Missing values" in html


def test_non_numeric_timeseries():
    # Test handling of non-numeric time series
    dates = pd.date_range(start="2023-01-01", periods=100, freq="D")
    df = pd.DataFrame({"date": dates, "category": ["A", "B", "C"] * 33 + ["A"]})
    profile = ProfileReport(df, tsmode=True)
    html = profile.to_html()
    # Should not identify categorical column as time series
    assert html.count(">Autocorrelation<") == 0


def test_timeseries_config_persistence():
    # Test that time series configuration persists
    df = pd.DataFrame({"value": range(100)})
    profile = ProfileReport(df, tsmode=True)
    assert profile.config.vars.timeseries.active is True
    # Test config after invalidating cache
    profile.invalidate_cache()
    assert profile.config.vars.timeseries.active is True
