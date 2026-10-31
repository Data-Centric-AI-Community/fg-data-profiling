"""
Test for issue 120:
https://github.com/Data-Centric-AI-Community/data-profiling/issues/120
"""
import numpy as np
import pandas as pd

from data_profiling import ProfileReport

NA = "not available in demo dataset"


def _analytics_export(n: int = 30, seed: int = 120) -> pd.DataFrame:
    """Shaped like the Google Analytics export attached to the issue: many
    constant placeholder columns, fully empty columns, ~1e18 visitor ids,
    yyyymmdd dates, epoch timestamps and sparse numeric totals."""
    rng = np.random.default_rng(seed)
    start = rng.integers(1_470_000_000, 1_500_000_000, n)
    visitor = rng.integers(10**17, 9 * 10**18, n, dtype=np.int64)
    df = pd.DataFrame(
        {
            "Unnamed: 0": rng.integers(100_000, 900_000, n),
            "channelGrouping": rng.choice(["Social", "Referral", "Direct"], n),
            "date": rng.choice([20160901, 20161226, 20170522, 20170202], n),
            "device_browser": rng.choice(["Safari", "Chrome"], n),
            "device_browserVersion": NA,
            "device_isMobile": rng.choice([True, False], n),
            "device_language": NA,
            "fullVisitorId": visitor,
            "geoNetwork_country": rng.choice(["Vietnam", "United States"], n),
            "geoNetwork_networkDomain": rng.choice(["unknown.unknown", "(not set)"], n),
            "sessionId": [f"{v:019d}_{s}" for v, s in zip(visitor, start)],
            "totals_visits": 1,
            "totals_hits": rng.integers(1, 4, n),
            "totals_pageviews": rng.choice([1.0, 2.0, np.nan], n),
            "totals_bounces": rng.choice([1.0, np.nan], n),
            "trafficSource_source": rng.choice(["youtube.com", "(direct)"], n),
            "trafficSource_keyword": np.nan,
            "visitStartTime": start,
            "trafficSource_referralPath": rng.choice(["/yt/about/", "/", None], n),
            "trafficSource_adwordsClickInfo_page": np.nan,
            "totals_transactionRevenue": np.nan,
        }
    )
    return df


def test_issue_120(tmp_path):
    file_name = tmp_path / "data_profiling_bug.txt"
    _analytics_export().to_csv(file_name, index=False)
    df = pd.read_csv(file_name)

    report = ProfileReport(
        df,
        correlations=None,
        progress_bar=False,
        pool_size=1,
        vars={"cat": {"words": True, "characters": True}},
    )
    _ = report.report
    assert report.description_set is not None
