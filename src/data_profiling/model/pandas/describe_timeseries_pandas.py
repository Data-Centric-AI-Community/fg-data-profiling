from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd
from scipy.fft import fft as scipy_fft
from scipy.signal import find_peaks
from statsmodels.tsa.stattools import adfuller

from data_profiling.config import Settings
from data_profiling.model.summary_algorithms import (
    describe_numeric_1d,
    describe_timeseries_1d,
    series_handle_nulls,
    series_hashable,
)


def stationarity_test(config: Settings, series: pd.Series) -> Tuple[bool, float]:
    # make sure the data has no missing values
    adfuller_test = adfuller(
        series.dropna(),
        autolag=config.vars.timeseries.autolag,
        maxlag=config.vars.timeseries.maxlag,
    )
    p_value = adfuller_test[1]

    significance_threshold = config.vars.timeseries.significance
    return p_value < significance_threshold, p_value


def fftfreq(n: int, d: float = 1.0) -> np.ndarray:
    """
    Return the Discrete Fourier Transform sample frequencies.

    Args:
        n : int
            Window length.
        d : scalar, optional
            Sample spacing (inverse of the sampling rate). Defaults to 1.

    Returns:
        f : ndarray
            Array of length `n` containing the sample frequencies.
    """
    val = 1.0 / (n * d)
    results = np.empty(n, int)
    N = (n - 1) // 2 + 1
    p1 = np.arange(0, N, dtype=int)
    results[:N] = p1
    p2 = np.arange(-(n // 2), 0, dtype=int)
    results[N:] = p2
    return results * val


def seasonality_test(series: pd.Series, mad_threshold: float = 6.0) -> Dict[str, Any]:
    """Detect seasonality with FFT

    Source: https://github.com/facebookresearch/Kats/blob/main/kats/detectors/seasonality.py

    Args:
        mad_threshold: Optional; float; constant for the outlier algorithm for peak
            detector. The larger the value the less sensitive the outlier algorithm
            is.

    Returns:
        FFT Plot with peaks, selected peaks, and outlier boundary line.
    """

    fft = get_fft(series)
    _, _, peaks = get_fft_peaks(fft, mad_threshold)
    seasonality_presence = len(peaks.index) > 0
    selected_seasonalities = []
    if seasonality_presence:
        selected_seasonalities = peaks["freq"].transform(lambda x: 1 / x).tolist()

    return {
        "seasonality_presence": seasonality_presence,
        "seasonalities": selected_seasonalities,
    }


def get_fft(series: pd.Series) -> pd.DataFrame:
    """Computes FFT

    Args:
        series: pd.Series
            time series

    Returns:
        DataFrame with columns 'freq' and 'ampl'.
    """
    data_fft = scipy_fft(series.to_numpy())
    data_psd = np.abs(data_fft) ** 2
    fftfreq_ = fftfreq(len(data_psd), 1.0)
    pos_freq_ix = fftfreq_ > 0

    freq = fftfreq_[pos_freq_ix]
    ampl = 10 * np.log10(data_psd[pos_freq_ix])

    return pd.DataFrame({"freq": freq, "ampl": ampl})


def get_fft_peaks(
    fft: pd.DataFrame, mad_threshold: float = 6.0
) -> Tuple[float, pd.DataFrame, pd.DataFrame]:
    """Computes peaks in fft, selects the highest peaks (outliers) and
        removes the harmonics (multiplies of the base harmonics found)

    Args:
        fft: FFT computed by get_fft
        mad_threshold: Optional; constant for the outlier algorithm for peak detector.
            The larger the value the less sensitive the outlier algorithm is.

    Returns:
        outlier threshold, peaks, selected peaks.
    """
    pos_fft = fft.loc[fft["ampl"] > 0]
    median = pos_fft["ampl"].median()
    pos_fft_above_med = pos_fft[pos_fft["ampl"] > median]
    mad = abs(pos_fft_above_med["ampl"] - pos_fft_above_med["ampl"].mean()).mean()

    threshold = median + mad * mad_threshold

    peak_indices = find_peaks(fft["ampl"], threshold=0.1)
    peaks = fft.loc[peak_indices[0], :]

    orig_peaks = peaks.copy()

    peaks = peaks.loc[peaks["ampl"] > threshold].copy()
    peaks["Remove"] = [False] * len(peaks.index)
    peaks.reset_index(inplace=True)

    # Filter out harmonics
    for idx1 in range(len(peaks)):
        curr = peaks.loc[idx1, "freq"]
        for idx2 in range(idx1 + 1, len(peaks)):
            if peaks.loc[idx2, "Remove"] is True:
                continue
            fraction = (peaks.loc[idx2, "freq"] / curr) % 1
            if fraction < 0.01 or fraction > 0.99:
                peaks.loc[idx2, "Remove"] = True
    peaks = peaks.loc[~peaks["Remove"]]
    peaks.drop(inplace=True, columns="Remove")
    return threshold, orig_peaks, peaks


def identify_gaps(
    gap: pd.Series, is_datetime: bool, gap_tolerance: int = 2
) -> Tuple[pd.Series, list]:
    zero = pd.Timedelta(0) if is_datetime else 0
    diff = gap.diff()

    non_zero_diff = diff[diff > zero]
    min_gap_size = gap_tolerance * non_zero_diff.mean()

    gap_stats = non_zero_diff[non_zero_diff > min_gap_size]
    anchors = gap[diff > min_gap_size].index

    gaps = []
    for i in anchors:
        gaps.append(gap.loc[gap.index[[i - 1, i]]].values)

    return gap_stats, gaps


def compute_gap_stats(series: pd.Series) -> pd.Series:
    """Computes the intertevals in the series normalized by the period.

    Args:
        series (pd.Series): time series data to analysis.

    Returns:
        A series with the gaps intervals.
    """

    gap = series.dropna()
    index_name = gap.index.name if gap.index.name else "index"
    gap = gap.reset_index()[index_name]
    gap.index.name = None

    is_datetime = isinstance(series.index, pd.DatetimeIndex)
    gap_stats, gaps = identify_gaps(gap, is_datetime)
    has_gaps = len(gap_stats) > 0

    stats = {
        "min": gap_stats.min() if has_gaps else 0,
        "max": gap_stats.max() if has_gaps else 0,
        "mean": gap_stats.mean() if has_gaps else 0,
        "std": gap_stats.std() if len(gap_stats) > 1 else 0,
        "series": series,
        "gaps": gaps,
        "n_gaps": len(gaps),
    }
    return stats


@describe_timeseries_1d.register
@series_hashable
@series_handle_nulls
def pandas_describe_timeseries_1d(
    config: Settings, series: pd.Series, summary: dict
) -> Tuple[Settings, pd.Series, dict]:
    """Describe a timeseries.

    Args:
        config: report Settings object
        series: The Series to describe.
        summary: The dict containing the series description so far.

    Returns:
        A dict containing calculated series description values.
    """
    config, series, stats = describe_numeric_1d(config, series, summary)

    stats["seasonal"] = seasonality_test(series)["seasonality_presence"]
    is_stationary, p_value = stationarity_test(config, series)
    stats["stationary"] = is_stationary and not stats["seasonal"]
    stats["addfuller"] = p_value
    stats["series"] = series
    stats["gap_stats"] = compute_gap_stats(series)

    return config, series, stats
