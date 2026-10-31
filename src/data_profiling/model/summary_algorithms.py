import functools
from typing import Any, Callable, Optional, Tuple, TypeVar, Union

import numpy as np
import pandas as pd
from multimethod import multimethod
from scipy.stats import chisquare

from data_profiling.config import Settings

T = TypeVar("T")


def func_nullable_series_contains(fn: Callable) -> Callable:
    @functools.wraps(fn)
    def inner(
        config: Settings, series: pd.Series, state: dict, *args, **kwargs
    ) -> bool:
        if series.hasnans:
            series = series.dropna()
            if series.empty:
                return False

        return fn(config, series, state, *args, **kwargs)

    return inner


def safe_histogram(
    values: np.ndarray,
    bins: Union[int, str, np.ndarray] = "auto",
    weights: Optional[np.ndarray] = None,
    density: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Wrapper to avoid
    ValueError: Too many bins for data range. Cannot create N finite-sized bins.
    """
    try:
        return np.histogram(values, bins=bins, weights=weights, density=density)
    except ValueError as exc:
        if "Too many bins for data range" in str(exc):
            try:
                return np.histogram(
                    values, bins="auto", weights=weights, density=density
                )
            except ValueError:
                finite = values[np.isfinite(values)]
                if finite.size == 0:
                    return np.array([]), np.array([])
                vmin = float(np.min(finite))
                vmax = float(np.max(finite))
                if vmin == vmax:
                    eps = 0.5 if vmin == 0 else abs(vmin) * 0.5
                    bin_edges = np.array([vmin - eps, vmin + eps])
                else:
                    bin_edges = np.array([vmin, vmax])
                return np.histogram(
                    values, bins=bin_edges, weights=weights, density=density
                )
        raise


def histogram_compute(
    config: Settings,
    finite_values: np.ndarray,
    n_unique: int,
    name: str = "histogram",
    weights: Optional[np.ndarray] = None,
) -> dict:
    stats = {}
    if len(finite_values) == 0:
        return {name: []}

    hist_config = config.plot.histogram

    # Compute data range
    finite = finite_values[np.isfinite(finite_values)]
    vmin = float(np.min(finite))
    vmax = float(np.max(finite))
    data_range = vmax - vmin

    # Choose of Bins based on observed data values
    if data_range == 0:
        eps = 0.5 if vmin == 0 else abs(vmin) * 0.1
        bins = np.array([vmin - eps, vmin + eps])
    else:
        requested_bins = hist_config.bins if hist_config.bins > 0 else "auto"

        if isinstance(requested_bins, int):
            safe_bins = min(requested_bins, n_unique, hist_config.max_bins)

            safe_bins = max(1, safe_bins)

            bins = np.linspace(vmin, vmax, safe_bins + 1)
        else:
            bins = np.histogram_bin_edges(finite_values, bins="auto")
            if len(bins) - 1 > hist_config.max_bins:
                bins = np.linspace(vmin, vmax, hist_config.max_bins + 1)

    hist = np.histogram(
        finite_values,
        bins=bins,
        weights=weights,
        density=hist_config.density,
    )

    stats[name] = hist

    # Compute truncated histogram if percentile_cutoff is set
    cutoff = hist_config.percentile_cutoff
    if cutoff > 0.0:
        lower = np.percentile(finite, cutoff * 100)
        upper = np.percentile(finite, (1 - cutoff) * 100)
        mask = (finite_values >= lower) & (finite_values <= upper)
        truncated_values = finite_values[mask]
        truncated_weights = weights[mask] if weights is not None else None

        if len(truncated_values) > 0:
            t_vmin = float(np.min(truncated_values))
            t_vmax = float(np.max(truncated_values))
            t_range = t_vmax - t_vmin

            if t_range == 0:
                eps = 0.5 if t_vmin == 0 else abs(t_vmin) * 0.1
                t_bins = np.array([t_vmin - eps, t_vmin + eps])
            else:
                requested_bins = hist_config.bins if hist_config.bins > 0 else "auto"
                if isinstance(requested_bins, int):
                    safe_bins = min(requested_bins, n_unique, hist_config.max_bins)
                    safe_bins = max(1, safe_bins)
                    t_bins = np.linspace(t_vmin, t_vmax, safe_bins + 1)
                else:
                    t_bins = np.histogram_bin_edges(truncated_values, bins="auto")
                    if len(t_bins) - 1 > hist_config.max_bins:
                        t_bins = np.linspace(t_vmin, t_vmax, hist_config.max_bins + 1)

            stats[f"{name}_truncated"] = np.histogram(
                truncated_values,
                bins=t_bins,
                weights=truncated_weights,
                density=hist_config.density,
            )

    return stats


def chi_square(
    values: Optional[np.ndarray] = None,
    histogram: Optional[np.ndarray] = None,
) -> dict:
    # Case 1: histogram not passed → we compute it
    if histogram is None:
        if values is None:
            return {"statistic": 0, "pvalue": 0}

        # Try NumPy "auto" binning (may fail under NumPy 2)
        try:
            bins = np.histogram_bin_edges(values, bins="auto")
        except ValueError:
            # Fallback: basic 1-bin histogram covering the min→max range
            finite = values[np.isfinite(values)]
            if finite.size == 0:
                return {"statistic": 0, "pvalue": 0}

            vmin = float(finite.min())
            vmax = float(finite.max())
            if vmin == vmax:
                bins = np.array([vmin - 0.5, vmin + 0.5])
            else:
                bins = np.array([vmin, vmax])

        histogram, _ = np.histogram(values, bins=bins)

    # Case 2: histogram exists but is empty
    if histogram.size == 0 or histogram.sum() == 0:
        return {"statistic": 0, "pvalue": 0}

    return dict(chisquare(histogram)._asdict())


def series_hashable(
    fn: Callable[[Settings, pd.Series, dict], Tuple[Settings, pd.Series, dict]]
) -> Callable[[Settings, pd.Series, dict], Tuple[Settings, pd.Series, dict]]:
    @functools.wraps(fn)
    def inner(
        config: Settings, series: pd.Series, summary: dict
    ) -> Tuple[Settings, pd.Series, dict]:
        if not summary["hashable"]:
            return config, series, summary
        return fn(config, series, summary)

    return inner


def series_handle_nulls(
    fn: Callable[[Settings, pd.Series, dict], Tuple[Settings, pd.Series, dict]]
) -> Callable[[Settings, pd.Series, dict], Tuple[Settings, pd.Series, dict]]:
    """Decorator for nullable series"""

    @functools.wraps(fn)
    def inner(
        config: Settings, series: pd.Series, summary: dict
    ) -> Tuple[Settings, pd.Series, dict]:
        if series.hasnans:
            series = series.dropna()

        return fn(config, series, summary)

    return inner


def named_aggregate_summary(series: pd.Series, key: str) -> dict:
    summary = {
        f"max_{key}": np.max(series),
        f"mean_{key}": np.mean(series),
        f"median_{key}": np.median(series),
        f"min_{key}": np.min(series),
    }

    return summary


@multimethod
def describe_counts(
    config: Settings, series: Any, summary: dict
) -> Tuple[Settings, Any, dict]:
    raise NotImplementedError()


@multimethod
def describe_supported(
    config: Settings, series: Any, series_description: dict
) -> Tuple[Settings, Any, dict]:
    raise NotImplementedError()


@multimethod
def describe_generic(
    config: Settings, series: Any, summary: dict
) -> Tuple[Settings, Any, dict]:
    raise NotImplementedError()


@multimethod
def describe_numeric_1d(
    config: Settings, series: Any, summary: dict
) -> Tuple[Settings, Any, dict]:
    raise NotImplementedError()


@multimethod
def describe_text_1d(
    config: Settings, series: Any, summary: dict
) -> Tuple[Settings, Any, dict, Any]:
    raise NotImplementedError()


@multimethod
def describe_date_1d(
    config: Settings, series: Any, summary: dict
) -> Tuple[Settings, Any, dict]:
    raise NotImplementedError()


@multimethod
def describe_categorical_1d(
    config: Settings, series: pd.Series, summary: dict
) -> Tuple[Settings, pd.Series, dict]:
    raise NotImplementedError()


@multimethod
def describe_url_1d(
    config: Settings, series: Any, summary: dict
) -> Tuple[Settings, Any, dict]:
    raise NotImplementedError()


@multimethod
def describe_file_1d(
    config: Settings, series: Any, summary: dict
) -> Tuple[Settings, Any, dict]:
    raise NotImplementedError()


@multimethod
def describe_path_1d(
    config: Settings, series: Any, summary: dict
) -> Tuple[Settings, Any, dict]:
    raise NotImplementedError()


@multimethod
def describe_image_1d(
    config: Settings, series: Any, summary: dict
) -> Tuple[Settings, Any, dict]:
    raise NotImplementedError()


@multimethod
def describe_boolean_1d(
    config: Settings, series: Any, summary: dict
) -> Tuple[Settings, Any, dict]:
    raise NotImplementedError()


@multimethod
def describe_timeseries_1d(
    config: Settings, series: Any, summary: dict
) -> Tuple[Settings, Any, dict]:
    raise NotImplementedError()
