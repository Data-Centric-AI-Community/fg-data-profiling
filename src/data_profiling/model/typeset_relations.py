import functools
from typing import Callable, Dict, Optional

import numpy as np
import pandas as pd
from pandas.api import types as pdt

from data_profiling.config import Settings
from data_profiling.model.type_rules import is_categorical_cardinality, is_few_numbers
from data_profiling.model.typeset_base import series_handle_nulls, step_cache
from data_profiling.utils.versions import is_pandas_1

# Checks that must hold for every value are first run on an evenly spaced
# sample: if a sampled value fails, the column fails, without scanning it.
# Checks on counts (distinct values, ratios) always use the full column, so
# sampling never changes the inferred type.
SAMPLE_SIZE = 1000


def _sample(series: pd.Series) -> Optional[pd.Series]:
    """An evenly spaced sample of the series, or None if it is small."""
    n = len(series)
    if n <= 2 * SAMPLE_SIZE:
        return None
    return series.iloc[np.linspace(0, n - 1, SAMPLE_SIZE).astype(np.int64)]


def _n_unique(series: pd.Series, state: dict) -> int:
    """Distinct non-null values, computed once per traversal step."""
    cache = step_cache(state)
    if "n_unique" not in cache:
        cache["n_unique"] = series.nunique()
    return cache["n_unique"]


def _expand(cached: Optional[pd.Series], series: pd.Series) -> Optional[pd.Series]:
    """Place a result computed on the non-null values back on the full series.

    Returns None (the caller recomputes) unless ``cached`` was computed on
    exactly the non-null values of ``series``. Missing positions get the
    missing value of the result's dtype, as the conversion functions do.
    """
    if cached is None or series.dtype != object:
        return None
    mask = series.notna().to_numpy()
    if not cached.index.equals(series.index[mask]):
        return None
    if mask.all():
        return cached
    indexer = np.full(len(series), -1, dtype=np.intp)
    indexer[mask] = np.arange(len(cached))
    values = cached.array.take(indexer, allow_fill=True)
    return pd.Series(values, index=series.index, name=series.name)


def is_nullable(series: pd.Series, state: dict) -> bool:
    return series.count() > 0


def try_func(fn: Callable) -> Callable:
    @functools.wraps(fn)
    def inner(series: pd.Series, *args, **kwargs) -> bool:
        try:
            return fn(series, *args, **kwargs)
        except:  # noqa: E722
            return False

    return inner


def string_is_bool(series: pd.Series, state: dict, k: Dict[str, bool]) -> bool:
    @series_handle_nulls
    @try_func
    def tester(s: pd.Series, state: dict) -> bool:
        sample = _sample(s)
        if sample is not None and not sample.str.lower().isin(k.keys()).all():
            return False
        return s.str.lower().isin(k.keys()).all()

    if isinstance(series.dtype, pd.CategoricalDtype):
        return False

    cache = step_cache(state)
    key = ("is_bool", frozenset(k.items()))
    if key not in cache:
        cache[key] = tester(series, state)
    return cache[key]


def string_to_bool(series: pd.Series, state: dict, k: Dict[str, bool]) -> pd.Series:
    return series.str.lower().map(k)


def numeric_is_category(series: pd.Series, state: dict, k: Settings) -> bool:
    return is_few_numbers(_n_unique(series, state), k)


def to_category(series: pd.Series, state: dict) -> pd.Series:
    hasnans = series.hasnans
    val = series.astype(str)
    if hasnans:
        val = val.replace("nan", np.nan)
        val = val.replace("<NA>", np.nan)

    return val.astype("string")


@series_handle_nulls
def series_is_string(series: pd.Series, state: dict) -> bool:
    # String dtypes (pandas "string", Arrow-backed string columns) guarantee
    # every non-null value is a str. Checking element-wise would call
    # astype(str), which for Arrow-backed columns builds a fixed-width numpy
    # array as wide as the longest document.
    if isinstance(series.dtype, pd.StringDtype) or (
        isinstance(series.dtype, pd.ArrowDtype) and pdt.is_string_dtype(series.dtype)
    ):
        return True
    if not all(isinstance(v, str) for v in series.values[0:5]):
        return False
    try:
        return (series.astype(str).values == series.values).all()
    except (TypeError, ValueError):
        return False


@series_handle_nulls
def string_is_category(series: pd.Series, state: dict, k: Settings) -> bool:
    """String is category, if the following conditions are met
    - has at least one and less or equal distinct values as threshold
    - (distinct values / count of all values) is less than threshold
    - is not bool"""
    return is_categorical_cardinality(
        _n_unique(series, state), series.size, k
    ) and not string_is_bool(series, state, k.vars.bool.mappings)


@series_handle_nulls
def string_is_datetime(series: pd.Series, state: dict) -> bool:
    """If we can transform data to datetime and at least one is valid date.

    Strings that parse as missing ("", "NaT") are allowed: they become missing
    values of the date column. Among the values that parse to a date, two kinds
    are not dates:

    - values without a digit: bare month and weekday names ("Jan" becomes
      0001-01-01) are categories;
    - columns of plain numbers: "2024" or "20240101" are read like any other
      number, so a year column gets the same type whether it has 3 or 30
      distinct values.
    """
    sample = _sample(series)
    try:
        if sample is not None:
            _parse_datetime(sample)
        parsed = _parse_datetime(series)
    except:  # noqa: E722
        return False

    is_date = parsed.notna()
    if not is_date.any():
        return False
    dates = series[is_date]
    if not dates.str.contains(r"\d", regex=True).all():
        return False
    step_cache(state)["datetime"] = parsed

    # A single non-number in the sample is enough to know the column is not
    # all plain numbers.
    dates_sample = _sample(dates)
    if dates_sample is not None and (
        pd.to_numeric(dates_sample, errors="coerce").isna().any()
    ):
        return True
    return not pd.to_numeric(dates, errors="coerce").notna().all()


@series_handle_nulls
def string_is_numeric(series: pd.Series, state: dict, k: Settings) -> bool:
    if pdt.is_bool_dtype(series) or object_is_bool(series, state):
        return False

    sample = _sample(series)
    try:
        if sample is not None:
            _ = sample.astype(float)
        _ = series.astype(float)
        converted = pd.to_numeric(series, errors="coerce")
    except:  # noqa: E722
        return False
    if converted.hasnans and converted.count() == 0:
        return False
    step_cache(state)["numeric"] = converted

    return not numeric_is_category(series, state, k)


def _parse_datetime(series: pd.Series) -> pd.Series:
    if is_pandas_1():
        return pd.to_datetime(series)
    return pd.to_datetime(series, format="mixed")


def string_to_datetime(series: pd.Series, state: dict) -> pd.Series:
    """Cast to datetime, reusing the parse done by ``string_is_datetime``."""
    cached = _expand(step_cache(state).get("datetime"), series)
    return _parse_datetime(series) if cached is None else cached


def string_to_numeric(series: pd.Series, state: dict) -> pd.Series:
    """Cast to numbers, reusing the conversion done by ``string_is_numeric``."""
    cached = _expand(step_cache(state).get("numeric"), series)
    return pd.to_numeric(series, errors="coerce") if cached is None else cached


hasnan_bool_name = "boolean"


def to_bool(series: pd.Series) -> pd.Series:
    dtype = hasnan_bool_name if series.hasnans else bool
    return series.astype(dtype)


@series_handle_nulls
def object_is_bool(series: pd.Series, state: dict) -> bool:
    if pdt.is_object_dtype(series):
        bool_set = {True, False}
        try:
            ret = all(item in bool_set for item in series)
        except:  # noqa: E722
            ret = False

        return ret
    return False
