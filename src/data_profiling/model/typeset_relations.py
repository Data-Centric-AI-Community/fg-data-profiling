import functools
from typing import Callable, Dict

import numpy as np
import pandas as pd
from pandas.api import types as pdt

from data_profiling.config import Settings
from data_profiling.model.typeset_base import series_handle_nulls
from data_profiling.utils.versions import is_pandas_1


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
        return s.str.lower().isin(k.keys()).all()

    if isinstance(series.dtype, pd.CategoricalDtype):
        return False

    return tester(series, state)


def string_to_bool(series: pd.Series, state: dict, k: Dict[str, bool]) -> pd.Series:
    return series.str.lower().map(k)


def numeric_is_category(series: pd.Series, state: dict, k: Settings) -> bool:
    n_unique = series.nunique()
    threshold = k.vars.num.low_categorical_threshold
    return 1 <= n_unique <= threshold


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
    n_unique = series.nunique()
    unique_threshold = k.vars.cat.percentage_cat_threshold
    threshold = k.vars.cat.cardinality_threshold
    return (
        1 <= n_unique <= threshold
        and (
            n_unique / series.size < unique_threshold
            if unique_threshold <= 1
            else n_unique / series.size <= unique_threshold
        )
        and not string_is_bool(series, state, k.vars.bool.mappings)
    )


@series_handle_nulls
def string_is_datetime(series: pd.Series, state: dict) -> bool:
    """If we can transform data to datetime and at least one is valid date.

    Two kinds of strings that the parser accepts are not dates:

    - values without a digit: bare month and weekday names ("Jan" becomes
      0001-01-01) are categories;
    - values that are plain numbers: "2024" or "20240101" are read like any
      other number, so a year column gets the same type whether it has 3 or
      30 distinct values.
    """
    try:
        if not series.str.contains(r"\d", regex=True).all():
            return False
        if pd.to_numeric(series, errors="coerce").notna().all():
            return False
        return not string_to_datetime(series, state).isna().all()
    except:  # noqa: E722
        return False


@series_handle_nulls
def string_is_numeric(series: pd.Series, state: dict, k: Settings) -> bool:
    if pdt.is_bool_dtype(series) or object_is_bool(series, state):
        return False

    try:
        _ = series.astype(float)
        r = pd.to_numeric(series, errors="coerce")
        if r.hasnans and r.count() == 0:
            return False
    except:  # noqa: E722
        return False

    return not numeric_is_category(series, state, k)


def string_to_datetime(series: pd.Series, state: dict) -> pd.Series:
    if is_pandas_1():
        return pd.to_datetime(series)
    return pd.to_datetime(series, format="mixed")


def string_to_numeric(series: pd.Series, state: dict) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


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
