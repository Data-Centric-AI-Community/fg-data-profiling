from typing import Tuple

import numpy as np
import pandas as pd

from data_profiling.config import Settings
from data_profiling.model.summary_algorithms import (
    chi_square,
    describe_date_1d,
    histogram_compute,
    series_handle_nulls,
    series_hashable,
)
from data_profiling.model.typeset_relations import is_pandas_1


def to_datetime(series: pd.Series) -> pd.Series:
    if is_pandas_1():
        return pd.to_datetime(series, errors="coerce")
    return pd.to_datetime(series, format="mixed", errors="coerce")


@describe_date_1d.register
@series_hashable
@series_handle_nulls
def pandas_describe_date_1d(
    config: Settings, series: pd.Series, summary: dict
) -> Tuple[Settings, pd.Series, dict]:
    """Describe a date series.

    Args:
        config: report Settings object
        series: The Series to describe.
        summary: The dict containing the series description so far.

    Returns:
        A dict containing calculated series description values.
    """
    og_series = series.dropna()
    series = to_datetime(og_series)
    invalid_values = og_series[series.isna()]

    series = series.dropna()

    if summary["value_counts_without_nan"].empty:
        values = series.values
        summary.update(
            {
                "min": pd.NaT,
                "max": pd.NaT,
                "range": 0,
            }
        )
    else:
        summary.update(
            {
                "min": pd.Timestamp.to_pydatetime(series.min()),
                "max": pd.Timestamp.to_pydatetime(series.max()),
            }
        )

        summary["range"] = summary["max"] - summary["min"]

        values = series.values.astype("datetime64[s]").astype(np.int64)

    if config.vars.num.chi_squared_threshold > 0.0:
        summary["chi_squared"] = chi_square(values)

    summary.update(histogram_compute(config, values, series.nunique()))
    summary.update(
        {
            "invalid_dates": invalid_values.nunique(),
            "n_invalid_dates": len(invalid_values),
            "p_invalid_dates": len(invalid_values) / summary["n"],
        }
    )
    return config, values, summary
