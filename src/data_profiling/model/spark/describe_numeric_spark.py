from typing import Tuple

import numpy as np
import pyspark.sql.functions as F
from pyspark.sql import DataFrame

from data_profiling.config import Settings
from data_profiling.model.summary_algorithms import histogram_compute


def numeric_stats_spark(df: DataFrame, summary: dict) -> dict:
    column = df.columns[0]

    # Removing null types from numeric summary stats to match Pandas defaults which skip na's (skipna=False)
    finite_filter = (
        F.col(column).isNotNull()
        & ~F.isnan(F.col(column))
        & ~F.col(column).isin([np.inf, -np.inf])
    )
    non_null_df = df.filter(finite_filter)

    expr = [
        F.mean(F.col(column)).alias("mean"),
        F.stddev(F.col(column)).alias("std"),
        F.variance(F.col(column)).alias("variance"),
        F.min(F.col(column)).alias("min"),
        F.max(F.col(column)).alias("max"),
        F.kurtosis(F.col(column)).alias("kurtosis"),
        F.skewness(F.col(column)).alias("skewness"),
        F.sum(F.col(column)).alias("sum"),
    ]
    return non_null_df.agg(*expr).first().asDict()


def describe_numeric_1d_spark(
    config: Settings, df: DataFrame, summary: dict
) -> Tuple[Settings, DataFrame, dict]:
    """Describe a boolean series.

    Args:
        series: The Series to describe.
        summary: The dict containing the series description so far.

    Returns:
        A dict containing calculated series description values.
    """

    stats = numeric_stats_spark(df, summary)
    summary["min"] = stats["min"]
    summary["max"] = stats["max"]
    summary["mean"] = stats["mean"]
    summary["std"] = stats["std"]
    summary["variance"] = stats["variance"]
    summary["skewness"] = stats["skewness"]
    summary["kurtosis"] = stats["kurtosis"]
    summary["sum"] = stats["sum"]

    value_counts = summary["value_counts"]

    n_infinite = (
        value_counts.where(F.col(df.columns[0]).isin([np.inf, -np.inf]))
        .agg(F.sum(F.col("count")).alias("count"))
        .first()
    )
    if n_infinite is None or n_infinite["count"] is None:
        n_infinite = 0
    else:
        n_infinite = n_infinite["count"]
    summary["n_infinite"] = n_infinite

    n_zeros = value_counts.where(f"`{df.columns[0]}` = 0").first()
    if n_zeros is None:
        n_zeros = 0
    else:
        n_zeros = n_zeros["count"]
    summary["n_zeros"] = n_zeros

    n_negative = (
        value_counts.where(f"`{df.columns[0]}` < 0")
        .agg(F.sum(F.col("count")).alias("count"))
        .first()
    )
    if n_negative is None or n_negative["count"] is None:
        n_negative = 0
    else:
        n_negative = n_negative["count"]
    summary["n_negative"] = n_negative

    quantiles = config.vars.num.quantiles
    quantile_threshold = 0.05

    if summary.get("n") == summary.get("n_missing"):
        # This means the entire column is null/nan, so summary values need to be hard-coded:
        summary.update({f"{percentile:.0%}": np.nan for percentile in quantiles})

        summary["mad"] = np.nan
        summary["iqr"] = np.nan

    else:
        summary.update(
            {
                f"{percentile:.0%}": value
                for percentile, value in zip(
                    quantiles,
                    df.stat.approxQuantile(
                        f"{df.columns[0]}",
                        quantiles,
                        quantile_threshold,
                    ),
                )
            }
        )

        median = summary.get("50%")

        summary["mad"] = df.select(
            # No cast: an int cast overflows (an error under ANSI mode) for
            # values beyond 2**31 and truncates non-integers.
            (F.abs(F.col(f"{df.columns[0]}") - median)).alias("abs_dev")
        ).stat.approxQuantile("abs_dev", [0.5], quantile_threshold)[0]

        summary["iqr"] = summary["75%"] - summary["25%"]

    # FIXME: move to fmt
    summary["p_negative"] = summary["n_negative"] / summary["n"]
    if summary["min"] is None or summary["max"] is None:
        summary["range"] = np.nan
    else:
        summary["range"] = summary["max"] - summary["min"]
    summary["cv"] = summary["std"] / summary["mean"] if summary["mean"] else np.nan
    summary["p_zeros"] = summary["n_zeros"] / summary["n"]
    summary["p_infinite"] = summary["n_infinite"] / summary["n"]

    # TODO - enable this feature
    # because spark doesn't have an indexing system, there isn't really the idea of monotonic increase/decrease
    # [feature enhancement] we could implement this if the user provides an ordinal column to use for ordering
    # ... https://stackoverflow.com/questions/60221841/how-to-detect-monotonic-decrease-in-pyspark
    summary["monotonic"] = 0

    # this function only displays the top N (see config) values for a histogram.
    # This might be confusing if there are a lot of values of equal magnitude, but we cannot bring all the values to
    # display in pandas display
    # the alternative is to do this in spark natively, but it is not trivial
    infinity_values = [np.inf, -np.inf]

    infinity_index = summary["value_counts_without_nan"].index.isin(infinity_values)

    summary.update(
        histogram_compute(
            config,
            summary["value_counts_without_nan"][~infinity_index].index.values,
            summary["n_distinct"],
            weights=summary["value_counts_without_nan"][~infinity_index].values,
        )
    )

    return config, df, summary
