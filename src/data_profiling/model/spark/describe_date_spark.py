from typing import Tuple

import pyspark.sql.functions as F
from numpy import array
from pyspark.sql import DataFrame

from data_profiling.config import Settings


def date_stats_spark(df: DataFrame, summary: dict) -> dict:
    column = df.columns[0]

    expr = [
        F.min(F.col(column)).alias("min"),
        F.max(F.col(column)).alias("max"),
    ]

    return df.agg(*expr).first().asDict()


def describe_date_1d_spark(
    config: Settings, df: DataFrame, summary: dict
) -> Tuple[Settings, DataFrame, dict]:
    """Describe a date series.

    Args:
        series: The Series to describe.
        summary: The dict containing the series description so far.

    Returns:
        A dict containing calculated series description values.
    """
    col_name = df.columns[0]
    stats = date_stats_spark(df, summary)

    summary.update({"min": stats["min"], "max": stats["max"]})

    summary["range"] = summary["max"] - summary["min"]

    # Convert date to numeric so we can compute the histogram
    df = df.withColumn(col_name, F.unix_timestamp(df[col_name]))

    # Get the number of bins
    bins = config.plot.histogram.bins
    bins_arg = "auto" if bins == 0 else min(bins, summary["n_distinct"])

    # Run the histogram
    bin_edges, hist = df.select(col_name).rdd.flatMap(lambda x: x).histogram(bins_arg)

    summary.update({"histogram": (array(hist), array(bin_edges))})

    # A Spark date/timestamp column is type-enforced, so there are no
    # unparseable "invalid" dates (unlike the pandas string-coercion path). Set
    # the same keys the pandas backend produces and the shared date renderer
    # reads, so rendering the report doesn't raise ``KeyError: 'n_invalid_dates'``.
    summary.update(
        {
            "invalid_dates": 0,
            "n_invalid_dates": 0,
            "p_invalid_dates": 0.0,
        }
    )
    return config, df, summary
