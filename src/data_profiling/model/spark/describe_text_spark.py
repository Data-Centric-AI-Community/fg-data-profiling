"""Text (free-form string) description for the Spark backend.

All per-row work (lengths, character and word extraction) runs on the cluster.
Only aggregated, bounded results are collected to the driver: one row per
distinct length, one row per distinct character, and the most frequent words
(``vars.text.spark_words_limit``).
"""
import string
from typing import Tuple

import pandas as pd
from pyspark.sql import DataFrame
from pyspark.sql import functions as F

from data_profiling.config import Settings
from data_profiling.model.pandas.describe_categorical_pandas import (
    length_summary_from_counts,
    unicode_summary_from_counts,
)
from data_profiling.model.summary_algorithms import describe_text_1d, histogram_compute

_TEXT = "_text"

# Same characters the pandas backend strips from word boundaries
# (``string.punctuation + string.whitespace``); escaped for a Java regex
# character class.
_STRIP_CLASS = "".join("\\" + c for c in string.punctuation) + "\\s"
_STRIP_PATTERN = f"^[{_STRIP_CLASS}]+|[{_STRIP_CLASS}]+$"


def _counts_to_series(pdf: pd.DataFrame, key: str) -> pd.Series:
    """Convert a two-column (key, count) frame to a Series sorted by count."""
    if pdf.empty:
        return pd.Series(dtype="int64")
    counts = pd.Series(pdf["count"].values, index=pdf[key].values)
    return counts.sort_values(ascending=False, kind="stable")


def _length_counts(text: DataFrame) -> pd.Series:
    pdf = text.groupBy(F.length(F.col(_TEXT)).alias("length")).count().toPandas()
    return _counts_to_series(pdf, "length")


def _character_counts(text: DataFrame) -> pd.Series:
    # substring() and char_length() work on code points, so characters outside
    # the Basic Multilingual Plane (emoji, CJK extensions) are counted once
    # instead of being split into UTF-16 surrogate halves.
    characters = text.where(F.length(F.col(_TEXT)) > 0).select(
        F.explode(
            F.expr(
                f"transform(sequence(1, char_length({_TEXT})), "
                f"i -> substring({_TEXT}, i, 1))"
            )
        ).alias("character")
    )
    pdf = characters.groupBy("character").count().toPandas()
    return _counts_to_series(pdf, "character")


def _word_counts(text: DataFrame, stop_words: list, limit: int) -> pd.Series:
    words = (
        text.select(F.explode(F.split(F.lower(F.col(_TEXT)), r"(?U)\s+")).alias("word"))
        .select(F.regexp_replace(F.col("word"), _STRIP_PATTERN, "").alias("word"))
        .where(F.col("word") != "")
    )
    if stop_words:
        words = words.where(~F.col("word").isin([w.lower() for w in stop_words]))

    pdf = (
        words.groupBy("word")
        .count()
        .orderBy(F.desc("count"), F.asc("word"))
        .limit(limit)
        .toPandas()
    )
    return _counts_to_series(pdf, "word")


@describe_text_1d.register
def describe_text_1d_spark(
    config: Settings, df: DataFrame, summary: dict
) -> Tuple[Settings, DataFrame, dict]:
    """Describe a text column on Spark.

    Mirrors ``pandas_describe_text_1d``: first rows, length summary and
    histogram, Unicode character summary and word counts, each controlled by
    ``config.vars.text``.

    Args:
        config: report Settings object
        df: single-column Spark DataFrame to describe.
        summary: the dict containing the series description so far.

    Returns:
        The config, the DataFrame and the updated summary.
    """
    text_config = config.vars.text
    text = (
        df.select(F.col(f"`{df.columns[0]}`").cast("string").alias(_TEXT))
        .where(F.col(_TEXT).isNotNull())
        .persist()
    )

    try:
        if not text_config.redact:
            summary["first_rows"] = (
                text.limit(5).toPandas()[_TEXT].rename(df.columns[0])
            )

        if not text.take(1):
            return config, df, summary

        if text_config.length:
            length_counts = _length_counts(text)
            summary.update(length_summary_from_counts(length_counts))
            summary.update(
                histogram_compute(
                    config,
                    length_counts.index.values,
                    len(length_counts),
                    name="histogram_length",
                    weights=length_counts.values,
                )
            )

        if text_config.characters:
            summary.update(unicode_summary_from_counts(_character_counts(text)))

        if text_config.words:
            word_counts = _word_counts(
                text, config.vars.cat.stop_words, text_config.spark_words_limit
            )
            if not word_counts.empty:
                summary["word_counts"] = word_counts
    finally:
        text.unpersist()

    return config, df, summary
