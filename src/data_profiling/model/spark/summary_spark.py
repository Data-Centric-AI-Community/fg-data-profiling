"""Compute statistical description of datasets."""
from typing import Tuple

import numpy as np
from pyspark.sql import DataFrame
from pyspark.sql.types import (
    ArrayType,
    BooleanType,
    ByteType,
    CharType,
    DataType,
    DateType,
    DecimalType,
    DoubleType,
    FloatType,
    IntegerType,
    LongType,
    ShortType,
    StringType,
    TimestampNTZType,
    TimestampType,
    VarcharType,
)
from tqdm import tqdm
from visions import VisionsTypeset

from data_profiling.config import Settings
from data_profiling.model.summarizer import BaseSummarizer
from data_profiling.utils.dataframe import sort_column_names

NUMERIC_TYPES = (
    ByteType,
    ShortType,
    IntegerType,
    LongType,
    FloatType,
    DoubleType,
    DecimalType,
)
DATETIME_TYPES = (DateType, TimestampType, TimestampNTZType)
CATEGORICAL_TYPES = (StringType, CharType, VarcharType, ArrayType)


def spark_vtype(data_type: DataType) -> str:
    """Map a Spark data type onto a profiling variable type.

    Matching is done on the type class rather than on ``simpleString()``:
    parameterised types (``decimal(10,0)``, ``char(10)``, ``array<int>``,
    ``struct<a:string>``, ``map<string,int>``) never compare equal to a fixed
    string, so a string-keyed lookup can only ever cover the unparameterised
    scalars.

    Types with no profiling support (struct, map, binary, void, variant,
    interval, ...) fall back to ``Unsupported``, which yields counts and
    missing-value statistics instead of aborting the whole report.
    """
    if isinstance(data_type, BooleanType):
        return "Boolean"
    if isinstance(data_type, NUMERIC_TYPES):
        return "Numeric"
    if isinstance(data_type, DATETIME_TYPES):
        return "DateTime"
    if isinstance(data_type, CATEGORICAL_TYPES):
        return "Categorical"
    return "Unsupported"


def spark_describe_1d(
    config: Settings,
    series: DataFrame,
    summarizer: BaseSummarizer,
    typeset: VisionsTypeset,
) -> dict:
    """Describe a series (infer the variable type, then calculate type-specific values).

    Args:
        config: report Settings object
        series: The Series to describe.
        summarizer: Summarizer object
        typeset: Typeset

    Returns:
        A Series containing calculated series description values.
    """

    # Make sure pd.NA is not in the series
    series = series.fillna(np.nan)

    # get `infer_dtypes` (bool) from config
    if config.infer_dtypes:
        # Infer variable types
        vtype = typeset.infer_type(series)
        series = typeset.cast_to_inferred(series)
    else:
        # Detect variable types from pandas dataframe (df.dtypes).
        # [new dtypes, changed using `astype` function are now considered]

        vtype = spark_vtype(series.schema[0].dataType)

    return summarizer.summarize(config, series, dtype=vtype)


def get_series_descriptions_spark(
    config: Settings,
    df: DataFrame,
    summarizer: BaseSummarizer,
    typeset: VisionsTypeset,
    pbar: tqdm,
) -> dict:
    """
    Compute series descriptions/statistics for a Spark DataFrame.

    Returns: A dict with the series descriptions for each column of a Dataset
    """

    def describe_column(name: str) -> Tuple[str, dict]:
        """Process a single Spark column using Spark's execution model."""
        description = spark_describe_1d(config, df.select(name), summarizer, typeset)
        pbar.set_postfix_str(f"Describe variable: {name}")
        pbar.update()

        # Clean up Spark-specific metadata
        description.pop(
            "value_counts", None
        )  # Use `.pop()` with default to avoid KeyError
        return name, description

    series_description = {col: describe_column(col)[1] for col in df.columns}

    # Sort and return descriptions
    return sort_column_names(series_description, config.sort)
