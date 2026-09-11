"""
Test for issue 1429:
https://github.com/Data-Centric-AI-Community/data-profiling/issues/1429
"""
from typing import List, Optional, Tuple

import numpy as np
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import types as T

from data_profiling import ProfileReport
from data_profiling.config import SparkSettings
from data_profiling.model.spark.describe_counts_spark import describe_counts_spark
from data_profiling.model.spark.describe_generic_spark import describe_generic_spark
from data_profiling.model.spark.describe_numeric_spark import (
    describe_numeric_1d_spark,
    numeric_stats_spark,
)
from data_profiling.model.spark.describe_supported_spark import describe_supported_spark

RowType = Tuple[
    Optional[str],
    Optional[float],
    Optional[int],
    Optional[bool],
    Optional[float],
    Optional[str],
]

config = SparkSettings()


def create_test_df(spark: SparkSession) -> DataFrame:
    schema = T.StructType(
        [
            T.StructField("category", T.StringType(), True),
            T.StructField("double", T.DoubleType(), True),
            T.StructField("int", T.IntegerType(), True),
            T.StructField("boolean", T.BooleanType(), True),
            T.StructField("null_double", T.DoubleType(), True),
            T.StructField("null_string", T.StringType(), True),
        ]
    )

    data: List[RowType] = [
        (f"test_{num + 1}", float(num), int(num), True, None, None)
        for num in range(205)
    ]

    # Adding dupes
    data.extend([("test_1", float(1), int(1), False, None, None) for _ in range(205)])

    # Adding nulls
    data.extend([(None, None, None, None, None, None) for _ in range(100)])

    return spark.createDataFrame(data, schema=schema)


def test_describe_numeric_spark(spark_session):
    test_df = create_test_df(spark_session)

    numeric_stats = numeric_stats_spark(df=test_df.select("double"), summary={})

    for _, value in numeric_stats.items():
        assert value is not None


def test_describe_numeric_1d_spark_for_null_column_edge_case(
    spark_session, test_output_dir
):
    spark = spark_session
    test_df = create_test_df(spark)

    _, _, summary = describe_counts_spark(
        config=config, series=test_df.select("null_double"), summary={}
    )

    _, _, summary = describe_generic_spark(
        config=config, df=test_df.select("null_double"), summary=summary
    )

    _, _, summary = describe_supported_spark(
        config=config, series=test_df.select("null_double"), summary=summary
    )

    _, _, summary = describe_numeric_1d_spark(
        config=config, df=test_df.select("null_double"), summary=summary
    )

    assert summary["iqr"] is np.nan
    assert summary["mad"] is np.nan
    assert summary["cv"] is np.nan
    assert summary["mean"] is None
    assert summary["histogram"] == []

    profile = ProfileReport(test_df, title="1429_edge_case.html", config=config)

    output_file = test_output_dir / "1429_edge_case.html"

    profile.to_file(output_file)

    assert output_file.exists()


def test_describe_counts_spark(spark_session):
    test_df = create_test_df(spark_session)

    _, _, summary = describe_counts_spark(
        config=config, series=test_df.select("category"), summary={}
    )

    assert summary["value_counts_without_nan"].loc["test_1"] == 206

    _, _, summary = describe_counts_spark(
        config=config, series=test_df.select("double"), summary={}
    )

    assert summary["value_counts_without_nan"].loc[float(1)] == 206

    _, _, summary = describe_counts_spark(
        config=config, series=test_df.select("int"), summary={}
    )

    assert summary["value_counts_without_nan"].loc[int(1)] == 206

    _, _, summary = describe_counts_spark(
        config=config, series=test_df.select("boolean"), summary={}
    )

    assert summary["value_counts_without_nan"].loc[True] == 205

    _, _, summary = describe_counts_spark(
        config=config, series=test_df.select("null_double"), summary={}
    )

    assert summary["value_counts_without_nan"].size == 0
