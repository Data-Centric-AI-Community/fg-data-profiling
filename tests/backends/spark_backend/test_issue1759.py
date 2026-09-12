"""
Test for issue 1759:
https://github.com/ydataai/ydata-profiling/issues/1759

Rendering the HTML report for a Spark DataFrame with a date/timestamp column
used to raise ``KeyError: 'n_invalid_dates'``: the Spark date describe never set
the ``invalid_dates`` / ``n_invalid_dates`` / ``p_invalid_dates`` keys that the
shared (backend-agnostic) date renderer reads.
"""

import datetime

from pyspark.sql import types as T

from data_profiling import ProfileReport


def test_spark_date_column_report_does_not_keyerror(test_output_dir, spark_session):
    spark = spark_session

    schema = T.StructType(
        [
            T.StructField("id", T.IntegerType(), True),
            T.StructField("event_date", T.DateType(), True),
        ]
    )
    data = [
        (i, datetime.date(2020, 1, 1) + datetime.timedelta(days=i)) for i in range(30)
    ]
    test_df = spark.createDataFrame(data, schema=schema)

    profile = ProfileReport(test_df, title="spark_date", explorative=True)
    output_file = test_output_dir / "spark_date.html"
    profile.to_file(output_file)

    assert output_file.exists()
