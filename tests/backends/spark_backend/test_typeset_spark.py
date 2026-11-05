"""Spark type inference: same types as the pandas typeset on the same data."""
import numpy as np
import pandas as pd
import pytest

from data_profiling.config import Settings, SparkSettings
from data_profiling.model.spark.typeset_spark import cast_frame, infer_spark_type
from data_profiling.model.typeset import ProfilingTypeSet

N = 60


def _cycle(values):
    return [values[i % len(values)] for i in range(N)]


STRING_COLUMNS = {
    "numbers": [str(i * 0.5) for i in range(N)],
    "numbers_with_missing": [str(i) for i in range(N - 5)] + [None] * 5,
    "few_number_strings": _cycle(["1", "2", "3"]),
    "python_number_spellings": _cycle(["1_000", "inf", "-nan", "2.5"])
    + [str(i) for i in range(N)],
    "java_suffix": _cycle(["1.0d", "2.0", "3.5"]),
    "iso_dates": [f"2024-01-{i % 28 + 1:02d}" for i in range(N)],
    "dates_with_markers": _cycle(["2024-01-01", "2024/02/15", "NaT", ""]),
    "years": [str(2000 + i % 30) for i in range(N)],
    "month_names": _cycle(["Jan", "Feb", "Mar"]),
    "yes_no": _cycle(["Yes", "no", None]),
    "words": _cycle(["alpha", "beta", "gamma"]),
    "free_text": [f"text {i}" for i in range(N)],
    "no_values": [None] * N,
    # Known difference: pandas reads day-first dates with dateutil, Spark not.
    "day_first_dates": _cycle(["20/04/2014", "13/10/2016"]),
}
NUMERIC_COLUMNS = {"few_ints": _cycle([1, 2, 3]), "floats": [i / 7 for i in range(N)]}
EXPECTED_DIFFERENCES = {"day_first_dates": ("DateTime", "Categorical")}


def _pandas_type(typeset, values):
    series = pd.Series(values, dtype=object if values[0] is None else None)
    return str(typeset.infer_type(series.fillna(np.nan).infer_objects()))


@pytest.mark.parametrize(
    "config", [Settings(), SparkSettings()], ids=["default", "spark"]
)
def test_types_match_pandas(spark_session, config):
    typeset = ProfilingTypeSet(config)
    differences = {}
    for columns, schema, start in [
        (STRING_COLUMNS, "c string", "Text"),
        (NUMERIC_COLUMNS, "c double", "Numeric"),
    ]:
        for name, values in columns.items():
            cast = float if start == "Numeric" else (lambda v: v)
            sdf = spark_session.createDataFrame([(cast(v),) for v in values], schema)
            spark_type, _ = infer_spark_type(config, sdf, typeset, start)
            pandas_type = _pandas_type(typeset, values)
            if spark_type != pandas_type:
                differences[name] = (pandas_type, spark_type)
    assert differences == EXPECTED_DIFFERENCES


def test_inferred_columns_are_cast(spark_session):
    rows = [
        (str(i * 1.5), f"2024-01-{i % 28 + 1:02d}", "yes" if i % 2 else "no", i % 3)
        for i in range(N)
    ]
    sdf = spark_session.createDataFrame(
        rows, "`a num` string, d string, f string, k long"
    )
    types = {"a num": "Numeric", "d": "DateTime", "f": "Boolean", "k": "Categorical"}
    cast = cast_frame(Settings(), sdf, types)
    assert [(f.name, f.dataType.simpleString()) for f in cast.schema.fields] == [
        ("a num", "double"),
        ("d", "timestamp"),
        ("f", "boolean"),
        ("k", "string"),
    ]
