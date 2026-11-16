"""Sample-based rejection and per-step caching in type inference.

Neither may change a result: the sample only rejects checks that must hold for
every value, and cached results are reused only for the series they were
computed on.
"""
import numpy as np
import pandas as pd
import pytest

from data_profiling.config import Settings
from data_profiling.model import typeset_relations
from data_profiling.model.typeset import ProfilingTypeSet

N = 10_000


@pytest.fixture(scope="module")
def typeset():
    return ProfilingTypeSet(Settings())


def _long_columns():
    rng = np.random.default_rng(0)
    dates = pd.Timestamp("2024-01-01") + pd.to_timedelta(
        rng.integers(0, 400, N), unit="D"
    )
    return {
        "numeric_str": pd.Series(rng.normal(size=N).round(3).astype(str)),
        "int_str_low": pd.Series(rng.integers(0, 4, N).astype(str)),
        "int_str_mid": pd.Series(rng.integers(0, 30, N).astype(str)),
        "date_str": pd.Series(dates.strftime("%Y-%m-%d")),
        "year_str": pd.Series(rng.integers(2000, 2030, N).astype(str)),
        "yes_no": pd.Series(rng.choice(["yes", "no"], N)),
        "words": pd.Series(rng.choice(["alpha", "beta", "gamma"], N)),
        "ids": pd.Series([f"C{i:06d}" for i in range(N)]),
        "ints": pd.Series(rng.integers(0, 1000, N)),
    }


def test_sampling_does_not_change_inference(monkeypatch):
    columns = _long_columns()
    sampled = {
        name: str(ProfilingTypeSet(Settings()).infer_type(s.copy()))
        for name, s in columns.items()
    }
    monkeypatch.setattr(typeset_relations, "SAMPLE_SIZE", 10**9)
    full = {
        name: str(ProfilingTypeSet(Settings()).infer_type(s.copy()))
        for name, s in columns.items()
    }
    assert sampled == full


def test_one_bad_value_outside_the_sample_still_rejects(typeset):
    cases = {
        "Numeric": (lambda i: f"{i * 0.5:.1f}", "abc"),
        "DateTime": (lambda i: f"2024-01-{i % 28 + 1:02d}", "not a date"),
        "Boolean": (lambda i: "yes" if i % 2 else "no", "maybe"),
    }
    for rejected, (values, bad) in cases.items():
        series = pd.Series([values(i) for i in range(N)])
        # Position 1 is not one of the evenly spaced sample positions.
        series.iloc[1] = bad
        assert str(typeset.infer_type(series)) != rejected, rejected


# Number cases have more distinct values than vars.num.low_categorical_threshold
# (5 by default), so string_is_numeric accepts them on the default config.
CAST_CASES = {
    "ints": ["1", "2", "3", "4", "5", "6"],
    "ints_nulls": ["1", None, "3", np.nan, "4", "5", "6", "7"],
    "floats_nulls": ["1.5", None, "-2e3", "0.25", "7", "8.5", "9"],
    "bigint_nulls": ["9007199254740993", None, "1", "2", "3", "4", "5"],
    "dates": ["2024-01-01", "2024-02-03"],
    "dates_nulls": ["2024-01-01", None, "2024-02-03 10:00"],
    "dates_missing_markers": ["2024-01-01", "NaT", None, ""],
    "dates_tz": ["2024-01-01T00:00:00+00:00", None, "2024-02-03T05:00:00+00:00"],
}


def test_cached_cast_equals_fresh_cast():
    config = Settings()
    checks = [
        (
            lambda s, st: typeset_relations.string_is_numeric(s, st, config),
            typeset_relations.string_to_numeric,
        ),
        (typeset_relations.string_is_datetime, typeset_relations.string_to_datetime),
    ]
    n_compared = 0
    for values in CAST_CASES.values():
        n = len(values)
        for index in [None, [7] * n, [f"r{i}" for i in range(n)]]:
            series = pd.Series(values, index=index, name="col", dtype=object)
            for check, cast in checks:
                state: dict = {}
                if check(series, state):
                    pd.testing.assert_series_equal(
                        cast(series, state), cast(series, {})
                    )
                    n_compared += 1
    # Every case passes exactly one of the two checks, for each index.
    assert n_compared == 3 * len(CAST_CASES)


def test_inferred_cast_equals_fresh_cast(typeset):
    # Through the whole traversal, on the default config, for short columns and
    # for long ones where the sample and the cache work together.
    def with_missing(values):
        return [None if i % 7 == 3 else v for i, v in enumerate(values)]

    columns = {
        "numbers_short": (
            ["1.5", None, "-2", "3", "4", "5", "6.25"] * 3,
            "Numeric",
            typeset_relations.string_to_numeric,
        ),
        "numbers_long": (
            with_missing([f"{i * 0.5}" for i in range(N)]),
            "Numeric",
            typeset_relations.string_to_numeric,
        ),
        "dates_short": (
            ["2024-01-01", None, "2024-02-03 10:00", "NaT", "2024-03-05"] * 4,
            "DateTime",
            typeset_relations.string_to_datetime,
        ),
        "dates_long": (
            with_missing([f"2024-01-{i % 28 + 1:02d}" for i in range(N)]),
            "DateTime",
            typeset_relations.string_to_datetime,
        ),
    }
    for name, (values, expected_type, cast) in columns.items():
        series = pd.Series(values, name=name, dtype=object)
        vtype, result = typeset.infer_and_cast(series)
        assert str(vtype) == expected_type, name
        pd.testing.assert_series_equal(result, cast(series, {}), obj=name)


def test_step_cache_is_cleared_after_a_transform(typeset):
    # Six distinct strings that are all the number 1: Text -> Numeric (more than
    # 5 distinct strings), then Numeric -> Categorical (1 distinct number). A
    # distinct count cached at the Text step must not be reused.
    series = pd.Series(["1", "1.0", "01", "1.00", "001", "1.000"] * 20)
    assert [s["chosen"] for s in typeset.explain_type(series)] == [
        "Text",
        "Numeric",
        "Categorical",
    ]
    assert str(typeset.infer_type(series)) == "Categorical"
