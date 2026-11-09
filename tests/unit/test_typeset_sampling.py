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


@pytest.mark.parametrize("name", list(_long_columns()))
def test_sampling_does_not_change_inference(name, monkeypatch):
    series = _long_columns()[name]
    sampled = ProfilingTypeSet(Settings()).infer_type(series.copy())
    monkeypatch.setattr(typeset_relations, "SAMPLE_SIZE", 10**9)
    full = ProfilingTypeSet(Settings()).infer_type(series.copy())
    assert str(sampled) == str(full)


@pytest.mark.parametrize(
    "values, bad, not_expected",
    [
        (lambda i: f"{i * 0.5:.1f}", "abc", "Numeric"),
        (lambda i: f"2024-01-{i % 28 + 1:02d}", "not a date", "DateTime"),
        (lambda i: "yes" if i % 2 else "no", "maybe", "Boolean"),
    ],
)
def test_one_bad_value_outside_the_sample_still_rejects(
    typeset, values, bad, not_expected
):
    series = pd.Series([values(i) for i in range(N)])
    # Position 1 is not one of the evenly spaced sample positions.
    series.iloc[1] = bad
    assert str(typeset.infer_type(series)) != not_expected


CAST_CASES = {
    "ints": ["1", "2", "3"],
    "ints_nulls": ["1", None, "3", np.nan],
    "floats_nulls": ["1.5", None, "-2e3"],
    "bigint_nulls": ["9007199254740993", None, "1"],
    "dates": ["2024-01-01", "2024-02-03"],
    "dates_nulls": ["2024-01-01", None, "2024-02-03 10:00"],
    "dates_missing_markers": ["2024-01-01", "NaT", None, ""],
    "dates_tz": ["2024-01-01T00:00:00+00:00", None, "2024-02-03T05:00:00+00:00"],
}


@pytest.mark.parametrize("case", list(CAST_CASES))
@pytest.mark.parametrize(
    "index", [None, [7, 7, 7, 7], ["r0", "r1", "r2", "r3"]], ids=["range", "dup", "str"]
)
def test_cached_cast_equals_fresh_cast(case, index):
    values = CAST_CASES[case]
    series = pd.Series(
        values,
        index=None if index is None else index[: len(values)],
        name="col",
        dtype=object,
    )
    checks = [
        (
            lambda s, st: typeset_relations.string_is_numeric(s, st, Settings()),
            typeset_relations.string_to_numeric,
        ),
        (typeset_relations.string_is_datetime, typeset_relations.string_to_datetime),
    ]
    for check, cast in checks:
        state: dict = {}
        if not check(series, state):
            continue
        pd.testing.assert_series_equal(cast(series, state), cast(series, {}))


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
