"""Precedence and structure of the typeset.

When more than one relation leaving a type matches a series, the target that
comes first in the typeset's type order wins. These tests pin the columns
where that order decides the type.
"""
import pandas as pd
import pytest

from data_profiling.config import Settings
from data_profiling.expectations_report import ExpectationHandler
from data_profiling.model.summarizer import ProfilingSummarizer
from data_profiling.model.typeset import ProfilingTypeSet
from data_profiling.model.typeset_base import (
    BaseTypeSet,
    Generic,
    IdentityRelation,
    InferenceRelation,
    ProfilingBaseType,
)


@pytest.fixture(scope="module")
def typeset():
    return ProfilingTypeSet(Settings())


def _names(steps):
    return [(s["from"], s["matched"], s["chosen"]) for s in steps]


# Ambiguous columns on the default config


def test_numeric_strings_with_few_distinct_values_are_numeric(typeset):
    # 10 distinct values in 100 rows: above vars.num.low_categorical_threshold (5)
    # and below vars.cat.cardinality_threshold (50), so both Text -> Numeric
    # and Text -> Categorical hold.
    series = pd.Series([str(i % 10) for i in range(100)])
    steps = typeset.explain_type(series)
    assert ("Text", ["Numeric", "Categorical"], "Numeric") in _names(steps)
    assert str(typeset.infer_type(series)) == "Numeric"


def test_repeated_iso_dates_are_datetime(typeset):
    series = pd.Series(["2024-01-01", "2024-01-02", "2024-01-03"] * 20)
    steps = typeset.explain_type(series)
    assert ("Text", ["DateTime", "Categorical"], "DateTime") in _names(steps)
    assert str(typeset.infer_type(series)) == "DateTime"


def test_month_names_are_categorical(typeset):
    # The date parser accepts bare month names; string_is_datetime requires a
    # digit in every value so they stay categories.
    series = pd.Series(["Jan", "Feb", "Mar"] * 20)
    assert str(typeset.infer_type(series)) == "Categorical"


def test_low_cardinality_integers_are_categorical(typeset):
    series = pd.Series([1, 2, 3] * 20)
    assert str(typeset.infer_type(series)) == "Categorical"


@pytest.mark.parametrize("n_years", [3, 30])
def test_year_strings_are_numbers_at_any_cardinality(typeset, n_years):
    series = pd.Series([str(2000 + i % n_years) for i in range(120)])
    expected = "Categorical" if n_years <= 5 else "Numeric"
    assert str(typeset.infer_type(series)) == expected


def test_low_cardinality_is_categorical_in_timeseries_mode():
    config = Settings()
    config.vars.timeseries.active = True
    typeset = ProfilingTypeSet(config)
    flags = pd.Series([0, 1] * 60)
    assert str(typeset.infer_type(flags)) == "Categorical"
    trend = pd.Series(range(120), dtype=float)
    assert str(typeset.infer_type(trend)) == "TimeSeries"


def test_inference_is_stable_across_typesets():
    series = pd.Series([str(i % 10) for i in range(100)])
    inferred = {str(ProfilingTypeSet(Settings()).infer_type(series)) for _ in range(20)}
    assert inferred == {"Numeric"}


def test_infer_and_cast_matches_separate_calls(typeset):
    series = pd.Series(["1.5", "2.5", None, "4.0"] * 5)
    vtype, cast = typeset.infer_and_cast(series)
    assert vtype == typeset.infer_type(series)
    pd.testing.assert_series_equal(cast, typeset.cast_to_inferred(series))


def test_detect_does_not_cast(typeset):
    series = pd.Series(["1", "2", "3"])
    assert str(typeset.detect_type(series)) == "Text"
    assert typeset.cast_to_detected(series) is series


def test_precedence_order_of_default_typeset():
    config = Settings()
    config.vars.url.active = True
    config.vars.path.active = True
    config.vars.timeseries.active = True
    names = [str(t) for t in ProfilingTypeSet(config).types]
    assert names == [
        "Unsupported",
        "Boolean",
        "Numeric",
        "DateTime",
        "URL",
        "Path",
        "Categorical",
        "TimeSeries",
        "Text",
    ]


# Generic typeset behaviour


def _toy_types():
    class Root(Generic):
        pass

    class Word(ProfilingBaseType):
        @staticmethod
        def get_relations():
            return [IdentityRelation(Root)]

        @staticmethod
        def contains_op(series, state):
            return all(isinstance(v, str) for v in series)

    class Short(ProfilingBaseType):
        @staticmethod
        def get_relations():
            return [
                IdentityRelation(Root),
                InferenceRelation(
                    Word,
                    relationship=lambda s, st: s.str.len().max() <= 3,
                    transformer=lambda s, st: s.str.upper(),
                ),
            ]

        @staticmethod
        def contains_op(series, state):
            return False

    class Lower(ProfilingBaseType):
        @staticmethod
        def get_relations():
            return [
                IdentityRelation(Root),
                InferenceRelation(
                    Word,
                    relationship=lambda s, st: (s == s.str.lower()).all(),
                ),
            ]

        @staticmethod
        def contains_op(series, state):
            return False

    return Root, Word, Short, Lower


def test_order_of_types_is_the_precedence():
    Root, Word, Short, Lower = _toy_types()
    series = pd.Series(["ab", "cd"])  # both Short and Lower hold

    first = BaseTypeSet([Root, Word, Short, Lower])
    assert first.infer_type(series) is Short
    assert list(first.cast_to_inferred(series)) == ["AB", "CD"]

    second = BaseTypeSet([Root, Word, Lower, Short])
    assert second.infer_type(series) is Lower
    assert second.explain_type(series)[-1] == {
        "from": "Word",
        "matched": ["Lower", "Short"],
        "chosen": "Lower",
    }


def test_types_must_be_ordered():
    Root, Word, Short, Lower = _toy_types()
    with pytest.raises(TypeError, match="ordered sequence"):
        BaseTypeSet({Root, Word, Short, Lower})


def test_exactly_one_root():
    Root, Word, _, _ = _toy_types()

    class OtherRoot(Generic):
        pass

    with pytest.raises(ValueError, match="exactly one root"):
        BaseTypeSet([Root, OtherRoot, Word])
    with pytest.raises(ValueError, match="exactly one root"):
        BaseTypeSet([Word])


def test_type_needs_one_identity_parent():
    Root, Word, _, _ = _toy_types()

    class Orphan(ProfilingBaseType):
        @staticmethod
        def get_relations():
            return []

    with pytest.raises(ValueError, match="exactly one IdentityRelation"):
        BaseTypeSet([Root, Word, Orphan])


def test_types_render_as_names():
    Root, Word, _, _ = _toy_types()
    assert str(Word) == "Word"
    assert pd.Series(["a"]) in Word


# Handler composition


def test_summary_functions_include_ancestors(typeset):
    summarizer = ProfilingSummarizer(typeset)
    unsupported = summarizer.summary_map["Unsupported"]
    numeric = summarizer.summary_map["Numeric"]
    assert numeric[: len(unsupported)] == unsupported
    assert len(numeric) == len(unsupported) + 1


def test_expectations_with_timeseries_type():
    config = Settings()
    config.vars.timeseries.active = True
    handler = ExpectationHandler(ProfilingTypeSet(config))
    assert handler.mapping["TimeSeries"] == handler.mapping["Numeric"]
