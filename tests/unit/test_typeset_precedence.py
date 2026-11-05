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

# Columns where more than one type relation matches, with the type the
# precedence gives them on the default config.
AMBIGUOUS_COLUMNS = {
    # Text -> Numeric and Text -> Categorical both hold: Numeric wins.
    "numeric_strings_10_distinct": ([str(i % 10) for i in range(100)], "Numeric"),
    # Text -> DateTime and Text -> Categorical both hold: DateTime wins.
    "repeated_iso_dates": (["2024-01-01", "2024-01-02", "2024-01-03"] * 20, "DateTime"),
    # Strings that parse as missing do not prevent a date column.
    "dates_with_NaT": (["2024-01-01", "2024-02-15", "NaT"] * 20, "DateTime"),
    "dates_with_empty": (["2024-01-01", "2024-02-15", ""] * 20, "DateTime"),
    # Month names and plain numbers parse as dates but are not dates.
    "month_names": (["Jan", "Feb", "Mar"] * 20, "Categorical"),
    "month_names_with_NaT": (["Jan", "NaT"] * 30, "Categorical"),
    "years_with_NaT": (["2024", "NaT"] * 30, "Categorical"),
    # Years are typed like other integer strings at any cardinality.
    "years_3_distinct": ([str(2000 + i % 3) for i in range(120)], "Categorical"),
    "years_30_distinct": ([str(2000 + i % 30) for i in range(120)], "Numeric"),
    "few_ints": ([1, 2, 3] * 20, "Categorical"),
}


@pytest.fixture(scope="module")
def typeset():
    return ProfilingTypeSet(Settings())


def test_ambiguous_columns(typeset):
    inferred = {
        name: str(typeset.infer_type(pd.Series(values)))
        for name, (values, _) in AMBIGUOUS_COLUMNS.items()
    }
    expected = {name: type_ for name, (_, type_) in AMBIGUOUS_COLUMNS.items()}
    assert inferred == expected


def test_precedence_and_timeseries_mode():
    config = Settings()
    config.vars.url.active = True
    config.vars.path.active = True
    config.vars.timeseries.active = True
    typeset = ProfilingTypeSet(config)
    assert [str(t) for t in typeset.types] == [
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
    # Few distinct numbers are categorical in every mode; trends are time series.
    assert str(typeset.infer_type(pd.Series([0, 1] * 60))) == "Categorical"
    assert str(typeset.infer_type(pd.Series(range(120), dtype=float))) == "TimeSeries"


def test_traversal_api(typeset):
    series = pd.Series([str(i % 10) for i in range(100)])
    steps = typeset.explain_type(series)
    assert steps[1] == {
        "from": "Text",
        "matched": ["Numeric", "Categorical"],
        "chosen": "Numeric",
    }
    assert {str(ProfilingTypeSet(Settings()).infer_type(series)) for _ in range(5)} == {
        "Numeric"
    }

    vtype, cast = typeset.infer_and_cast(series)
    assert vtype == typeset.infer_type(series)
    pd.testing.assert_series_equal(cast, typeset.cast_to_inferred(series))

    dates = pd.Series(["2024-01-01", "2024-02-15", "NaT"] * 20)
    assert typeset.cast_to_inferred(dates).isna().sum() == 20

    strings = pd.Series(["1", "2", "3"])
    assert str(typeset.detect_type(strings)) == "Text"
    assert typeset.cast_to_detected(strings) is strings


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


def test_explain_does_not_change_the_inference_path():
    # The chosen relation caches a mode in nested state; a later candidate
    # overwrites it and returns False. The trace must still follow the path
    # that inference takes (Root -> Word -> Tagged -> Upper).
    class Root(Generic):
        pass

    class Word(ProfilingBaseType):
        @staticmethod
        def get_relations():
            return [IdentityRelation(Root)]

        @staticmethod
        def contains_op(series, state):
            return all(isinstance(v, str) for v in series)

    def choose(series, state):
        state.setdefault("cache", {})["mode"] = "upper"
        return True

    def overwrite(series, state):
        state.setdefault("cache", {})["mode"] = "lower"
        return False

    class Tagged(ProfilingBaseType):
        @staticmethod
        def get_relations():
            return [
                IdentityRelation(Root),
                InferenceRelation(Word, relationship=choose),
            ]

        @staticmethod
        def contains_op(series, state):
            return False

    class Other(ProfilingBaseType):
        @staticmethod
        def get_relations():
            return [
                IdentityRelation(Root),
                InferenceRelation(Word, relationship=overwrite),
            ]

        @staticmethod
        def contains_op(series, state):
            return False

    class Upper(ProfilingBaseType):
        @staticmethod
        def get_relations():
            return [
                IdentityRelation(Root),
                InferenceRelation(
                    Tagged,
                    relationship=lambda s, st: st["cache"]["mode"] == "upper",
                ),
            ]

        @staticmethod
        def contains_op(series, state):
            return False

    typeset = BaseTypeSet([Root, Word, Tagged, Other, Upper])
    series = pd.Series(["ab", "cd"])
    _, path, _ = typeset.infer(series)
    steps = typeset.explain_type(series)
    assert [str(t) for t in path] == ["Root", "Word", "Tagged", "Upper"]
    assert [s["chosen"] for s in steps] == [str(t) for t in path[1:]]


def test_invalid_typesets():
    Root, Word, Short, Lower = _toy_types()

    class OtherRoot(Generic):
        pass

    class Orphan(ProfilingBaseType):
        @staticmethod
        def get_relations():
            return []

    with pytest.raises(TypeError, match="ordered sequence"):
        BaseTypeSet({Root, Word, Short, Lower})
    with pytest.raises(ValueError, match="exactly one root"):
        BaseTypeSet([Root, OtherRoot, Word])
    with pytest.raises(ValueError, match="exactly one root"):
        BaseTypeSet([Word])
    with pytest.raises(ValueError, match="exactly one IdentityRelation"):
        BaseTypeSet([Root, Word, Orphan])
    assert str(Word) == "Word"
    assert pd.Series(["a"]) in Word


# Handler composition


def test_handlers_include_ancestor_functions(typeset):
    summary_map = ProfilingSummarizer(typeset).summary_map
    unsupported = summary_map["Unsupported"]
    assert summary_map["Numeric"][: len(unsupported)] == unsupported
    assert len(summary_map["Numeric"]) == len(unsupported) + 1

    config = Settings()
    config.vars.timeseries.active = True
    handler = ExpectationHandler(ProfilingTypeSet(config))
    assert handler.mapping["TimeSeries"] == handler.mapping["Numeric"]
