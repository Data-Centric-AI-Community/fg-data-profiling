"""Spark text description: type detection and parity with the pandas backend."""
import pandas as pd
import pytest

from data_profiling import ProfileReport
from data_profiling.config import SparkSettings
from data_profiling.model.pandas.describe_categorical_pandas import (
    length_summary_vc,
    unicode_summary_vc,
    word_summary_vc,
)
from data_profiling.model.spark.describe_text_spark import describe_text_1d_spark
from data_profiling.model.spark.summary_spark import spark_string_vtype

CORPUS = [
    "The quick brown fox jumps over the lazy dog.",
    "Synthetic data, real problems: (near-)duplicates everywhere!",
    "Élysée Palace — Paris; café au lait.",
    "Emoji test 😀 and CJK 数据 质量",
    "  leading and trailing whitespace  ",
    "",
    "The quick brown fox jumps over the lazy dog.",
    None,
]


@pytest.fixture
def text_df(spark_session):
    return spark_session.createDataFrame([(v,) for v in CORPUS], "text string")


def _value_counts(values):
    return pd.Series([v for v in values if v is not None]).value_counts()


def test_string_vtype_low_cardinality_is_categorical(spark_session):
    df = spark_session.createDataFrame(
        [(v,) for v in ["a", "b", "a", "b", "a", "b", None]], "cat string"
    )
    assert spark_string_vtype(SparkSettings(), df) == "Categorical"


def test_string_vtype_high_cardinality_is_text(spark_session):
    df = spark_session.createDataFrame(
        [(f"document number {i}",) for i in range(100)], "doc string"
    )
    assert spark_string_vtype(SparkSettings(), df) == "Text"


def test_string_vtype_all_null_stays_categorical(spark_session):
    df = spark_session.createDataFrame([(None,), (None,)], "empty string")
    assert spark_string_vtype(SparkSettings(), df) == "Categorical"


def test_describe_text_spark_matches_pandas(text_df):
    config = SparkSettings()
    _, _, summary = describe_text_1d_spark(config, text_df, {"count": 7})

    vc = _value_counts(CORPUS)

    expected_length = length_summary_vc(vc)
    for key in ("max_length", "min_length", "mean_length", "median_length"):
        assert summary[key] == pytest.approx(expected_length[key])
    assert summary["length_histogram"].to_dict() == (
        expected_length["length_histogram"].to_dict()
    )
    assert "histogram_length" in summary

    expected_unicode = unicode_summary_vc(vc)
    assert summary["n_characters"] == expected_unicode["n_characters"]
    assert summary["n_characters_distinct"] == expected_unicode["n_characters_distinct"]
    assert summary["character_counts"].to_dict() == (
        expected_unicode["character_counts"].to_dict()
    )
    assert summary["script_counts"].to_dict() == (
        expected_unicode["script_counts"].to_dict()
    )
    # code points, not UTF-16 surrogate halves
    assert summary["character_counts"]["😀"] == 1

    expected_words = word_summary_vc(vc)["word_counts"]
    # pandas also counts the empty token left by punctuation-only words
    expected_words = expected_words[expected_words.index != ""]
    assert summary["word_counts"].to_dict() == expected_words.to_dict()

    assert list(summary["first_rows"]) == CORPUS[:5]


def test_describe_text_spark_stop_words_and_limit(text_df):
    config = SparkSettings()
    config.vars.cat.stop_words = ["The", "and"]
    config.vars.text.spark_words_limit = 3
    _, _, summary = describe_text_1d_spark(config, text_df, {"count": 7})

    words = summary["word_counts"]
    assert len(words) == 3
    assert "the" not in words.index
    assert "and" not in words.index
    assert words.is_monotonic_decreasing


def test_describe_text_spark_flags(text_df):
    config = SparkSettings()
    config.vars.text.length = False
    config.vars.text.characters = False
    config.vars.text.words = False
    config.vars.text.redact = True
    _, _, summary = describe_text_1d_spark(config, text_df, {"count": 7})

    assert "first_rows" not in summary
    assert "max_length" not in summary
    assert "character_counts" not in summary
    assert "word_counts" not in summary


def test_describe_text_spark_all_null(spark_session):
    df = spark_session.createDataFrame([(None,), (None,)], "text string")
    _, _, summary = describe_text_1d_spark(SparkSettings(), df, {"count": 0})
    assert "max_length" not in summary
    assert "word_counts" not in summary


def test_report_spark_text_column(spark_session, tmp_path):
    rows = [(i, f"review {i}: the product works, rating {i % 5}") for i in range(60)]
    df = spark_session.createDataFrame(rows, "id int, review string")

    report = ProfileReport(df, title="Spark text")
    description = report.get_description()
    review = description.variables["review"]

    assert review["type"] == "Text"
    assert review["max_length"] > 0
    assert "word_counts" in review
    assert "character_counts" in review

    output = tmp_path / "spark_text.html"
    report.to_file(output, silent=True)
    html = output.read_text()
    assert ">Words<" in html
    assert ">Characters<" in html
