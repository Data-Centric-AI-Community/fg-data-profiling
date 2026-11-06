import pandas as pd
import pytest

from data_profiling.model.pandas.describe_categorical_pandas import word_summary_vc

value_counts_w_words = pd.Series(index=["The dog", "is hungry"], data=[2, 1])


def test_word_summary_vc():
    assert (
        word_summary_vc(vc=value_counts_w_words)["word_counts"].to_dict()
        == pd.Series(index=["the", "dog", "is", "hungry"], data=[2, 2, 1, 1]).to_dict()
    )


@pytest.mark.parametrize("stop_words", [["The"], ["the", "a"]])
def test_word_summary_vc_with_stop_words(stop_words):
    assert (
        word_summary_vc(vc=value_counts_w_words, stop_words=stop_words)[
            "word_counts"
        ].to_dict()
        == pd.Series(index=["dog", "is", "hungry"], data=[2, 1, 1]).to_dict()
    )


def test_word_summary_vc_ignores_punctuation_only_tokens():
    value_counts = pd.Series(index=["The dog - hungry", "a cat ... !"], data=[2, 1])
    assert (
        word_summary_vc(vc=value_counts)["word_counts"].to_dict()
        == pd.Series(
            index=["the", "dog", "hungry", "a", "cat"], data=[2, 2, 2, 1, 1]
        ).to_dict()
    )


@pytest.mark.parametrize("token", ["—", "–", "…", "«", "»", "“", "”"])
def test_word_summary_vc_ignores_unicode_punctuation_only_tokens(token):
    value_counts = pd.Series(index=[f"Star Wars {token} Episode IV"], data=[2])
    assert word_summary_vc(vc=value_counts)["word_counts"].to_dict() == {
        "star": 2,
        "wars": 2,
        "episode": 2,
        "iv": 2,
    }


@pytest.mark.parametrize("word", ["—hello—", "–hello–", "hello…", "«hello»", "“hello”"])
def test_word_summary_vc_strips_unicode_punctuation_around_words(word):
    value_counts = pd.Series(index=[word], data=[1])
    assert word_summary_vc(vc=value_counts)["word_counts"].to_dict() == {"hello": 1}


@pytest.mark.parametrize("word", ["don't", "don’t", "e-mail"])
def test_word_summary_vc_keeps_inner_punctuation(word):
    value_counts = pd.Series(index=[f"({word})"], data=[1])
    assert word_summary_vc(vc=value_counts)["word_counts"].to_dict() == {word: 1}


@pytest.mark.parametrize(
    "values",
    [
        ["-", "--", "...", "_"],
        ["—", "–", "…"],
        ["— –", "- …", "«»"],
    ],
)
def test_word_summary_vc_punctuation_only_column(values):
    value_counts = pd.Series(index=values, data=range(len(values), 0, -1))
    assert "word_counts" not in word_summary_vc(vc=value_counts)
