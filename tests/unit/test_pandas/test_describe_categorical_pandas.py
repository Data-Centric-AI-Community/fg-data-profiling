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
