"""Arrow text kernels: parity with the pandas helpers and the engine switch."""
import pandas as pd
import pytest

pa = pytest.importorskip("pyarrow")

from data_profiling import ProfileReport  # noqa: E402
from data_profiling.model.arrow.text_arrow import (  # noqa: E402
    character_counts,
    length_counts,
    to_string_array,
    word_counts,
)
from data_profiling.model.pandas.describe_categorical_pandas import (  # noqa: E402
    length_summary_from_counts,
    length_summary_vc,
    unicode_summary_from_counts,
    unicode_summary_vc,
    word_summary_vc,
)

CORPUS = [
    "The quick brown fox jumps over the lazy dog.",
    "Synthetic data, real problems: (near-)duplicates everywhere!",
    "Élysée Palace — Paris; café au lait.",
    "Emoji test 😀 and CJK 数据 质量",
    "  leading and trailing whitespace  ",
    "line one\nline two\ttabbed",
    "",
    "The quick brown fox jumps over the lazy dog.",
    None,
]
NON_NULL = [v for v in CORPUS if v is not None]


@pytest.fixture
def value_counts():
    return pd.Series(NON_NULL).value_counts()


@pytest.fixture(params=[65_536, 2], ids=["one-chunk", "many-chunks"])
def chunk_size(request):
    return request.param


@pytest.fixture(params=["object", "arrow"])
def series(request):
    if request.param == "object":
        return pd.Series(CORPUS, dtype=object)
    return pd.Series(CORPUS, dtype=pd.ArrowDtype(pa.string()))


def test_to_string_array_drops_nulls(series):
    values = to_string_array(series)
    assert pa.types.is_large_string(values.type)
    assert values.to_pylist() == NON_NULL


def test_to_string_array_sliced_chunks():
    table = pa.chunked_array([pa.array(CORPUS[:4]), pa.array(CORPUS[4:])])
    values = to_string_array(table.slice(1, 6))
    assert values.to_pylist() == CORPUS[1:7]


def test_length_parity(series, value_counts):
    expected = length_summary_vc(value_counts)
    result = length_summary_from_counts(length_counts(to_string_array(series)))
    for key in ("max_length", "min_length", "mean_length", "median_length"):
        assert result[key] == pytest.approx(expected[key])
    assert result["length_histogram"].to_dict() == (
        expected["length_histogram"].to_dict()
    )


def test_character_parity(series, value_counts, chunk_size):
    expected = unicode_summary_vc(value_counts)
    counts = character_counts(to_string_array(series), chunk_size=chunk_size)
    result = unicode_summary_from_counts(counts)

    assert counts.to_dict() == expected["character_counts"].to_dict()
    for key in (
        "n_characters",
        "n_characters_distinct",
        "n_category",
        "n_scripts",
        "n_block_alias",
    ):
        assert result[key] == expected[key]
    assert result["script_counts"].to_dict() == expected["script_counts"].to_dict()
    # code points, not UTF-8 bytes or UTF-16 surrogate halves
    assert counts["😀"] == 1
    assert counts["数"] == 1


def test_word_parity(series, value_counts, chunk_size):
    expected = word_summary_vc(value_counts)["word_counts"]
    # the pandas helper also counts the empty token of punctuation-only words
    expected = expected[expected.index != ""]
    result = word_counts(to_string_array(series), chunk_size=chunk_size)
    assert result.to_dict() == expected.to_dict()
    assert result.is_monotonic_decreasing


def test_word_stop_words(series):
    result = word_counts(to_string_array(series), stop_words=["The", "AND"])
    assert "the" not in result.index
    assert "and" not in result.index
    assert result["quick"] == 2


def test_empty_input():
    values = to_string_array(pd.Series([], dtype=pd.ArrowDtype(pa.string())))
    assert length_counts(values).empty
    assert character_counts(values).empty
    assert word_counts(values).empty


def _text_description(df, **kwargs):
    report = ProfileReport(df, progress_bar=False, **kwargs)
    return report.get_description().variables["text"]


def _corpus_frame(dtype):
    corpus = [f"{v} #{i}" if v else v for i, v in enumerate(CORPUS * 10)]
    return pd.DataFrame({"text": pd.Series(corpus, dtype=dtype)})


@pytest.fixture
def arrow_calls(monkeypatch):
    """Count calls to the Arrow word kernel (imported lazily by the describer)."""
    import data_profiling.model.arrow.text_arrow as text_arrow

    calls = []
    original = text_arrow.word_counts

    def counting(*args, **kwargs):
        calls.append(1)
        return original(*args, **kwargs)

    monkeypatch.setattr(text_arrow, "word_counts", counting)
    return calls


def test_engine_auto_uses_arrow_for_arrow_backed_columns(arrow_calls):
    description = _text_description(_corpus_frame(pd.ArrowDtype(pa.string())))
    assert len(arrow_calls) == 1
    assert description["type"] == "Text"
    assert "word_counts" in description


def test_engine_auto_keeps_pandas_for_object_columns(arrow_calls):
    _text_description(_corpus_frame(object))
    assert arrow_calls == []


def test_engine_pandas_forced_on_arrow_backed_column(arrow_calls):
    df = _corpus_frame(pd.ArrowDtype(pa.string()))
    description = _text_description(df, vars={"text": {"engine": "pandas"}})
    assert arrow_calls == []
    assert "word_counts" in description


@pytest.mark.parametrize("dtype", [object, "arrow"])
def test_engines_agree(dtype):
    if dtype == "arrow":
        dtype = pd.ArrowDtype(pa.string())
    df = _corpus_frame(dtype)
    by_pandas = _text_description(df, vars={"text": {"engine": "pandas"}})
    by_arrow = _text_description(df, vars={"text": {"engine": "arrow"}})

    for key in ("max_length", "min_length", "mean_length", "n_characters"):
        assert by_arrow[key] == pytest.approx(by_pandas[key])
    assert by_arrow["character_counts"].to_dict() == (
        by_pandas["character_counts"].to_dict()
    )
    pandas_words = by_pandas["word_counts"]
    assert by_arrow["word_counts"].to_dict() == (
        pandas_words[pandas_words.index != ""].to_dict()
    )
    assert list(by_arrow["first_rows"]) == list(by_pandas["first_rows"])
