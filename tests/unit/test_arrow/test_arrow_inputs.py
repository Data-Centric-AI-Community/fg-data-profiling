"""Arrow-compatible inputs: adapter conversion and end-to-end reports."""
import json

import pandas as pd
import pytest

pa = pytest.importorskip("pyarrow")

from data_profiling import ProfileReport  # noqa: E402
from data_profiling.utils.arrow import (  # noqa: E402
    arrow_to_pandas,
    from_arrow_compatible,
    is_arrow_compatible,
    to_arrow_table,
)

N = 120


@pytest.fixture
def table():
    return pa.table(
        {
            "text": [
                f"review {i}: the product works, rating {i % 5}" for i in range(N)
            ],
            "label": [["neg", "neu", "pos"][i % 3] for i in range(N)],
            "score": [i / 10 for i in range(N)],
            "flag": [i % 2 == 0 for i in range(N)],
        }
    )


def _check_report(df, n_rows=N):
    report = ProfileReport(df, progress_bar=False)
    description = report.get_description()
    assert description.table["n"] == n_rows
    variables = description.variables
    assert variables["text"]["type"] == "Text"
    assert variables["label"]["type"] == "Categorical"
    assert variables["score"]["type"] == "Numeric"
    assert variables["flag"]["type"] == "Boolean"
    assert "word_counts" in variables["text"]
    assert len(report.to_html()) > 0
    json.loads(report.to_json())
    return description


def test_is_arrow_compatible(table):
    assert is_arrow_compatible(table)
    assert not is_arrow_compatible(None)
    assert not is_arrow_compatible(table.to_pandas())
    assert from_arrow_compatible(None) is None


def test_arrow_to_pandas_keeps_strings_in_arrow(table):
    df = arrow_to_pandas(table)
    assert isinstance(df["text"].dtype, pd.ArrowDtype)
    assert isinstance(df["label"].dtype, pd.ArrowDtype)
    assert df["score"].dtype == "float64"
    assert df["flag"].dtype == "bool"


def test_large_string_columns(table):
    large = table.cast(
        pa.schema(
            [
                pa.field("text", pa.large_string()),
                pa.field("label", pa.large_string()),
                pa.field("score", pa.float64()),
                pa.field("flag", pa.bool_()),
            ]
        )
    )
    assert isinstance(arrow_to_pandas(large)["text"].dtype, pd.ArrowDtype)
    _check_report(large)


@pytest.mark.parametrize(
    "make",
    [
        lambda t: t,
        lambda t: t.to_batches()[0],
        lambda t: pa.RecordBatchReader.from_batches(t.schema, t.to_batches()),
        lambda t: __import__("pyarrow.dataset", fromlist=["dataset"]).dataset(t),
    ],
    ids=["Table", "RecordBatch", "RecordBatchReader", "pyarrow.dataset"],
)
def test_pyarrow_inputs(table, make):
    assert to_arrow_table(make(table)).equals(table)
    _check_report(make(table))


def test_unsupported_type():
    with pytest.raises(TypeError):
        to_arrow_table(object())


def test_hf_dataset(table):
    datasets = pytest.importorskip("datasets")
    _check_report(datasets.Dataset(table))


def test_hf_dataset_reads_through_indices_mapping(table):
    datasets = pytest.importorskip("datasets")
    dataset = datasets.Dataset(table)

    filtered = dataset.filter(lambda row: row["label"] == "pos")
    assert filtered.data.num_rows == N  # the mapping hides 2/3 of the table
    assert to_arrow_table(filtered).num_rows == N // 3
    description = _check_report(filtered, n_rows=N // 3)
    assert description.variables["label"]["n_distinct"] == 1

    shuffled = dataset.shuffle(seed=0)
    assert to_arrow_table(shuffled).column("score").to_pylist() == (shuffled["score"])


def test_polars(table):
    pl = pytest.importorskip("polars")
    df = pl.from_arrow(table)
    _check_report(df)
    with pytest.raises(TypeError, match="collect"):
        to_arrow_table(df.lazy())


def test_duckdb(table):
    duckdb = pytest.importorskip("duckdb")
    connection = duckdb.connect()
    connection.register("reviews", table)
    relation = connection.sql("SELECT * FROM reviews WHERE score >= 6")
    _check_report(relation, n_rows=N - 60)


@pytest.mark.parametrize(
    "dtype",
    [pd.StringDtype(), "arrow-string", "arrow-large-string"],
)
def test_series_is_string_uses_dtype(dtype, monkeypatch):
    """String dtypes are recognised without the element-wise astype(str) check,
    which builds a fixed-width numpy array as wide as the longest document."""
    from data_profiling.model.typeset_relations import series_is_string

    if dtype == "arrow-string":
        dtype = pd.ArrowDtype(pa.string())
    elif dtype == "arrow-large-string":
        dtype = pd.ArrowDtype(pa.large_string())
    series = pd.Series(["a long document " * 50, "short", None], dtype=dtype)

    def fail(*args, **kwargs):
        raise AssertionError("astype should not be called")

    monkeypatch.setattr(pd.Series, "astype", fail)
    assert series_is_string(series, {})
