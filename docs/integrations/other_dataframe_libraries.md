# Other DataFrame libraries

`data-profiling` is built on `pandas` and `numpy`. Pandas supports a
wide range of data formats including CSV, XLSX, SQL, JSON, HDF5, SAS,
BigQuery and Stata. Read more on [supported formats by
Pandas](https://pandas.pydata.org/docs/user_guide/io.html).

## Arrow-compatible inputs

With the `arrow` extra, `ProfileReport` accepts Arrow-compatible objects
directly, with no manual conversion:

```bash
pip install "fg-data-profiling[arrow]"
```

| Input | Notes |
|-------|-------|
| `pyarrow.Table`, `RecordBatch`, `RecordBatchReader`, `pyarrow.dataset.Dataset` | |
| Hugging Face `datasets.Dataset` | Read through its indices mapping (see below) |
| Polars `DataFrame` | `LazyFrame` must be collected first |
| DuckDB relation | |
| Any object implementing `__arrow_c_stream__` | Arrow PyCapsule interface |

```python linenums="1" title="Profiling a Hugging Face dataset"
from datasets import load_dataset
from data_profiling import ProfileReport

dataset = load_dataset("fancyzhx/ag_news", split="train")
report = ProfileReport(dataset, title="AG News")
report.to_file("ag_news.html")
```

```python linenums="1" title="Profiling a DuckDB query"
import duckdb
from data_profiling import ProfileReport

relation = duckdb.sql("SELECT * FROM 'reviews.parquet' WHERE stars <= 2")
ProfileReport(relation).to_file("low_star_reviews.html")
```

The input is converted to a pandas DataFrame whose string columns stay in
Arrow memory (`pd.ArrowDtype`): text is not copied into Python objects, and
text statistics (lengths, characters, words) are computed with Arrow kernels.
Numeric, boolean, date and categorical columns are profiled exactly as for a
pandas input. The `vars.text.engine` setting controls this: `auto` (default)
uses Arrow kernels for Arrow-backed columns only, `arrow` uses them for every
text column, and `pandas` never does.

!!! note "Hugging Face datasets after `filter`, `shuffle` or `select`"

    These operations keep the original table and add an indices mapping.
    The report always profiles the rows the dataset exposes, which copies
    the selected rows into memory once. An INFO log line reports how many.

Streaming (`IterableDataset`) Hugging Face datasets are not supported:
materialize a sample first, e.g. `Dataset.from_list(list(ds.take(100_000)))`.

## Other libraries

For other frameworks of the Python Data ecosystem, convert to a pandas
`DataFrame` first. Large datasets might require
sampling (as seen in our documentation on [how to profile large datasets](../features/big_data.md)).

```python linenums="1" title="Dask to Pandas"
# Convert dask DataFrame to a pandas DataFrame
df = df.compute()
```

```python linenums="1" title="Vaex to Pandas"
# Convert vaex DataFrame to a pandas DataFrame
df = df.to_pandas_df()
```
!!! note "Modin interface"

    This is not part of the API as pandas.DataFrame, naturally, does not posses such a method.
    You can use the private method DataFrame._to_pandas() to do this conversion.
    If you would like to do this through the official API you can always save the Modin DataFrame to
    storage (csv, hdf, sql, ect) and then read it back using Pandas. This will probably be the safer
    way when working big DataFrames, to avoid out of memory issues."
    Source: https://github.com/modin-project/modin/issues/896

```python linenums="1" title="Modin to Pandas"
# Convert modin DataFrame to pandas DataFrame
df = df._to_pandas()
```

<img referrerpolicy="no-referrer-when-downgrade" src="https://static.scarf.sh/a.png?x-pxid=baa0e45f-0c03-4190-9646-9d8ea2640ba2" />