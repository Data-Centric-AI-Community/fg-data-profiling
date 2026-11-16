"""Arrow input support.

Converts Arrow-compatible objects (pyarrow tables and readers, Hugging Face
``datasets.Dataset``, and any object implementing the Arrow PyCapsule stream
interface such as Polars and DuckDB) into a pandas DataFrame whose string
columns stay Arrow-backed (``pd.ArrowDtype``). Text statistics can then run on
the Arrow buffers without copying every document into Python objects.

pyarrow is an optional dependency: ``pip install fg-data-profiling[arrow]``.
"""
import importlib.util
import logging
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)

_INSTALL_HINT = (
    "Profiling Arrow, Hugging Face datasets, Polars or DuckDB inputs requires "
    "pyarrow. Install it with `pip install fg-data-profiling[arrow]`."
)


def is_pyarrow_installed() -> bool:
    """Check if pyarrow is installed without importing it."""
    return importlib.util.find_spec("pyarrow") is not None


def _import_pyarrow() -> Any:
    try:
        import pyarrow as pa
    except ImportError as e:  # pragma: no cover - depends on the environment
        raise ImportError(_INSTALL_HINT) from e
    return pa


def _module_root(obj: Any) -> str:
    return type(obj).__module__.split(".")[0]


def _is_hf_dataset(obj: Any) -> bool:
    return _module_root(obj) == "datasets" and hasattr(obj, "with_format")


def is_arrow_compatible(obj: Any) -> bool:
    """Return True for inputs this module can convert.

    pandas and Spark DataFrames are excluded: they keep their own code paths.
    """
    if obj is None or isinstance(obj, pd.DataFrame):
        return False
    root = _module_root(obj)
    if root == "pyspark":
        return False
    if root == "pyarrow" or _is_hf_dataset(obj):
        return True
    return hasattr(obj, "__arrow_c_stream__")


def _hf_dataset_to_table(dataset: Any) -> Any:
    if type(dataset).__name__ == "IterableDataset":
        raise TypeError(
            "Streaming (Iterable) Hugging Face datasets are not supported. "
            "Materialize a sample first, e.g. "
            "`Dataset.from_list(list(ds.take(100_000)))`."
        )
    if getattr(dataset, "_indices", None) is not None:
        # filter/shuffle/select keep the original table and add an indices
        # mapping. Reading `dataset.data` would profile rows the user removed,
        # so read through the mapping (one copy of the selected rows).
        logger.info(
            "Hugging Face dataset has an indices mapping (from filter, shuffle "
            "or select); copying %d of %d rows into memory to profile it.",
            len(dataset),
            dataset.data.num_rows,
        )
    return dataset.with_format("arrow")[:]


def to_arrow_table(obj: Any) -> Any:
    """Convert an Arrow-compatible object into a ``pyarrow.Table``.

    Supported inputs, in resolution order:

    - ``pyarrow.Table``, ``pyarrow.RecordBatch``, ``pyarrow.RecordBatchReader``
      and ``pyarrow.dataset.Dataset``
    - Hugging Face ``datasets.Dataset`` (read through its indices mapping)
    - any object implementing ``__arrow_c_stream__`` (Polars DataFrame,
      DuckDB relation, ...)
    """
    pa = _import_pyarrow()

    if isinstance(obj, pa.Table):
        return obj
    if isinstance(obj, pa.RecordBatch):
        return pa.Table.from_batches([obj])
    if isinstance(obj, pa.RecordBatchReader):
        return obj.read_all()
    if _module_root(obj) == "pyarrow" and hasattr(obj, "to_table"):
        return obj.to_table()
    if _is_hf_dataset(obj):
        return _hf_dataset_to_table(obj)
    if hasattr(obj, "__arrow_c_stream__"):
        return pa.table(obj)
    if type(obj).__name__ == "LazyFrame":
        raise TypeError("Polars LazyFrame is not supported; call `.collect()` first.")
    raise TypeError(f"Cannot convert an object of type {type(obj)!r} to Arrow.")


def _string_types_mapper(pa: Any) -> Any:
    string_types = (pa.string(), pa.large_string())

    def mapper(arrow_type: Any) -> Any:
        if arrow_type in string_types:
            return pd.ArrowDtype(arrow_type)
        return None  # default pandas conversion

    return mapper


def _cast_string_views(pa: Any, table: Any) -> Any:
    """Cast string_view columns (Polars' export type) to large_string.

    pandas operations on ``ArrowDtype(string_view)`` fail for several kernels
    (e.g. filtering), so these columns are copied once into large_string.
    """
    if not hasattr(pa, "string_view"):
        return table
    fields = [
        field.with_type(pa.large_string()) if field.type == pa.string_view() else field
        for field in table.schema
    ]
    schema = pa.schema(fields, metadata=table.schema.metadata)
    return table if schema.equals(table.schema) else table.cast(schema)


def arrow_to_pandas(table: Any) -> pd.DataFrame:
    """Convert a ``pyarrow.Table`` to pandas, keeping string columns in Arrow.

    String columns become ``pd.ArrowDtype`` (zero-copy for string and
    large_string; string_view is cast to large_string first); every other
    column is converted as ``Table.to_pandas()`` does by default, so numeric,
    boolean, date and categorical columns behave exactly as for a regular
    pandas input.
    """
    pa = _import_pyarrow()
    table = _cast_string_views(pa, table)
    return table.to_pandas(types_mapper=_string_types_mapper(pa))


def from_arrow_compatible(obj: Any) -> Any:
    """Return a pandas DataFrame for Arrow-compatible inputs, else ``obj``."""
    if not is_arrow_compatible(obj):
        return obj
    return arrow_to_pandas(to_arrow_table(obj))
