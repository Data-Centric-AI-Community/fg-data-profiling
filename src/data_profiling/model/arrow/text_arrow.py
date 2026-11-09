"""Text statistics computed on Arrow string arrays.

Each kernel returns the same pandas structures as the value-count based pandas
helpers in ``describe_categorical_pandas`` so the existing summary helpers
(``length_summary_from_counts``, ``unicode_summary_from_counts``) and the report
renderers can be reused unchanged.

Work is done chunk by chunk with pyarrow.compute and numpy, never by iterating
over Python strings, so memory stays bounded by ``chunk_size`` rows.
"""
import string
from collections import Counter
from typing import Iterator, List, Optional

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc

DEFAULT_CHUNK_SIZE = 65_536

# Same characters the pandas helpers strip from word boundaries.
_STRIP_CHARACTERS = string.punctuation + string.whitespace


def to_string_array(values: object) -> pa.ChunkedArray:
    """Return the non-null values of a Series/Array as a large_string ChunkedArray.

    Arrow-backed pandas Series are converted without copying the text; other
    Series are converted with ``astype(str)`` first, as the pandas path does.
    """
    if isinstance(values, pd.Series):
        if isinstance(values.dtype, pd.ArrowDtype):
            values = values.array.__arrow_array__()
        else:
            values = pa.array(values.dropna().astype(str), type=pa.large_string())
    if isinstance(values, pa.Array):
        values = pa.chunked_array([values])
    if not isinstance(values, pa.ChunkedArray):
        raise TypeError(f"Unsupported input type {type(values)!r}")
    if not pa.types.is_large_string(values.type):
        values = values.cast(pa.large_string())
    return pc.drop_null(values)


def _iter_chunks(values: pa.ChunkedArray, chunk_size: int) -> Iterator[pa.Array]:
    for chunk in values.chunks:
        for start in range(0, len(chunk), chunk_size):
            yield chunk.slice(start, chunk_size)


def _value_counts_to_series(counts: pa.StructArray) -> pd.Series:
    if len(counts) == 0:
        return pd.Series(dtype="int64")
    values = counts.field("values").to_numpy(zero_copy_only=False)
    frequencies = counts.field("counts").to_numpy()
    series = pd.Series(frequencies, index=values)
    return series.sort_values(ascending=False, kind="stable")


def length_counts(values: pa.ChunkedArray) -> pd.Series:
    """String length (in code points) -> number of values with that length."""
    if len(values) == 0:
        return pd.Series(dtype="int64")
    counts = pc.value_counts(pc.utf8_length(values))
    return _value_counts_to_series(counts)


def _utf8_bytes(chunk: pa.Array) -> np.ndarray:
    """The UTF-8 bytes of a (sliced) large_string array, without copying."""
    offsets = np.frombuffer(chunk.buffers()[1], dtype=np.int64)
    start, end = offsets[chunk.offset], offsets[chunk.offset + len(chunk)]
    data = chunk.buffers()[2]
    if data is None:
        return np.empty(0, dtype=np.uint8)
    return np.frombuffer(data, dtype=np.uint8)[start:end]


def _encode_characters(data: np.ndarray) -> np.ndarray:
    """One uint32 key per UTF-8 character: its encoded bytes, big-endian packed.

    Lead bytes are those not of the form 0b10xxxxxx; the byte length of each
    character follows from its lead byte. Distinct keys are later decoded back
    to characters, so only the (small) set of distinct characters is ever
    turned into Python strings.
    """
    lead = np.flatnonzero((data & 0xC0) != 0x80)
    first = data[lead].astype(np.uint32)
    n_bytes = np.where(
        first < 0x80, 1, np.where(first < 0xE0, 2, np.where(first < 0xF0, 3, 4))
    )
    padded = np.concatenate([data, np.zeros(3, dtype=np.uint8)]).astype(np.uint32)
    keys = first << 24
    keys |= np.where(n_bytes > 1, padded[lead + 1], 0) << 16
    keys |= np.where(n_bytes > 2, padded[lead + 2], 0) << 8
    keys |= np.where(n_bytes > 3, padded[lead + 3], 0)
    return keys


def _decode_key(key: int) -> str:
    raw = int(key).to_bytes(4, "big").rstrip(b"\x00") or b"\x00"
    return raw.decode("utf-8", errors="replace")


def character_counts(
    values: pa.ChunkedArray, chunk_size: int = DEFAULT_CHUNK_SIZE
) -> pd.Series:
    """Character (code point) -> number of occurrences, most frequent first."""
    totals: Counter = Counter()
    for chunk in _iter_chunks(values, chunk_size):
        data = _utf8_bytes(chunk)
        if data.size == 0:
            continue
        keys, counts = np.unique(_encode_characters(data), return_counts=True)
        totals.update(dict(zip(keys.tolist(), counts.tolist())))

    if not totals:
        return pd.Series(dtype="int64")
    characters: Counter = Counter()
    for key, count in totals.items():
        characters[_decode_key(key)] += count
    series = pd.Series(characters, dtype="int64")
    return series.sort_values(ascending=False, kind="stable")


def word_counts(
    values: pa.ChunkedArray,
    stop_words: Optional[List[str]] = None,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
) -> pd.Series:
    """Word -> number of occurrences, most frequent first.

    Mirrors the pandas helper: lowercase, split on whitespace, strip
    punctuation and whitespace from both ends, drop stop words. Tokens that are
    empty after stripping (punctuation-only words) are dropped, as in the Spark
    backend.
    """
    stop = None
    if stop_words:
        stop = pa.array([w.lower() for w in stop_words], type=pa.large_string())

    # Count per chunk (bounded memory), then add the partial counts up with a
    # single Arrow group-by.
    partial_counts = []
    for chunk in _iter_chunks(values, chunk_size):
        words = pc.list_flatten(pc.utf8_split_whitespace(pc.utf8_lower(chunk)))
        words = pc.utf8_trim(words, characters=_STRIP_CHARACTERS)
        words = words.filter(pc.not_equal(words, ""))
        if stop is not None:
            words = words.filter(pc.invert(pc.is_in(words, value_set=stop)))
        if len(words) > 0:
            counts = pc.value_counts(words)
            partial_counts.append(
                pa.table({"word": counts.field("values"), "n": counts.field("counts")})
            )

    if not partial_counts:
        return pd.Series(dtype="int64")
    totals = pa.concat_tables(partial_counts)
    count_column = "n"
    if len(partial_counts) > 1:
        totals = totals.group_by("word").aggregate([("n", "sum")])
        count_column = "n_sum"
    series = pd.Series(
        totals.column(count_column).to_numpy(),
        index=totals.column("word").to_numpy(zero_copy_only=False),
    )
    return series.sort_values(ascending=False, kind="stable")
