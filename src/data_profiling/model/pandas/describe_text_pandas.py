from typing import Tuple

import pandas as pd

from data_profiling.config import Settings
from data_profiling.model.pandas.describe_categorical_pandas import (
    length_summary_from_counts,
    length_summary_vc,
    unicode_summary_from_counts,
    unicode_summary_vc,
    word_summary_vc,
)
from data_profiling.model.summary_algorithms import (
    histogram_compute,
    series_handle_nulls,
    series_hashable,
)
from data_profiling.utils.arrow import is_pyarrow_installed


@series_hashable
@series_handle_nulls
def pandas_describe_text_1d(
    config: Settings,
    series: pd.Series,
    summary: dict,
) -> Tuple[Settings, pd.Series, dict]:
    """Describe string series.

    Args:
        config: report Settings object
        series: The Series to describe.
        summary: The dict containing the series description so far.

    Returns:
        A dict containing calculated series description values.
    """

    if _use_arrow(config, series):
        _describe_text_arrow(config, series, summary)
        return config, series, summary

    series = series.astype(str)

    # Only run if at least 1 non-missing value
    value_counts = summary["value_counts_without_nan"]
    value_counts.index = value_counts.index.astype(str)

    summary.update({"first_rows": series.head(5)})

    if config.vars.text.length:
        summary.update(length_summary_vc(value_counts))
        summary.update(
            histogram_compute(
                config,
                summary["length_histogram"].index.values,
                len(summary["length_histogram"]),
                name="histogram_length",
                weights=summary["length_histogram"].values,
            )
        )

    if config.vars.text.characters:
        summary.update(unicode_summary_vc(value_counts))

    if config.vars.text.words:
        summary.update(word_summary_vc(value_counts, config.vars.cat.stop_words))

    return config, series, summary


def _use_arrow(config: Settings, series: pd.Series) -> bool:
    engine = config.vars.text.engine
    if engine == "pandas":
        return False
    if engine == "arrow":
        if not is_pyarrow_installed():
            raise ImportError(
                "vars.text.engine='arrow' requires pyarrow: "
                "`pip install fg-data-profiling[arrow]`."
            )
        return True
    return isinstance(series.dtype, pd.ArrowDtype)


def _describe_text_arrow(config: Settings, series: pd.Series, summary: dict) -> None:
    """Text statistics on the column's Arrow buffers (see model.arrow)."""
    from data_profiling.model.arrow.text_arrow import (
        character_counts,
        length_counts,
        to_string_array,
        word_counts,
    )

    values = to_string_array(series)
    summary.update({"first_rows": series.head(5).astype(str)})

    if config.vars.text.length:
        lengths = length_counts(values)
        summary.update(length_summary_from_counts(lengths))
        summary.update(
            histogram_compute(
                config,
                lengths.index.values,
                len(lengths),
                name="histogram_length",
                weights=lengths.values,
            )
        )

    if config.vars.text.characters:
        summary.update(unicode_summary_from_counts(character_counts(values)))

    if config.vars.text.words:
        words = word_counts(values, config.vars.cat.stop_words)
        if not words.empty:
            summary["word_counts"] = words
