# mypy: ignore-errors

from dataclasses import asdict
from typing import Any, Callable, Dict, List, Type, Union

import numpy as np
import pandas as pd

from data_profiling.config import Settings
from data_profiling.model import BaseDescription
from data_profiling.model.handler import Handler
from data_profiling.model.pandas import (
    pandas_describe_boolean_1d,
    pandas_describe_categorical_1d,
    pandas_describe_counts,
    pandas_describe_date_1d,
    pandas_describe_file_1d,
    pandas_describe_generic,
    pandas_describe_image_1d,
    pandas_describe_numeric_1d,
    pandas_describe_path_1d,
    pandas_describe_text_1d,
    pandas_describe_timeseries_1d,
    pandas_describe_url_1d,
)
from data_profiling.model.pandas.describe_supported_pandas import (
    pandas_describe_supported,
)
from data_profiling.model.summary_algorithms import (  # Check what is this method used for
    describe_file_1d,
    describe_image_1d,
    describe_path_1d,
    describe_timeseries_1d,
    describe_url_1d,
)
from data_profiling.model.typeset_base import BaseTypeSet, ProfilingBaseType
from data_profiling.utils.backend import is_pyspark_installed


class BaseSummarizer(Handler):
    """A base summarizer

    Can be used to define custom summarizations
    """

    def summarize(
        self,
        config: Settings,
        series: pd.Series,
        dtype: Union[str, Type[ProfilingBaseType]],
    ) -> dict:
        """Generates the summary for a given series"""
        return self.handle(str(dtype), config, series, {"type": str(dtype)})


# Revisit this with the correct support for Spark as well.
class ProfilingSummarizer(BaseSummarizer):
    """A summarizer for Pandas DataFrames."""

    def __init__(self, typeset: BaseTypeSet, use_spark: bool = False):
        self.use_spark = use_spark and is_pyspark_installed()
        self._summary_map = self._create_summary_map()
        super().__init__(self._summary_map, typeset)

    @property
    def summary_map(self) -> Dict[str, List[Callable]]:
        """Allows users to modify the summary map after initialization."""
        return self._summary_map

    def _create_summary_map(self) -> Dict[str, List[Callable]]:
        """Creates the summary map for Pandas summarization."""
        if self.use_spark:
            from data_profiling.model.spark import (
                describe_boolean_1d_spark,
                describe_categorical_1d_spark,
                describe_counts_spark,
                describe_date_1d_spark,
                describe_generic_spark,
                describe_numeric_1d_spark,
                describe_supported_spark,
                describe_text_1d_spark,
            )

            summary_map = {
                "Unsupported": [
                    describe_counts_spark,
                    describe_generic_spark,
                    describe_supported_spark,
                ],
                "Numeric": [describe_numeric_1d_spark],
                "DateTime": [describe_date_1d_spark],
                "Text": [describe_text_1d_spark],
                "Categorical": [describe_categorical_1d_spark],
                "Boolean": [describe_boolean_1d_spark],
                "URL": [describe_url_1d],
                "Path": [describe_path_1d],
                "File": [describe_file_1d],
                "Image": [describe_image_1d],
                "TimeSeries": [describe_timeseries_1d],
            }
        else:
            summary_map = {
                "Unsupported": [
                    pandas_describe_counts,
                    pandas_describe_generic,
                    pandas_describe_supported,
                ],
                "Numeric": [pandas_describe_numeric_1d],
                "DateTime": [pandas_describe_date_1d],
                "Text": [pandas_describe_text_1d],
                "Categorical": [pandas_describe_categorical_1d],
                "Boolean": [pandas_describe_boolean_1d],
                "URL": [pandas_describe_url_1d],
                "Path": [pandas_describe_path_1d],
                "File": [pandas_describe_file_1d],
                "Image": [pandas_describe_image_1d],
                "TimeSeries": [pandas_describe_timeseries_1d],
            }
        return summary_map


def format_summary(summary: Union[BaseDescription, dict]) -> dict:
    """Prepare summary for export to json file.

    Args:
        summary (Union[BaseDescription, dict]): summary to export

    Returns:
        dict: summary as dict
    """

    def fmt(v: Any) -> Any:
        if isinstance(v, dict):
            return {k: fmt(va) for k, va in v.items()}
        else:
            if isinstance(v, pd.Series):
                return fmt(v.to_dict())
            elif (
                isinstance(v, tuple)
                and len(v) == 2
                and all(isinstance(x, np.ndarray) for x in v)
            ):
                return {"counts": v[0].tolist(), "bin_edges": v[1].tolist()}
            else:
                return v

    if isinstance(summary, BaseDescription):
        summary = asdict(summary)

    summary = {k: fmt(v) for k, v in summary.items()}
    return summary


def _redact_column(column: Dict[str, Any]) -> Dict[str, Any]:
    def redact_key(data: Dict[str, Any]) -> Dict[str, Any]:
        return {f"REDACTED_{i}": v for i, (_, v) in enumerate(data.items())}

    def redact_value(data: Dict[str, Any]) -> Dict[str, Any]:
        return {k: f"REDACTED_{i}" for i, (k, _) in enumerate(data.items())}

    keys_to_redact = [
        "block_alias_char_counts",
        "block_alias_values",
        "category_alias_char_counts",
        "category_alias_values",
        "character_counts",
        "script_char_counts",
        "value_counts_index_sorted",
        "value_counts_without_nan",
        "word_counts",
    ]

    values_to_redact = ["first_rows"]

    for field in keys_to_redact:
        if field not in column:
            continue
        is_dict = (isinstance(v, dict) for v in column[field].values())
        if any(is_dict):
            column[field] = {k: redact_key(v) for k, v in column[field].items()}
        else:
            column[field] = redact_key(column[field])

    for field in values_to_redact:
        if field not in column:
            continue
        is_dict = (isinstance(v, dict) for v in column[field].values())
        if any(is_dict):
            column[field] = {k: redact_value(v) for k, v in column[field].items()}
        else:
            column[field] = redact_value(column[field])

    return column


def redact_summary(summary: dict, config: Settings) -> dict:
    """Redact summary to export to json file.

    Args:
        summary (dict): summary to redact

    Returns:
        dict: redacted summary
    """
    for _, col in summary["variables"].items():
        if (config.vars.cat.redact and col["type"] == "Categorical") or (
            config.vars.text.redact and col["type"] == "Text"
        ):
            col = _redact_column(col)
    return summary
