# mypy: ignore-errors

from typing import Callable, Dict, List, Type

import pandas as pd
from visions import VisionsBaseType, VisionsTypeset

from data_profiling.config import Settings
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
from data_profiling.utils.backend import is_pyspark_installed


class BaseSummarizer(Handler):
    """A base summarizer

    Can be used to define custom summarizations
    """

    def summarize(
        self, config: Settings, series: pd.Series, dtype: Type[VisionsBaseType]
    ) -> dict:
        """Generates the summary for a given series"""
        return self.handle(str(dtype), config, series, {"type": str(dtype)})


# Revisit this with the correct support for Spark as well.
class ProfilingSummarizer(BaseSummarizer):
    """A summarizer for Pandas DataFrames."""

    def __init__(self, typeset: VisionsTypeset, use_spark: bool = False):
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


# Kept importable from here for backwards compatibility; the implementations
# live with the rest of the deprecated JSON export.
from data_profiling.model.legacy_json import (  # noqa: E402,F401
    _redact_column,
    format_summary,
    redact_summary,
)
