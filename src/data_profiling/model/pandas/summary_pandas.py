"""Compute statistical description of datasets."""
import multiprocessing
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm

from data_profiling.config import Settings
from data_profiling.model.typeset import ProfilingTypeSet
from data_profiling.model.typeset_base import BaseTypeSet
from data_profiling.utils.compat import optional_option_context
from data_profiling.utils.dataframe import sort_column_names

BaseSummarizer: Any = "BaseSummarizer"  # type: ignore


def _is_cast_type_defined(typeset: BaseTypeSet, series: str) -> bool:
    return isinstance(typeset, ProfilingTypeSet) and series in typeset.type_schema


def pandas_describe_1d(
    config: Settings,
    series: pd.Series,
    summarizer: BaseSummarizer,
    typeset: BaseTypeSet,
) -> dict:
    """Describe a series (infer the variable type, then calculate type-specific values).

    Args:
        config: report Settings object
        series: The Series to describe.
        summarizer: Summarizer object
        typeset: Typeset

    Returns:
        A Series containing calculated series description values.
    """

    # Make sure pd.NA is not in the series
    with optional_option_context("future.no_silent_downcasting", True):
        series = series.fillna(np.nan).infer_objects(copy=False)

    has_cast_type = _is_cast_type_defined(typeset, series.name)  # type:ignore
    cast_type = (
        str(typeset.type_schema[series.name]) if has_cast_type else None
    )  # type:ignore

    if has_cast_type and not series.isna().all():
        vtype = typeset.type_schema[series.name]  # type:ignore

    elif config.infer_dtypes:
        # Infer variable types
        vtype, series = typeset.infer_and_cast(series)
    else:
        # Detect variable types from pandas dataframe (df.dtypes).
        # [new dtypes, changed using `astype` function are now considered]
        vtype = typeset.detect_type(series)

    typeset.type_schema[series.name] = vtype  # type:ignore
    summary = summarizer.summarize(config, series, dtype=vtype)
    # Cast type is only used on unsupported columns rendering pipeline
    # to indicate the correct variable type when inference is not possible
    summary["cast_type"] = cast_type

    return summary


def pandas_get_series_descriptions(
    config: Settings,
    df: pd.DataFrame,
    summarizer: BaseSummarizer,
    typeset: BaseTypeSet,
    pbar: tqdm,
) -> dict:
    def describe_column(name: str, series: pd.Series) -> Tuple[str, dict]:
        """Process a single series to get the column description."""
        pbar.set_postfix_str(f"Describe variable: {name}")
        description = pandas_describe_1d(config, series, summarizer, typeset)
        pbar.update()
        return name, description

    pool_size = (
        config.pool_size if config.pool_size > 0 else multiprocessing.cpu_count()
    )

    series_description = {}

    with ThreadPoolExecutor(max_workers=pool_size) as executor:
        future_to_col = {
            executor.submit(describe_column, name, series): name  # type:ignore
            for name, series in df.items()
        }

        for future in tqdm(future_to_col.keys(), total=len(future_to_col)):
            name, description = future.result()
            series_description[name] = description

    return sort_column_names(series_description, config.sort)
