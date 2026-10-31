"""Organize the calculation of statistics for each series in this DataFrame."""
from datetime import datetime, timezone
from typing import Any, Dict, Optional, Union

import pandas as pd
from tqdm.auto import tqdm
from visions import VisionsTypeset

from data_profiling.config import Settings
from data_profiling.model import BaseAnalysis, BaseDescription
from data_profiling.model.alerts import get_alerts
from data_profiling.model.correlations import (
    calculate_correlation,
    get_active_correlations,
)
from data_profiling.model.dataframe import preprocess
from data_profiling.model.description import TimeIndexAnalysis
from data_profiling.model.duplicates import get_duplicates
from data_profiling.model.missing import get_missing_active, get_missing_diagram
from data_profiling.model.pairwise import get_scatter_plot, get_scatter_tasks
from data_profiling.model.sample import get_custom_sample, get_sample
from data_profiling.model.summarizer import BaseSummarizer
from data_profiling.model.summary import get_series_descriptions
from data_profiling.model.table import get_table_stats
from data_profiling.model.timeseries_index import get_time_index_description
from data_profiling.utils.progress_bar import progress
from data_profiling.version import __version__


def describe(
    config: Settings,
    df: Union[pd.DataFrame, "pyspark.sql.DataFrame"],  # type: ignore[name-defined] # noqa: F821
    summarizer: BaseSummarizer,
    typeset: VisionsTypeset,
    sample: Optional[dict] = None,
) -> BaseDescription:  # noqa: TC301
    """Calculate the statistics for each series in this DataFrame.

    Args:
        config: report Settings object
        df: DataFrame.
        summarizer: summarizer object
        typeset: visions typeset
        sample: optional, dict with custom sample

    Returns:
        This function returns a dictionary containing:
            - table: overall statistics.
            - variables: descriptions per series.
            - correlations: correlation matrices.
            - missing: missing value diagrams.
            - alerts: direct special attention to these patterns in your data.
            - package: package details.
    """
    # ** Validate Input types **
    if not isinstance(config, Settings):
        raise TypeError(f"`config` must be of type `Settings`, got {type(config)}")

    # Validate df input type

    if not isinstance(df, pd.DataFrame):
        try:
            from pyspark.sql import DataFrame as SparkDataFrame  # type: ignore

            if not isinstance(df, SparkDataFrame):  # noqa: TC301
                raise TypeError(  # noqa: TC301
                    f"`df` must be either a `pandas.DataFrame` or a `pyspark.sql.DataFrame`, but got {type(df)}."
                )
        except ImportError as ex:
            raise TypeError(
                f"`df must be either a `pandas.DataFrame` or a `pyspark.sql.DataFrame`, but got {type(df)}."
                f"If using Spark, make sure PySpark is installed."
            ) from ex

    df = preprocess(config, df)

    number_of_tasks = 5

    with tqdm(
        total=number_of_tasks,
        desc="Summarize dataset",
        disable=not config.progress_bar,
        position=0,
    ) as pbar:
        date_start = datetime.now(timezone.utc).replace(tzinfo=None)

        # Variable-specific
        pbar.total += len(df.columns)
        series_description = get_series_descriptions(
            config, df, summarizer, typeset, pbar
        )

        pbar.set_postfix_str("Get variable types")
        pbar.total += 1
        variables = {
            column: description["type"]
            for column, description in series_description.items()
        }
        supported_columns = [
            column
            for column, type_name in variables.items()
            if type_name != "Unsupported"
        ]
        interval_columns = [
            column
            for column, type_name in variables.items()
            if type_name in {"Numeric", "TimeSeries"}
        ]
        pbar.update()

        # Table statistics
        table_stats = progress(get_table_stats, pbar, "Get dataframe statistics")(
            config, df, series_description
        )

        # Get correlations
        if table_stats["n"] != 0:
            correlation_names = get_active_correlations(config)
            pbar.total += len(correlation_names)

            correlations = {
                correlation_name: progress(
                    calculate_correlation,
                    pbar,
                    f"Calculate {correlation_name} correlation",
                )(config, df, correlation_name, series_description)
                for correlation_name in correlation_names
            }

            # make sure correlations is not None
            correlations = {
                key: value for key, value in correlations.items() if value is not None
            }
        else:
            correlations = {}

        # Scatter matrix
        pbar.set_postfix_str("Get scatter matrix")
        scatter_tasks = get_scatter_tasks(config, interval_columns)
        pbar.total += len(scatter_tasks)
        scatter_matrix: Dict[Any, Dict[Any, Any]] = {
            x: {y: None} for x, y in scatter_tasks
        }
        for x, y in scatter_tasks:
            scatter_matrix[x][y] = progress(
                get_scatter_plot, pbar, f"scatter {x}, {y}"
            )(config, df, x, y, interval_columns)

        # missing diagrams
        missing_map = get_missing_active(config, table_stats)
        pbar.total += len(missing_map)
        missing = {
            name: progress(get_missing_diagram, pbar, f"Missing diagram {name}")(
                config, df, settings
            )
            for name, settings in missing_map.items()
        }
        missing = {name: value for name, value in missing.items() if value is not None}

        # Sample
        pbar.set_postfix_str("Take sample")
        if sample is None:
            samples = get_sample(config, df)
        else:
            samples = get_custom_sample(sample)
        pbar.update()

        # Duplicates
        metrics, duplicates = progress(get_duplicates, pbar, "Detecting duplicates")(
            config, df, supported_columns
        )
        table_stats.update(metrics)

        alerts = progress(get_alerts, pbar, "Get alerts")(
            config, table_stats, series_description, correlations
        )

        if config.vars.timeseries.active:
            tsindex_description = get_time_index_description(config, df, table_stats)

        pbar.set_postfix_str("Get reproduction details")
        package = {
            "data_profiling_version": __version__,
            "data_profiling_config": config.json(),
        }
        pbar.update()

        pbar.set_postfix_str("Completed")

        date_end = datetime.now(timezone.utc).replace(tzinfo=None)

    analysis = BaseAnalysis(config.title, date_start, date_end)
    time_index_analysis = None
    if config.vars.timeseries.active and tsindex_description:
        time_index_analysis = TimeIndexAnalysis(**tsindex_description)

    description = BaseDescription(
        analysis=analysis,
        time_index_analysis=time_index_analysis,
        table=table_stats,
        variables=series_description,
        scatter=scatter_matrix,
        correlations=correlations,
        missing=missing,
        alerts=alerts,
        package=package,
        sample=samples,
        duplicates=duplicates,
    )
    return description
