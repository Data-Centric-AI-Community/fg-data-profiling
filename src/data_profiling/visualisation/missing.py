"""Plotting functions for the missing values diagrams"""

from typing import Any, List

from matplotlib import pyplot as plt

from data_profiling.config import Settings
from data_profiling.visualisation import charts
from data_profiling.visualisation.charts import builders
from data_profiling.visualisation.context import manage_matplotlib_context
from data_profiling.visualisation.plot import (
    missing_bar,
    missing_heatmap,
    missing_matrix,
)
from data_profiling.visualisation.utils import hex_to_rgb, plot_360_n0sc0pe


def get_font_size(columns: List[str]) -> float:
    """Calculate font size based on number of columns

    Args:
        columns: List of column names.

    Returns:
        Font size for missing values plots.
    """
    max_label_length = max(len(label) for label in columns)

    if len(columns) < 20:
        font_size = 13.0
    elif 20 <= len(columns) < 40:
        font_size = 12.0
    elif 40 <= len(columns) < 60:
        font_size = 10.0
    else:
        font_size = 8.0

    font_size *= min(1.0, 20.0 / max_label_length)
    return font_size


@manage_matplotlib_context()
def _plot_missing_matrix_mpl(
    config: Settings, notnull: Any, columns: List[str], nrows: int
) -> str:
    """Generate missing values matrix plot

    Args:
        config: report Settings object
        notnull: Missing data indicator matrix.
        columns: List of column names.
        nrows: Number of rows in the dataframe.

    Returns:
        The resulting missing values matrix encoded as a string.
    """

    missing_matrix(
        notnull=notnull,
        height=nrows,
        columns=columns,
        figsize=(10, 4),
        fontsize=get_font_size(columns) / 20 * 16,
        color=hex_to_rgb(config.html.style.primary_colors[0]),
        labels=config.plot.missing.force_labels,
    )
    plt.subplots_adjust(left=0.1, right=0.9, top=0.7, bottom=0.2)
    return plot_360_n0sc0pe(config)


@manage_matplotlib_context()
def _plot_missing_bar_mpl(
    config: Settings, notnull_counts: list, nrows: int, columns: List[str]
) -> str:
    """Generate missing values bar plot.

    Args:
        config: report Settings object
        notnull_counts: Number of nonnull values per column.
        nrows: Number of rows in the dataframe.
        columns: List of column names.

    Returns:
        The resulting missing values bar plot encoded as a string.
    """
    missing_bar(
        notnull_counts=notnull_counts,
        nrows=nrows,
        figsize=(10, 5),
        fontsize=get_font_size(columns),
        color=hex_to_rgb(config.html.style.primary_colors[0]),
        labels=config.plot.missing.force_labels,
    )
    for ax0 in plt.gcf().get_axes():
        ax0.grid(False)
    plt.subplots_adjust(left=0.1, right=0.9, top=0.8, bottom=0.3)

    return plot_360_n0sc0pe(config)


@manage_matplotlib_context()
def _plot_missing_heatmap_mpl(
    config: Settings, corr_mat: Any, mask: Any, columns: List[str]
) -> str:
    """Generate missing values heatmap plot.

    Args:
        config: report Settings object
        corr_mat: Correlation matrix.
        maks: Upper-triangle mask.
        columns: List of column names.

    Returns:
        The resulting missing values heatmap plot encoded as a string.
    """

    height = 4
    if len(columns) > 10:
        height += int((len(columns) - 10) / 5)
    height = min(height, 10)

    font_size = get_font_size(columns)
    if len(columns) > 40:
        font_size /= 1.4

    missing_heatmap(
        corr_mat=corr_mat,
        mask=mask,
        figsize=(10, height),
        fontsize=font_size,
        cmap=config.plot.missing.cmap,
        labels=config.plot.missing.force_labels,
    )

    if len(columns) > 40:
        plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.3)
    else:
        plt.subplots_adjust(left=0.2, right=0.9, top=0.8, bottom=0.3)

    return plot_360_n0sc0pe(config)


# Public functions: unchanged signatures; native renderer returns inline SVG.


def plot_missing_matrix(
    config: Settings, notnull: Any, columns: List[str], nrows: int
) -> str:
    """Generate missing values matrix plot."""
    if charts.use_native(config):
        return charts.render(builders.missing_matrix(config, notnull, columns, nrows))
    return _plot_missing_matrix_mpl(config, notnull, columns, nrows)


def plot_missing_bar(
    config: Settings, notnull_counts: list, nrows: int, columns: List[str]
) -> str:
    """Generate missing values bar plot."""
    if charts.use_native(config):
        return charts.render(
            builders.missing_bar(config, notnull_counts, nrows, columns)
        )
    return _plot_missing_bar_mpl(config, notnull_counts, nrows, columns)


def plot_missing_heatmap(
    config: Settings, corr_mat: Any, mask: Any, columns: List[str]
) -> str:
    """Generate missing values heatmap plot."""
    if charts.use_native(config):
        return charts.render(builders.missing_heatmap(config, corr_mat, mask, columns))
    return _plot_missing_heatmap_mpl(config, corr_mat, mask, columns)
