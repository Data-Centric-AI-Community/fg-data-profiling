from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from data_profiling.config import Settings
from data_profiling.model import BaseDescription
from data_profiling.report.presentation.core import Container, CorrelationTable, Image
from data_profiling.report.presentation.core.renderable import Renderable
from data_profiling.visualisation import plot

CORRELATION_NOTES = {
    "auto": "Auto picks a coefficient per pair from the variable types, so "
    "numeric, categorical and boolean variables share one matrix.",
    "pearson": "Linear correlation between numeric variables (-1 to 1); "
    "sensitive to outliers.",
    "spearman": "Rank correlation between numeric variables (-1 to 1); robust "
    "to skew and outliers.",
    "kendall": "Rank agreement between numeric variables (-1 to 1); "
    "conservative on small samples.",
    "phi_k": "φk captures non-linear dependency between variables of any "
    "type (0 to 1).",
    "cramers": "Cramér's V measures association between categorical "
    "variables (0 to 1).",
}

PAIR_METHOD_LABELS = {
    "auto": "Auto",
    "pearson": "Pearson",
    "spearman": "Spearman",
    "kendall": "Kendall",
    "phi_k": "Phik",
    "cramers": "Cramér's V",
}


def get_strongest_pairs(
    config: Settings, correlations: Dict[str, Any], top: int = 8
) -> List[Dict[str, Any]]:
    """Strongest off-diagonal pairs across all computed correlation methods.

    For a comparison report every dataset is ranked together and the dataset
    label is shown next to the method.
    """
    labels = list(config.html.style._labels)
    rows: List[Dict[str, Any]] = []
    for key, item in correlations.items():
        matrices = item if isinstance(item, list) else [item]
        for idx, matrix in enumerate(matrices):
            if not isinstance(matrix, pd.DataFrame) or matrix.empty:
                continue
            values = matrix.to_numpy(dtype=float, na_value=np.nan)
            cols = [str(c) for c in matrix.columns]
            for i in range(len(cols)):
                for j in range(i + 1, min(len(cols), values.shape[1])):
                    v = values[i, j]
                    if np.isnan(v):
                        continue
                    method = PAIR_METHOD_LABELS.get(key, key)
                    if len(matrices) > 1 and idx < len(labels):
                        method = f"{method} · {labels[idx]}"
                    rows.append(
                        {
                            "x": cols[i],
                            "y": cols[j],
                            "method": method,
                            "value": float(v),
                            "width": min(abs(float(v)), 1.0) * 100,
                            "negative": v < 0,
                        }
                    )
    rows.sort(key=lambda r: -abs(r["value"]))
    return rows[:top]


def get_correlation_items(
    config: Settings, summary: BaseDescription
) -> Optional[Renderable]:
    """Create the list of correlation items

    Args:
        config: report Settings object
        summary: dict of correlations

    Returns:
        List of correlation items to show in the interface.
    """
    items: List[Renderable] = []

    key_to_data = {
        "pearson": (-1, "Pearson's r"),
        "spearman": (-1, "Spearman's ρ"),
        "kendall": (-1, "Kendall's τ"),
        "phi_k": (0, "Phik (φk)"),
        "cramers": (0, "Cramér's V (φc)"),
        "auto": (-1, "Auto"),
    }

    image_format = config.plot.image_format

    for key, item in summary.correlations.items():
        vmin, name = key_to_data[key]

        if isinstance(item, list):
            diagrams: List[Renderable] = []
            for idx, i in enumerate(item):
                diagram: Renderable = Image(
                    plot.correlation_matrix(config, i, vmin=vmin),
                    image_format=image_format,
                    alt=name,
                    anchor_id=f"{key}_diagram",
                    name=config.html.style._labels[idx],
                    classes="correlation-diagram",
                )
                diagrams.append(diagram)

            diagrams_grid = Container(
                diagrams,
                anchor_id=f"{key}_diagram_with_desc",
                name="Heatmap" if config.correlation_table else name,
                sequence_type="batch_grid",
                batch_size=len(config.html.style._labels),
            )

            if config.correlation_table:
                tables: List[Renderable] = []
                for idx, i in enumerate(item):
                    table = CorrelationTable(
                        name=config.html.style._labels[idx],
                        correlation_matrix=i,
                        anchor_id=f"{key}_table",
                    )
                    tables.append(table)

                tables_tab = Container(
                    tables,
                    anchor_id=f"{key}_tables",
                    name="Table",
                    sequence_type="batch_grid",
                    batch_size=len(config.html.style._labels),
                )

                diagrams_tables_tab = Container(
                    [diagrams_grid, tables_tab],
                    anchor_id=f"{key}_diagram_table",
                    name=name,
                    sequence_type="tabs",
                )

                items.append(diagrams_tables_tab)
            else:
                items.append(diagrams_grid)
        else:
            diagram = Image(
                plot.correlation_matrix(config, item, vmin=vmin),
                image_format=image_format,
                alt=name,
                anchor_id=f"{key}_diagram",
                name="Heatmap" if config.correlation_table else name,
                classes="correlation-diagram",
            )

            if config.correlation_table:
                table = CorrelationTable(
                    name="Table", correlation_matrix=item, anchor_id=f"{key}_table"
                )

                diagram_table_tabs = Container(
                    [diagram, table],
                    anchor_id=f"{key}_diagram_table",
                    name=name,
                    sequence_type="tabs",
                )

                items.append(diagram_table_tabs)
            else:
                items.append(diagram)

    corr = Container(
        items,
        sequence_type="correlations",
        name="Correlations",
        anchor_id="correlations_tab",
        methods=[
            {
                "key": key,
                "label": key_to_data[key][1],
                "note": CORRELATION_NOTES.get(key, ""),
            }
            for key in summary.correlations
        ],
        pairs=get_strongest_pairs(config, summary.correlations),
    )

    if len(items) > 0:
        return corr

    return None
