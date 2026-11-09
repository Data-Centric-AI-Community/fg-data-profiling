from datetime import datetime
from typing import Any, Dict, List, Optional, Sequence
from urllib.parse import quote

from data_profiling.config import Settings
from data_profiling.model import BaseDescription
from data_profiling.model.alerts import AlertType
from data_profiling.model.description import TimeIndexAnalysis
from data_profiling.report.formatters import (
    fmt,
    fmt_bytesize,
    fmt_number,
    fmt_numeric,
    fmt_percent,
    fmt_timespan,
    fmt_timespan_timedelta,
    list_args,
)
from data_profiling.report.presentation.core import Alerts, Container
from data_profiling.report.presentation.core import Image as ImageWidget
from data_profiling.report.presentation.core import Table
from data_profiling.report.presentation.core.renderable import Renderable
from data_profiling.visualisation.plot import plot_overview_timeseries


def get_dataset_overview(config: Settings, summary: BaseDescription) -> Renderable:
    table_metrics = [
        {
            "name": "Number of variables",
            "value": fmt_number(summary.table["n_var"]),
        },
        {
            "name": "Number of observations",
            "value": fmt_number(summary.table["n"]),
        },
        {
            "name": "Missing cells",
            "value": fmt_number(summary.table["n_cells_missing"]),
        },
        {
            "name": "Missing cells (%)",
            "value": fmt_percent(summary.table["p_cells_missing"]),
        },
    ]
    if "n_duplicates" in summary.table:
        table_metrics.extend(
            [
                {
                    "name": "Duplicate rows",
                    "value": fmt_number(summary.table["n_duplicates"]),
                },
                {
                    "name": "Duplicate rows (%)",
                    "value": fmt_percent(summary.table["p_duplicates"]),
                },
            ]
        )
    if "memory_size" in summary.table:
        table_metrics.extend(
            [
                {
                    "name": "Total size in memory",
                    "value": fmt_bytesize(summary.table["memory_size"]),
                },
                {
                    "name": "Average record size in memory",
                    "value": fmt_bytesize(summary.table["record_size"]),
                },
            ]
        )

    dataset_info = Table(
        table_metrics, name="Dataset statistics", style=config.html.style
    )

    dataset_types = Table(
        [
            {
                "name": str(type_name),
                "value": fmt_numeric(count, precision=config.report.precision),
            }
            for type_name, count in summary.table["types"].items()
        ],
        name="Variable types",
        style=config.html.style,
    )

    return Container(
        [dataset_info, dataset_types],
        anchor_id="dataset_overview",
        name="Overview",
        sequence_type="grid",
    )


def get_dataset_schema(config: Settings, metadata: dict) -> Container:
    about_dataset = []
    for key in ["description", "creator", "author"]:
        if key in metadata and len(metadata[key]) > 0:
            about_dataset.append(
                {"name": key.capitalize(), "value": fmt(metadata[key])}
            )

    if "url" in metadata:
        about_dataset.append(
            {
                "name": "URL",
                "value": f'<a href="{metadata["url"]}">{metadata["url"]}</a>',
            }
        )

    if "copyright_holder" in metadata and len(metadata["copyright_holder"]) > 0:
        if "copyright_year" not in metadata:
            about_dataset.append(
                {
                    "name": "Copyright",
                    "value": fmt(f"(c) {metadata['copyright_holder']}"),
                }
            )
        else:
            about_dataset.append(
                {
                    "name": "Copyright",
                    "value": fmt(
                        f"(c) {metadata['copyright_holder']} {metadata['copyright_year']}"
                    ),
                }
            )

    return Container(
        [
            Table(
                about_dataset,
                name="Dataset",
                anchor_id="metadata_dataset",
                style=config.html.style,
            )
        ],
        name="Dataset",
        anchor_id="dataset",
        sequence_type="grid",
    )


def get_dataset_reproduction(config: Settings, summary: BaseDescription) -> Renderable:
    """Dataset reproduction part of the report

    Args:
        config: settings object
        summary: the dataset summary.

    Returns:
        A renderable object
    """

    version = summary.package["data_profiling_version"]
    config_file = summary.package["data_profiling_config"]
    date_start = summary.analysis.date_start
    date_end = summary.analysis.date_end
    duration = summary.analysis.duration

    @list_args
    def fmt_version(version: str) -> str:
        return f'<a href="https://github.com/Data-Centric-AI-Community/data-profiling">data-profiling v{version}</a>'

    @list_args
    def fmt_config(config: str) -> str:
        return f'<a download="config.json" href="data:text/plain;charset=utf-8,{quote(config)}">config.json</a>'

    reproduction_table = Table(
        [
            {"name": "Analysis started", "value": fmt(date_start)},
            {"name": "Analysis finished", "value": fmt(date_end)},
            {"name": "Duration", "value": fmt_timespan(duration)},
            {"name": "Software version", "value": fmt_version(version)},
            {"name": "Download configuration", "value": fmt_config(config_file)},
        ],
        name="Reproduction",
        anchor_id="overview_reproduction",
        style=config.html.style,
    )

    return Container(
        [reproduction_table],
        name="Reproduction",
        anchor_id="reproduction",
        sequence_type="grid",
    )


def get_dataset_column_definitions(config: Settings, definitions: dict) -> Container:
    """Generate an overview section for the variable description

    Args:
        config: settings object
        definitions: the variable descriptions.

    Returns:
        A container object
    """

    variable_descriptions = [
        Table(
            [
                {"name": column, "value": fmt(value)}
                for column, value in definitions.items()
            ],
            name="Variable descriptions",
            anchor_id="variable_definition_table",
            style=config.html.style,
        )
    ]

    return Container(
        variable_descriptions,
        name="Variables",
        anchor_id="variable_descriptions",
        sequence_type="grid",
    )


def get_dataset_alerts(config: Settings, alerts: list) -> Alerts:
    """Obtain the alerts for the report

    Args:
        config: settings object
        alerts: list of alerts

    Returns:
        Alerts renderable object
    """
    # add up alerts from multiple reports
    if isinstance(alerts, tuple):
        count = 0

        # Initialize
        combined_alerts = {
            f"{alert.alert_type}_{alert.column_name}": [
                None for _ in range(len(alerts))
            ]
            for report_alerts in alerts
            for alert in report_alerts
        }

        for report_idx, report_alerts in enumerate(alerts):
            for alert in report_alerts:
                combined_alerts[f"{alert.alert_type}_{alert.column_name}"][
                    report_idx
                ] = alert

            count += len(
                [
                    alert
                    for alert in report_alerts
                    if alert.alert_type != AlertType.REJECTED
                ]
            )

        return Alerts(
            alerts=combined_alerts,
            name=f"Alerts ({count})",
            anchor_id="alerts",
            style=config.html.style,
        )

    count = len([alert for alert in alerts if alert.alert_type != AlertType.REJECTED])
    return Alerts(
        alerts=alerts,
        name=f"Alerts ({count})",
        anchor_id="alerts",
        style=config.html.style,
    )


def get_timeseries_items(config: Settings, summary: BaseDescription) -> Container:
    @list_args
    def fmt_tsindex_limit(limit: Any) -> str:
        if isinstance(limit, datetime):
            return limit.strftime("%Y-%m-%d %H:%M:%S")
        else:
            return fmt_number(limit)

    assert isinstance(summary.time_index_analysis, TimeIndexAnalysis)
    table_stats = [
        {
            "name": "Number of series",
            "value": fmt_number(summary.time_index_analysis.n_series),
        },
        {
            "name": "Time series length",
            "value": fmt_number(summary.time_index_analysis.length),
        },
        {
            "name": "Starting point",
            "value": fmt_tsindex_limit(summary.time_index_analysis.start),
        },
        {
            "name": "Ending point",
            "value": fmt_tsindex_limit(summary.time_index_analysis.end),
        },
        {
            "name": "Period",
            "value": fmt_timespan_timedelta(summary.time_index_analysis.period),
        },
    ]

    ts_info = Table(table_stats, name="Timeseries statistics", style=config.html.style)

    dpi_bak = config.plot.dpi
    config.plot.dpi = 300
    timeseries = ImageWidget(
        plot_overview_timeseries(config, summary.variables),
        image_format=config.plot.image_format,
        alt="ts_plot",
        name="Original",
        anchor_id="ts_plot_overview",
    )
    timeseries_scaled = ImageWidget(
        plot_overview_timeseries(config, summary.variables, scale=True),
        image_format=config.plot.image_format,
        alt="ts_plot_scaled",
        name="Scaled",
        anchor_id="ts_plot_scaled_overview",
    )
    config.plot.dpi = dpi_bak
    ts_tab = Container(
        [timeseries, timeseries_scaled],
        anchor_id="ts_plot_overview",
        name="",
        sequence_type="tabs",
    )

    return Container(
        [ts_info, ts_tab],
        anchor_id="timeseries_overview",
        name="Time Series",
        sequence_type="grid",
    )


def _as_list(value: Any) -> list:
    return list(value) if isinstance(value, (list, tuple)) else [value]


def _alert_counts(alerts: Any) -> List[Dict[str, int]]:
    """Alert counts per alert type, one dict per dataset."""
    groups = alerts if isinstance(alerts, tuple) else (alerts or [],)
    counts = []
    for group in groups:
        per_type: Dict[str, int] = {}
        for alert in group:
            if alert.alert_type == AlertType.REJECTED:
                continue
            key = alert.alert_type.name.replace("_", " ").capitalize()
            per_type[key] = per_type.get(key, 0) + 1
        counts.append(per_type)
    return counts


def get_overview_kpis(
    config: Settings, summary: BaseDescription, alerts: Any
) -> List[Dict[str, Any]]:
    """Headline numbers shown as tiles at the top of the HTML report.

    Values are lists with one entry per dataset (one entry for a single
    report), ``fractions`` drive the small bar under each value.
    """
    table = summary.table
    n = _as_list(table["n"])
    n_var = _as_list(table["n_var"])
    labels: Sequence[Optional[str]] = (
        list(config.html.style._labels) if len(n) > 1 else [None]
    )
    colors = list(config.html.style.primary_colors) or ["#0d6efd"]

    def tile(
        label: str,
        values: List[str],
        fractions: Optional[List[float]] = None,
        sub: str = "",
        tone: str = "neutral",
        segments: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        return {
            "label": label,
            "values": [
                {
                    "text": v,
                    "label": labels[i] if i < len(labels) else None,
                    "color": colors[i % len(colors)],
                }
                for i, v in enumerate(values)
            ],
            "fractions": fractions,
            "sub": sub,
            "tone": tone,
            "segments": segments,
        }

    kpis = [
        tile(
            "Rows",
            [fmt_number(v) for v in n],
            sub=f"{fmt_number(n_var[0])} columns",
        )
    ]

    types = table.get("types", {})
    if len(n) == 1 and isinstance(types, dict) and types:
        total = max(sum(int(c) for c in types.values()), 1)
        kpis.append(
            tile(
                "Variables",
                [fmt_number(n_var[0])],
                sub=" · ".join(f"{int(c)} {str(t).lower()}" for t, c in types.items()),
                segments=[
                    {"type": str(t), "fraction": int(c) / total}
                    for t, c in types.items()
                ],
            )
        )
    else:
        kpis.append(tile("Variables", [fmt_number(v) for v in n_var]))

    p_missing = [float(v) for v in _as_list(table["p_cells_missing"])]
    kpis.append(
        tile(
            "Missing cells",
            [fmt_percent(v) for v in p_missing],
            fractions=p_missing,
            sub=" · ".join(fmt_number(v) for v in _as_list(table["n_cells_missing"]))
            + " cells",
            tone="warn" if any(v > 0 for v in p_missing) else "ok",
        )
    )

    if "n_vars_with_missing" in table:
        with_missing = [int(v) for v in _as_list(table["n_vars_with_missing"])]
        kpis.append(
            tile(
                "Variables with missing",
                [fmt_number(v) for v in with_missing],
                fractions=[v / max(int(t), 1) for v, t in zip(with_missing, n_var)],
                sub="of " + fmt_number(n_var[0]) + " variables",
                tone="warn" if any(with_missing) else "ok",
            )
        )

    if "n_duplicates" in table:
        p_dup = [float(v) for v in _as_list(table["p_duplicates"])]
        kpis.append(
            tile(
                "Duplicate rows",
                [fmt_number(v) for v in _as_list(table["n_duplicates"])],
                fractions=p_dup,
                sub=" · ".join(fmt_percent(v) for v in p_dup),
                tone="warn" if any(v > 0 for v in p_dup) else "ok",
            )
        )

    counts = _alert_counts(alerts)
    totals = [sum(c.values()) for c in counts]
    top = sorted(counts[0].items(), key=lambda kv: -kv[1])[:3] if counts else []
    kpis.append(
        tile(
            "Alerts",
            [fmt_number(v) for v in totals],
            sub=" · ".join(f"{c} {k.lower()}" for k, c in top) or "No alerts",
            tone="info" if any(totals) else "ok",
        )
    )
    return kpis


def get_dataset_items(config: Settings, summary: BaseDescription, alerts: list) -> list:
    """Returns the dataset overview (at the top of the report)

    Args:
        config: settings object
        summary: the calculated summary
        alerts: the alerts

    Returns:
        A list with components for the dataset overview (overview, reproduction, alerts)
    """

    items: List[Renderable] = [get_dataset_overview(config, summary)]

    metadata = {key: config.dataset.dict()[key] for key in config.dataset.dict().keys()}

    if len(metadata) > 0 and any(len(value) > 0 for value in metadata.values()):
        items.append(get_dataset_schema(config, metadata))

    column_details = {
        key: config.variables.descriptions[key]
        for key in config.variables.descriptions.keys()
    }

    if len(column_details) > 0:
        items.append(get_dataset_column_definitions(config, column_details))

    if summary.time_index_analysis:
        items.append(get_timeseries_items(config, summary))

    if alerts:
        items.append(get_dataset_alerts(config, alerts))

    items.append(get_dataset_reproduction(config, summary))

    return items
