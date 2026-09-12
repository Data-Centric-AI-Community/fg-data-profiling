from typing import Any, Dict

from data_profiling.config import Settings
from data_profiling.report.formatters import fmt, fmt_bytesize, fmt_percent
from data_profiling.report.presentation.core import Container, Table, VariableInfo
from data_profiling.report.utils import image_or_empty
from data_profiling.visualisation.plot import histogram, mini_histogram


def render_date(config: Settings, summary: Dict[str, Any]) -> Dict[str, Any]:
    varid = summary["varid"]
    template_variables = {}

    image_format = config.plot.image_format

    # Top
    info = VariableInfo(
        summary["varid"],
        summary["varname"],
        "Date",
        summary["alerts"],
        summary["description"],
        style=config.html.style,
    )

    table1 = Table(
        [
            {
                "name": "Distinct",
                "value": fmt(summary["n_distinct"]),
                "alert": False,
            },
            {
                "name": "Distinct (%)",
                "value": fmt_percent(summary["p_distinct"]),
                "alert": False,
            },
            {
                "name": "Missing",
                "value": fmt(summary["n_missing"]),
                "alert": False,
            },
            {
                "name": "Missing (%)",
                "value": fmt_percent(summary["p_missing"]),
                "alert": False,
            },
            {
                "name": "Memory size",
                "value": fmt_bytesize(summary["memory_size"]),
                "alert": False,
            },
        ],
        style=config.html.style,
    )

    table2 = Table(
        [
            {"name": "Minimum", "value": fmt(summary["min"]), "alert": False},
            {"name": "Maximum", "value": fmt(summary["max"]), "alert": False},
            {
                "name": "Invalid dates",
                "value": fmt(summary.get("n_invalid_dates", 0)),
                "alert": False,
            },
            {
                "name": "Invalid dates (%)",
                "value": fmt_percent(summary.get("p_invalid_dates", 0.0)),
                "alert": False,
            },
        ],
        style=config.html.style,
    )

    summary_histogram = summary.get("histogram", [])

    mini_hist_data = None

    if summary_histogram:
        if isinstance(summary_histogram, list):
            mini_hist_data = mini_histogram(
                config,
                [x[0] for x in summary["histogram"]],
                [x[1] for x in summary["histogram"]],
                date=True,
            )
        else:
            mini_hist_data = mini_histogram(
                config, summary["histogram"][0], summary["histogram"][1], date=True
            )

    mini_histo = image_or_empty(
        mini_hist_data,
        alt="Mini histogram",
        image_format=image_format,
        name="Mini Histogram",
        anchor_id=f"{varid}minihistogram",
    )

    template_variables["top"] = Container(
        [info, table1, table2, mini_histo], sequence_type="grid"
    )

    hist_data = None
    hist_caption = None

    if summary_histogram:
        if isinstance(summary_histogram, list):
            hist_data = histogram(
                config,
                [x[0] for x in summary["histogram"]],
                [x[1] for x in summary["histogram"]],
                date=True,
            )
        else:
            hist_data = histogram(
                config, summary["histogram"][0], summary["histogram"][1], date=True
            )

        n_bins = len(summary["histogram"][1]) - 1 if summary["histogram"] else 0
        hist_caption = (
            f"<strong>Histogram with fixed size bins</strong> (bins={n_bins})"
        )

    hist = image_or_empty(
        hist_data,
        image_format=image_format,
        alt="Histogram",
        caption=hist_caption,
        name="Histogram",
        anchor_id=f"{varid}histogram",
    )

    # Bottom
    bottom = Container(
        [hist],
        sequence_type="tabs",
        anchor_id=summary["varid"],
    )

    template_variables["bottom"] = bottom

    return template_variables
