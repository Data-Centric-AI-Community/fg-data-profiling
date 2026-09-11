from data_profiling.config import Settings
from data_profiling.report.formatters import fmt, fmt_bytesize, fmt_numeric, fmt_percent
from data_profiling.report.presentation.core import (
    Container,
    FrequencyTable,
    Table,
    VariableInfo,
)
from data_profiling.report.structure.variables.render_common import render_common
from data_profiling.report.utils import image_or_empty
from data_profiling.visualisation.plot import histogram, mini_histogram


def render_count(config: Settings, summary: dict) -> dict:
    template_variables = render_common(config, summary)
    image_format = config.plot.image_format

    # Top
    info = VariableInfo(
        summary["varid"],
        summary["varname"],
        "Real number (&Ropf; / &Ropf;<sub>&ge;0</sub>)",
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
        ],
        style=config.html.style,
    )

    table2 = Table(
        [
            {
                "name": "Mean",
                "value": fmt_numeric(
                    summary["mean"], precision=config.report.precision
                ),
                "alert": False,
            },
            {
                "name": "Minimum",
                "value": fmt_numeric(summary["min"], precision=config.report.precision),
                "alert": False,
            },
            {
                "name": "Maximum",
                "value": fmt_numeric(summary["max"], precision=config.report.precision),
                "alert": False,
            },
            {
                "name": "Zeros",
                "value": fmt(summary["n_zeros"]),
                "alert": False,
            },
            {
                "name": "Zeros (%)",
                "value": fmt_percent(summary["p_zeros"]),
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

    summary_histogram = summary.get("histogram", [])

    mini_hist_data = None

    if summary_histogram:
        mini_hist_data = mini_histogram(config, *summary["histogram"])

    mini_histo = image_or_empty(
        mini_hist_data,
        alt="Mini histogram",
        image_format=image_format,
        name="Mini Histogram",
    )

    template_variables["top"] = Container(
        [info, table1, table2, mini_histo], sequence_type="grid"
    )

    hist_data = None
    hist_caption = None

    if summary_histogram:
        hist_data = histogram(config, *summary["histogram"])
        hist_caption = f"<strong>Histogram with fixed size bins</strong> (bins={len(summary['histogram'][1]) - 1})"

    hist = image_or_empty(
        hist_data,
        alt="Histogram",
        image_format=image_format,
        caption=hist_caption,
        name="Histogram",
        anchor_id="histogram",
    )

    seqs = [hist]

    fq = FrequencyTable(
        template_variables["freq_table_rows"],
        name="Common values",
        anchor_id="common_values",
        redact=False,
    )

    evs = Container(
        [
            FrequencyTable(
                template_variables["firstn_expanded"],
                name=f"Minimum {config.n_extreme_obs} values",
                anchor_id="firstn",
                redact=False,
            ),
            FrequencyTable(
                template_variables["lastn_expanded"],
                name=f"Maximum {config.n_extreme_obs} values",
                anchor_id="lastn",
                redact=False,
            ),
        ],
        sequence_type="tabs",
        name="Extreme values",
        anchor_id="extreme_values",
    )

    template_variables["bottom"] = Container(
        [
            Container(
                seqs, sequence_type="tabs", name="Histogram(s)", anchor_id="histograms"
            ),
            fq,
            evs,
        ],
        sequence_type="tabs",
        anchor_id=summary["varid"],
    )

    return template_variables
