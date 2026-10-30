from data_profiling.config import Settings
from data_profiling.report.formatters import fmt, fmt_bytesize, fmt_numeric, fmt_percent
from data_profiling.report.presentation.core import (
    HTML,
    Container,
    Image,
    Table,
    VariableInfo,
)
from data_profiling.visualisation.plot import scatter_complex


def render_complex(config: Settings, summary: dict) -> dict:
    varid = summary["varid"]
    template_variables = {}
    image_format = config.plot.image_format

    # Top
    info = VariableInfo(
        summary["varid"],
        summary["varname"],
        "Complex number (&Copf;)",
        summary["alerts"],
        summary["description"],
        style=config.html.style,
    )

    table1 = Table(
        [
            {"name": "Distinct", "value": fmt(summary["n_distinct"])},
            {
                "name": "Distinct (%)",
                "value": fmt_percent(summary["p_distinct"]),
            },
            {"name": "Missing", "value": fmt(summary["n_missing"])},
            {
                "name": "Missing (%)",
                "value": fmt_percent(summary["p_missing"]),
            },
            {
                "name": "Memory size",
                "value": fmt_bytesize(summary["memory_size"]),
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
            },
            {
                "name": "Minimum",
                "value": fmt_numeric(summary["min"], precision=config.report.precision),
            },
            {
                "name": "Maximum",
                "value": fmt_numeric(summary["max"], precision=config.report.precision),
            },
            {
                "name": "Zeros",
                "value": fmt_numeric(
                    summary["n_zeros"], precision=config.report.precision
                ),
            },
            {"name": "Zeros (%)", "value": fmt_percent(summary["p_zeros"])},
        ],
        style=config.html.style,
    )

    placeholder = HTML("")

    template_variables["top"] = Container(
        [info, table1, table2, placeholder], sequence_type="grid"
    )

    # Bottom
    items = [
        Image(
            scatter_complex(config, summary["scatter_data"]),
            image_format=image_format,
            alt="Scatterplot",
            caption="Scatterplot in the complex plane",
            name="Scatter",
            anchor_id=f"{varid}scatter",
        )
    ]

    bottom = Container(items, sequence_type="tabs", anchor_id=summary["varid"])

    template_variables["bottom"] = bottom

    return template_variables
