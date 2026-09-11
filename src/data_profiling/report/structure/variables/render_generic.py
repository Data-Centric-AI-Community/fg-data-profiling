from data_profiling.config import Settings
from data_profiling.report.formatters import fmt, fmt_bytesize, fmt_percent
from data_profiling.report.presentation.core import HTML, Container, Table, VariableInfo


def render_generic(config: Settings, summary: dict) -> dict:
    info = VariableInfo(
        anchor_id=summary["varid"],
        alerts=summary["alerts"],
        var_type=summary["cast_type"] or "Unsupported",
        var_name=summary["varname"],
        description=summary["description"],
        style=config.html.style,
    )

    table = Table(
        [
            {
                "name": "Missing",
                "value": fmt(summary["n_missing"]),
                "alert": "n_missing" in summary["alert_fields"],
            },
            {
                "name": "Missing (%)",
                "value": fmt_percent(summary["p_missing"]),
                "alert": "p_missing" in summary["alert_fields"],
            },
            {
                "name": "Memory size",
                "value": fmt_bytesize(summary["memory_size"]),
                "alert": False,
            },
        ],
        style=config.html.style,
    )

    return {
        "top": Container([info, table, HTML("")], sequence_type="grid"),
        "bottom": None,
    }
