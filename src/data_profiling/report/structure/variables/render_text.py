from typing import Any, Dict, List

from data_profiling.config import Settings
from data_profiling.report.formatters import fmt, fmt_bytesize, fmt_percent
from data_profiling.report.presentation.core import (
    Container,
    FrequencyTable,
    FrequencyTableSmall,
    Image,
    Table,
)
from data_profiling.report.presentation.core.variable_info import VariableInfo
from data_profiling.report.structure.variables.render_categorical import (
    _get_n,
    freq_table,
    render_categorical,
    render_categorical_frequency,
    render_categorical_length,
    render_categorical_unicode,
)
from data_profiling.report.structure.variables.render_common import render_common
from data_profiling.visualisation.plot import plot_word_cloud


def _top_words(word_counts: Any, k: int) -> Any:
    """Most frequent ``k`` words, without an "Other values" remainder."""
    if isinstance(word_counts, list):
        return [wc.sort_values(ascending=False).head(k) for wc in word_counts]
    return word_counts.sort_values(ascending=False).head(k)


def render_text(config: Settings, summary: Dict[str, Any]) -> Dict[str, Any]:
    if config.vars.text.redact:
        render = render_categorical(config, summary)
        return render

    varid = summary["varid"]
    words = config.vars.text.words
    characters = config.vars.text.characters
    length = config.vars.text.length

    template_variables = render_common(config, summary)

    top_items: List[Any] = []
    var_info = VariableInfo(
        anchor_id=varid,
        var_name=summary["varname"],
        var_type=summary["type"],
        alerts=summary["alerts"],
        description=summary["description"],
        style=config.html.style,
    )
    top_items.append(var_info)

    table = Table(
        [
            {
                "name": "Distinct",
                "value": fmt(summary["n_distinct"]),
                "alert": "n_distinct" in summary["alert_fields"],
            },
            {
                "name": "Distinct (%)",
                "value": fmt_percent(summary["p_distinct"]),
                "alert": "p_distinct" in summary["alert_fields"],
            },
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
    top_items.append(table)

    if words and "word_counts" in summary:
        mini_wordcloud = Image(
            plot_word_cloud(config, summary["word_counts"]),
            image_format=config.plot.image_format,
            alt="Mini wordcloud",
        )
        top_items.append(mini_wordcloud)
        top_words = _top_words(summary["word_counts"], 5)
        top_items.append(
            FrequencyTableSmall(
                freq_table(
                    freqtable=top_words,
                    n=_get_n(top_words),
                    max_number_to_print=5,
                ),
                redact=config.vars.text.redact,
                name="Top 5 words",
            )
        )
    template_variables["top"] = Container(top_items, sequence_type="grid")

    # ============================================================================================

    bottom_items = []
    overview_items = []
    # length isn't being computed for categorical in spark
    if length and "max_length" in summary:
        length_table, length_histo = render_categorical_length(config, summary, varid)
        overview_items.append(length_table)

    # characters isn't being computed for categorical in spark
    unitab = None
    if characters and "category_alias_counts" in summary:
        overview_table_char, unitab = render_categorical_unicode(config, summary, varid)
        overview_items.append(overview_table_char)

    unique_stats = render_categorical_frequency(config, summary, varid)
    overview_items.append(unique_stats)

    if not config.vars.text.redact:
        rows = ("1st row", "2nd row", "3rd row", "4th row", "5th row")

        if isinstance(summary["first_rows"], list):
            sample = Table(
                [
                    {
                        "name": name,
                        "value": fmt(value),
                        "alert": False,
                    }
                    for name, *value in zip(rows, *summary["first_rows"])
                ],
                name="Sample",
                style=config.html.style,
            )
        else:
            sample = Table(
                [
                    {
                        "name": name,
                        "value": fmt(value),
                        "alert": False,
                    }
                    for name, value in zip(rows, summary["first_rows"])
                ],
                name="Sample",
                style=config.html.style,
            )
        overview_items.append(sample)
    overview = Container(
        overview_items,
        name="Overview",
        anchor_id=f"{varid}overview",
        sequence_type="batch_grid",
        batch_size=len(overview_items),
        titles=False,
    )
    bottom_items.append(overview)

    if words and "word_counts" in summary:
        woc = freq_table(
            freqtable=summary["word_counts"],
            n=_get_n(summary["word_counts"]),
            max_number_to_print=10,
        )

        fqwo = FrequencyTable(
            woc,
            name="Common words",
            anchor_id=f"{varid}cwo",
            redact=config.vars.text.redact,
        )

        image = Image(
            plot_word_cloud(config, summary["word_counts"]),
            image_format=config.plot.image_format,
            alt="Wordcloud",
        )

        bottom_items.append(
            Container(
                [fqwo, image],
                name="Words",
                anchor_id=f"{varid}word",
                sequence_type="grid",
            )
        )

    if unitab is not None:
        bottom_items.append(
            Container(
                [unitab],
                name="Characters",
                anchor_id=f"{varid}characters",
                sequence_type="grid",
            )
        )

    template_variables["bottom"] = Container(
        bottom_items, sequence_type="tabs", anchor_id=f"{varid}bottom"
    )

    return template_variables
