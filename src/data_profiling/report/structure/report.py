import os
from typing import List, Sequence

import pandas as pd
from tqdm.auto import tqdm

from data_profiling.config import Settings
from data_profiling.model import BaseDescription
from data_profiling.model.alerts import AlertType
from data_profiling.model.handler import get_render_map
from data_profiling.report.presentation.core import (
    HTML,
    Collapse,
    Container,
    Dropdown,
    Duplicate,
)
from data_profiling.report.presentation.core import Image as ImageWidget
from data_profiling.report.presentation.core import Sample, ToggleButton, Variable
from data_profiling.report.presentation.core.renderable import Renderable
from data_profiling.report.presentation.core.root import Root
from data_profiling.report.structure.correlations import get_correlation_items
from data_profiling.report.structure.overview import get_dataset_items
from data_profiling.utils.dataframe import slugify


def get_missing_items(config: Settings, summary: BaseDescription) -> list:
    """Return the missing diagrams

    Args:
        config: report Settings object
        summary: the dataframe summary

    Returns:
        A list with the missing diagrams
    """
    items = [
        ImageWidget(
            item["matrix"],
            image_format=config.plot.image_format,
            alt=item["name"],
            name=item["name"],
            anchor_id=key,
            caption=item["caption"],
        )
        if isinstance(item["name"], str)
        else Container(
            [
                ImageWidget(
                    item["matrix"][i],
                    image_format=config.plot.image_format,
                    alt=item["name"][i],
                    name=config.html.style._labels[i],
                    anchor_id=key,
                    caption=item["caption"][i],
                )
                for i in range(len(item["name"]))
            ],
            sequence_type="batch_grid",
            batch_size=len(config.html.style._labels),
            anchor_id=key,
            name=item["name"][0],
        )
        for key, item in summary.missing.items()
    ]

    return items


def render_variables_section(
    config: Settings, dataframe_summary: BaseDescription
) -> list:
    """Render the HTML for each of the variables in the DataFrame.

    Args:
        config: report Settings object
        dataframe_summary: The statistics for each variable.

    Returns:
        The rendered HTML, where each row represents a variable.
    """

    templs = []

    descriptions = config.variables.descriptions
    show_description = config.show_variable_description
    reject_variables = config.reject_variables

    render_map = get_render_map()

    for idx, summary in dataframe_summary.variables.items():
        # Common template variables
        if not isinstance(dataframe_summary.alerts, tuple):
            alerts = [
                alert.fmt()
                for alert in dataframe_summary.alerts
                if alert.column_name == idx
            ]

            alert_fields = {
                field
                for alert in dataframe_summary.alerts
                if alert.column_name == idx
                for field in alert.fields
            }

            alert_types = {
                alert.alert_type
                for alert in dataframe_summary.alerts
                if alert.column_name == idx
            }
        else:
            alerts = tuple(
                [alert.fmt() for alert in summary_alerts if alert.column_name == idx]
                for summary_alerts in dataframe_summary.alerts
            )  # type: ignore

            alert_fields = {
                field
                for summary_alerts in dataframe_summary.alerts
                for alert in summary_alerts
                if alert.column_name == idx
                for field in alert.fields
            }

            alert_types = {
                alert.alert_type
                for summary_alerts in dataframe_summary.alerts
                for alert in summary_alerts
                if alert.column_name == idx
            }

        template_variables = {
            "varname": idx,
            "varid": hash(idx),
            "alerts": alerts,
            "description": descriptions.get(idx, "") if show_description else "",
            "alert_fields": alert_fields,
        }

        template_variables.update(summary)

        # Per type template variables
        if isinstance(summary["type"], list):
            types = set(summary["type"])
            if len(types) == 1:
                variable_type = list(types)[0]
            else:
                # This logic may be treated by the typeset
                if (
                    types == {"Numeric", "Categorical"}
                    or types == {"Categorical", "Unsupported"}
                    or types == {"Categorical", "Text"}
                ):
                    # Treating numeric as categorical, if one is unsupported, still render as categorical
                    variable_type = "Categorical"
                else:
                    raise ValueError(f"Types for {idx} are not compatible: {types}")
        else:
            variable_type = summary["type"]
        render_map_type = render_map.get(variable_type, render_map["Unsupported"])
        template_variables.update(render_map_type(config, template_variables))

        # Ignore these
        if reject_variables:
            ignore = AlertType.REJECTED in alert_types
        else:
            ignore = False

        bottom = None
        if "bottom" in template_variables and template_variables["bottom"] is not None:
            btn = ToggleButton("More details", anchor_id=template_variables["varid"])
            bottom = Collapse(btn, template_variables["bottom"])

        var = Variable(
            template_variables["top"],
            bottom=bottom,
            anchor_id=template_variables["varid"],
            name=idx,
            ignore=ignore,
        )

        templs.append(var)

    return templs


def get_duplicates_items(
    config: Settings, duplicates: pd.DataFrame
) -> List[Renderable]:
    """Create the list of duplicates items

    Args:
        config: settings object
        duplicates: DataFrame of duplicates

    Returns:
        List of duplicates items to show in the interface.
    """
    items: List[Renderable] = []
    if duplicates is not None and len(duplicates) > 0:
        if isinstance(duplicates, list):
            generator = (d is None for d in duplicates)
            if any(generator):
                return items

            for idx, df in enumerate(duplicates):
                items.append(
                    Duplicate(
                        duplicate=df,
                        name=config.html.style._labels[idx],
                        anchor_id="duplicates",
                    )
                )
        else:
            items.append(
                Duplicate(
                    duplicate=duplicates,
                    name="Most frequently occurring",
                    anchor_id="duplicates",
                )
            )
    return items


def get_definition_items(definitions: pd.DataFrame) -> Sequence[Renderable]:
    """Create the list of duplicates items

    Args:
        definitions: DataFrame of column definitions

    Returns:
        List of column definitions to show in the interface.
    """
    items = []
    if definitions is not None and len(definitions) > 0:
        items.append(
            Duplicate(
                duplicate=definitions,
                name="Columns",
                anchor_id="definitions",
            )
        )
    return items


def get_sample_items(config: Settings, sample: dict) -> List[Renderable]:
    """Create the list of sample items

    Args:
        config: settings object
        sample: dict of samples

    Returns:
        List of sample items to show in the interface.
    """
    items: List[Renderable] = []
    if isinstance(sample, tuple):
        for s in zip(*sample):
            items.append(
                Container(
                    [
                        Sample(
                            sample=obj.data,
                            name=config.html.style._labels[idx],
                            anchor_id=obj.id,
                            caption=obj.caption,
                        )
                        for idx, obj in enumerate(s)
                    ],
                    sequence_type="batch_grid",
                    batch_size=len(sample),
                    anchor_id=f"sample_{slugify(s[0].name)}",
                    name=s[0].name,
                )
            )
    else:
        for obj in sample:
            items.append(
                Sample(
                    sample=obj.data,
                    name=obj.name,
                    anchor_id=obj.id,
                    caption=obj.caption,
                )
            )
    return items


def get_interactions(config: Settings, interactions: dict) -> list:
    """Returns the interaction components for the report

    Args:
        config: report Settings object
        interactions: a nested dict containing the scatter plots

    Returns:
        A list of components for the interaction section of the report
    """
    titems: List[Renderable] = []

    for x_idx, (x_col, y_cols) in enumerate(interactions.items()):
        items: List[Renderable] = []
        for y_idx, (y_col, splot) in enumerate(y_cols.items()):
            if not isinstance(splot, list):
                items.append(
                    ImageWidget(
                        splot,
                        image_format=config.plot.image_format,
                        alt=f"{x_col} x {y_col}",
                        anchor_id=f"interactions_{x_idx}_{y_idx}",
                        name=y_col,
                    )
                )
            else:
                items.append(
                    Container(
                        [
                            ImageWidget(
                                zplot,
                                image_format=config.plot.image_format,
                                alt=f"{x_col} x {y_col}",
                                anchor_id=f"interactions_{x_idx}_{y_idx}",
                                name=config.html.style._labels[idx],
                            )
                            if zplot != ""
                            else HTML(
                                f"<h4 class='indent'>{config.html.style._labels[idx]}</h4><br />"
                                f"<em>Interaction plot not present for dataset</em>"
                            )
                            for idx, zplot in enumerate(splot)
                        ],
                        sequence_type="batch_grid",
                        batch_size=len(config.html.style._labels),
                        anchor_id=f"interactions_{x_idx}_{y_idx}",
                        name=y_col,
                    )
                )

        titems.append(
            Container(
                items,
                sequence_type="tabs" if len(items) <= 10 else "select",
                name=x_col,
                nested=len(interactions) > 10,
                anchor_id=f"interactions_{x_idx}",
            )
        )
    return titems


def get_report_structure(config: Settings, summary: BaseDescription) -> Root:
    """Generate a HTML report from summary statistics and a given sample.

    Args:
      config: report Settings object
      summary: Statistics to use for the overview, variables, correlations and missing values.

    Returns:
      The profile report in HTML format
    """
    disable_progress_bar = not config.progress_bar
    with tqdm(
        total=1, desc="Generate report structure", disable=disable_progress_bar
    ) as pbar:
        alerts = summary.alerts

        section_items: List[Renderable] = [
            Container(
                get_dataset_items(config, summary, alerts),
                sequence_type="overview_tabs",
                name="Overview",
                anchor_id="overview",
                oss=not bool(os.getenv("YDATA_SUPPRESS_BANNER", "")),
            ),
        ]

        if len(summary.variables) > 0:
            section_items.append(
                Dropdown(
                    name="Variables",
                    anchor_id="variables-dropdown",
                    id="variables-dropdown",
                    is_row=False,
                    classes=["form-select"],
                    items=list(summary.variables),
                    item=Container(
                        render_variables_section(config, summary),
                        sequence_type="accordion",
                        name="Variables",
                        anchor_id="variables",
                    ),
                )
            )

        scatter_items = get_interactions(config, summary.scatter)
        if len(scatter_items) > 0:
            section_items.append(
                Container(
                    scatter_items,
                    sequence_type="tabs" if len(scatter_items) <= 10 else "select",
                    name="Interactions",
                    anchor_id="interactions",
                ),
            )

        corr = get_correlation_items(config, summary)
        if corr is not None:
            section_items.append(corr)

        missing_items = get_missing_items(config, summary)
        if len(missing_items) > 0:
            section_items.append(
                Container(
                    missing_items,
                    sequence_type="tabs",
                    name="Missing values",
                    anchor_id="missing",
                )
            )

        sample_items = get_sample_items(config, summary.sample)
        if len(sample_items) > 0:
            section_items.append(
                Container(
                    items=sample_items,
                    sequence_type="tabs",
                    name="Sample",
                    anchor_id="sample",
                )
            )

        duplicate_items = get_duplicates_items(config, summary.duplicates)
        if len(duplicate_items) > 0:
            section_items.append(
                Container(
                    items=duplicate_items,
                    sequence_type="batch_grid",
                    batch_size=len(duplicate_items),
                    name="Duplicate rows",
                    anchor_id="duplicate",
                )
            )

        sections = Container(
            section_items,
            name="Root",
            sequence_type="sections",
            full_width=config.html.full_width,
        )
        pbar.update()

    footer = HTML(
        content='Report generated by <a href="https://ydata.ai/?utm_source=opensource&utm_medium=pandasprofiling&utm_campaign=report">YData</a>.'
    )

    return Root("Root", sections, footer, style=config.html.style)
