from typing import List

from ipywidgets import widgets

from data_profiling.report.presentation.core.container import Container
from data_profiling.report.presentation.core.renderable import Renderable


def get_name(item: Renderable) -> str:
    if hasattr(item, "name"):
        return item.name
    else:
        return item.anchor_id


def get_tabs(items: List[Renderable]) -> widgets.Tab:
    children = []
    titles = []
    for item in items:
        children.append(item.render())
        titles.append(get_name(item))

    tab = widgets.Tab()
    tab.children = children
    for id, title in enumerate(titles):
        tab.set_title(id, title)
    return tab


def get_list(items: List[Renderable]) -> widgets.VBox:
    return widgets.VBox([item.render() for item in items])


def get_named_list(items: List[Renderable]) -> widgets.VBox:
    return widgets.VBox(
        [
            widgets.VBox(
                [widgets.HTML(f"<strong>{get_name(item)}</strong>"), item.render()]
            )
            for item in items
        ]
    )


def get_row(items: List[Renderable]) -> widgets.GridBox:
    if len(items) == 1:
        layout = widgets.Layout(width="100%", grid_template_columns="100%")
    elif len(items) == 2:
        layout = widgets.Layout(width="100%", grid_template_columns="50% 50%")
    elif len(items) == 3:
        layout = widgets.Layout(width="100%", grid_template_columns="25% 25% 50%")
    elif len(items) == 4:
        layout = widgets.Layout(width="100%", grid_template_columns="25% 25% 25% 25%")
    else:
        raise ValueError("Layout undefined for this number of columns")

    return widgets.GridBox([item.render() for item in items], layout=layout)


def get_batch_grid(
    items: List[Renderable], batch_size: int, titles: bool, subtitles: bool
) -> widgets.GridBox:
    layout = widgets.Layout(
        width="100%",
        grid_template_columns=" ".join([f"{int(100 / batch_size)}%"] * batch_size),
    )
    out = []
    for item in items:
        if subtitles:
            out.append(
                widgets.VBox(
                    [widgets.HTML(f"<h5><em>{ item.name }</em></h5>"), item.render()]
                )
            )
        elif titles:
            out.append(
                widgets.VBox([widgets.HTML(f"<h4>{ item.name }</h4>"), item.render()])
            )
        else:
            out.append(item.render())

    return widgets.GridBox(out, layout=layout)


def get_accordion(items: List[Renderable]) -> widgets.Accordion:
    children = []
    titles = []
    for item in items:
        children.append(item.render())
        titles.append(get_name(item))

    accordion = widgets.Accordion(children=children)
    for id, title in enumerate(titles):
        accordion.set_title(id, title)

    return accordion


class WidgetContainer(Container):
    def render(self) -> widgets.Widget:
        if self.sequence_type == "list":
            widget = get_list(self.content["items"])
        elif self.sequence_type == "named_list":
            widget = get_named_list(self.content["items"])
        elif self.sequence_type in ["tabs", "sections", "select", "overview_tabs"]:
            widget = get_tabs(self.content["items"])
        elif self.sequence_type == "accordion":
            widget = get_accordion(self.content["items"])
        elif self.sequence_type == "grid":
            widget = get_row(self.content["items"])
        elif self.sequence_type == "batch_grid":
            widget = get_batch_grid(
                self.content["items"],
                self.content["batch_size"],
                self.content.get("titles", True),
                self.content.get("subtitles", False),
            )
        else:
            raise ValueError("widget type not understood", self.sequence_type)

        return widget
