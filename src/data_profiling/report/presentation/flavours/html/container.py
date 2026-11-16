"""
    Container class definition
"""
from data_profiling.report.presentation.core.container import Container
from data_profiling.report.presentation.flavours.html import templates


class HTMLContainer(Container):
    def render(self) -> str:
        if self.sequence_type in ["list", "accordion"]:
            return templates.template("sequence/list.html").render(
                anchor_id=self.content["anchor_id"], items=self.content["items"]
            )
        elif self.sequence_type == "named_list":
            return templates.template("sequence/named_list.html").render(
                anchor_id=self.content["anchor_id"], items=self.content["items"]
            )
        elif self.sequence_type == "overview_tabs":
            return templates.template("sequence/overview_tabs.html").render(
                tabs=self.content["items"],
                anchor_id=self.content["anchor_id"],
                nested=self.content["nested"],
                oss=self.oss,
                kpis=self.content.get("kpis"),
            )
        elif self.sequence_type == "tabs":
            return templates.template("sequence/tabs.html").render(
                tabs=self.content["items"],
                anchor_id=self.content["anchor_id"],
                nested=self.content["nested"],
            )
        elif self.sequence_type == "select":
            return templates.template("sequence/select.html").render(
                tabs=self.content["items"],
                anchor_id=self.content["anchor_id"],
                nested=self.content["nested"],
            )
        elif self.sequence_type == "sections":
            return templates.template("sequence/sections.html").render(
                sections=self.content["items"],
                full_width=self.content["full_width"],
            )
        elif self.sequence_type == "grid":
            return templates.template("sequence/grid.html").render(
                items=self.content["items"]
            )
        elif self.sequence_type == "batch_grid":
            return templates.template("sequence/batch_grid.html").render(
                items=self.content["items"],
                batch_size=self.content["batch_size"],
                titles=self.content.get("titles", True),
                subtitles=self.content.get("subtitles", False),
            )
        elif self.sequence_type == "interactions":
            return templates.template("sequence/interactions.html").render(
                items=self.content["items"],
                anchor_id=self.content["anchor_id"],
                stats={
                    (p["x"], p["y"]): p["stats"]
                    for p in self.content.get("pairs") or []
                },
                default_pair=self.content.get("default_pair"),
            )
        elif self.sequence_type == "correlations":
            return templates.template("sequence/correlations.html").render(
                items=self.content["items"],
                anchor_id=self.content["anchor_id"],
                methods=self.content.get("methods") or [],
                pairs=self.content.get("pairs") or [],
            )
        elif self.sequence_type == "cards":
            return templates.template("sequence/cards.html").render(
                items=self.content["items"],
                anchor_id=self.content["anchor_id"],
                counts=self.content.get("counts"),
                labels=self.content.get("labels") or [],
                colors=self.content.get("colors") or [],
            )
        elif self.sequence_type == "scores":
            return templates.template("sequence/scores.html").render(
                anchor_id=self.content["anchor_id"], items=self.content["items"]
            )
        else:
            raise ValueError("Template not understood", self.sequence_type)
