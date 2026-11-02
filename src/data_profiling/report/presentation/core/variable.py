from typing import Any, Callable, Optional

from data_profiling.report.presentation.core.item_renderer import ItemRenderer
from data_profiling.report.presentation.core.renderable import Renderable


class Variable(ItemRenderer):
    def __init__(
        self,
        top: Renderable,
        bottom: Optional[Renderable] = None,
        ignore: bool = False,
        var_type: Optional[str] = None,
        n_alerts: int = 0,
        **kwargs,
    ):
        super().__init__(
            "variable",
            {
                "top": top,
                "bottom": bottom,
                "ignore": ignore,
                "var_type": var_type,
                "n_alerts": n_alerts,
            },
            **kwargs,
        )

    def __str__(self):
        top_text = str(self.content["top"]).replace("\n", "\n\t")
        bottom_text = str(self.content["bottom"]).replace("\n", "\n\t")

        text = "Variable\n"
        text += f"- top: {top_text}"
        text += f"- bottom: {bottom_text}"
        return text

    def __repr__(self):
        return "Variable"

    def render(self) -> Any:
        raise NotImplementedError()

    @classmethod
    def convert_to_class(cls, obj: Renderable, flv: Callable) -> None:
        obj.__class__ = cls
        if "top" in obj.content and obj.content["top"] is not None:
            flv(obj.content["top"])
        if "bottom" in obj.content and obj.content["bottom"] is not None:
            flv(obj.content["bottom"])
