from data_profiling.report.presentation.core.root import Root
from data_profiling.report.presentation.flavours.html import templates
from data_profiling.report.presentation.flavours.html.theme import theme_tokens


class HTMLRoot(Root):
    def render(self, **kwargs) -> str:
        nav_items = [
            (section.name, section.anchor_id)
            for section in self.content["body"].content["items"]
        ]

        context = {**self.content, **kwargs}
        context.setdefault("dp", theme_tokens(context.get("style")))
        return templates.template("report.html").render(**context, nav_items=nav_items)
