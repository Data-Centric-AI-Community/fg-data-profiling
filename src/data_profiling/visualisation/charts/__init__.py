"""Native chart rendering: data-driven specs rendered as inline SVG.

The public plot functions in ``data_profiling.visualisation`` keep their
signatures and return type (``str``). When the native renderer is active they
return inline ``<svg>`` markup instead of a matplotlib figure encoded as an
image, so every existing call site, presentation item and flavour keeps working.

Renderer selection (``config.plot.renderer``):

* ``"native"``: inline SVG built from chart specs (no images).
* ``"matplotlib"``: the legacy matplotlib renderer, unchanged.
* unset (default): native, except when ``plot.image_format`` is ``png``; users
  who asked for raster images keep getting them.
"""

from data_profiling.config import ImageType, PlotRenderer, Settings
from data_profiling.visualisation.charts.svg import render


def use_native(config: Settings) -> bool:
    renderer = config.plot.renderer
    if renderer is None:
        return config.plot.image_format != ImageType.png
    return PlotRenderer(renderer) == PlotRenderer.native


def is_inline_svg(image: str) -> bool:
    """True for markup produced by the native renderer."""
    return isinstance(image, str) and image.lstrip().startswith("<svg")


__all__ = ["render", "use_native", "is_inline_svg"]
