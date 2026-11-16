"""Colour tokens for the HTML report shell.

The report is styled through CSS custom properties (``--dp-*``) that are
defined per colour mode in ``wrapper/assets/style.css``. The only tokens that
depend on the configuration are the accent shades, derived here from
``html.style.primary_colors[0]`` so links, buttons and badges stay readable in
both light and dark mode.
"""
from typing import Any, Dict

from data_profiling.visualisation.charts import palette as pal

LIGHT_SURFACE = "#ffffff"
DARK_SURFACE = "#141414"


def _channel(v: float) -> float:
    v = v / 255
    return v / 12.92 if v <= 0.03928 else ((v + 0.055) / 1.055) ** 2.4


def luminance(color: str) -> float:
    r, g, b = pal.hex_to_rgb255(color)
    return 0.2126 * _channel(r) + 0.7152 * _channel(g) + 0.0722 * _channel(b)


def contrast(c1: str, c2: str) -> float:
    l1, l2 = sorted((luminance(c1), luminance(c2)), reverse=True)
    return (l1 + 0.05) / (l2 + 0.05)


def readable_on(color: str, background: str, target: float = 4.5) -> str:
    """Shift ``color`` towards black or white until it reaches ``target``
    contrast on ``background`` (WCAG 2.x ratio)."""
    toward = "#000000" if luminance(background) > 0.5 else "#ffffff"
    shade = color
    step = 0.0
    while contrast(shade, background) < target and step < 1.0:
        step = round(step + 0.05, 2)
        shade = pal.mix(color, toward, step)
    return shade


def theme_tokens(style: Any) -> Dict[str, Any]:
    """Template context for the report shell, from ``html.style``."""
    try:
        accent = pal.to_hex(style.accent_color)
        mode = style.theme_mode
        transparent = bool(style.transparent_background)
    except (AttributeError, IndexError, TypeError):
        accent, mode, transparent = "#0d6efd", "light", False
    if not accent.startswith("#"):
        accent = "#0d6efd"

    light_link = readable_on(accent, LIGHT_SURFACE)
    dark_link = readable_on(accent, DARK_SURFACE)
    return {
        "mode": mode,
        "transparent": transparent,
        "accent": accent,
        "light": {
            "link": light_link,
            "strong": pal.mix(light_link, "#000000", 0.25),
            "soft": pal.mix(accent, LIGHT_SURFACE, 0.88),
            "rgb": ", ".join(str(v) for v in pal.hex_to_rgb255(light_link)),
        },
        "dark": {
            "link": dark_link,
            "strong": pal.mix(dark_link, "#ffffff", 0.3),
            "soft": pal.mix(accent, DARK_SURFACE, 0.72),
            "rgb": ", ".join(str(v) for v in pal.hex_to_rgb255(dark_link)),
        },
    }
