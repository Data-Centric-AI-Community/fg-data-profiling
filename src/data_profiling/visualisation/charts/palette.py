"""Colour helpers for the native (SVG) chart renderer.

Colour math only: nothing here draws an image. matplotlib is used, when it is
installed, solely to sample named colormaps configured by the user (e.g.
``plot.correlation.cmap``); the default palettes below do not need it.
"""

from typing import List, Sequence, Tuple

# Categorical palette, fixed order (validated for colour-vision deficiency on
# adjacent pairs). Used only where the legacy renderer fell back to
# matplotlib's default colour cycle; user-configured colours always win.
CATEGORICAL: List[str] = [
    "#2a78d6",
    "#eb6834",
    "#1baf7a",
    "#eda100",
    "#e87ba4",
    "#008300",
    "#6250d6",
    "#e34948",
]

# Diverging blue <-> red with a neutral grey midpoint. Same polarity as
# matplotlib's "RdBu" (negative = red, positive = blue), which is the default
# value of ``plot.correlation.cmap`` and ``plot.missing.cmap``.
DIVERGING_RDBU: List[str] = [
    "#b4322f",
    "#e34948",
    "#ef9a8f",
    "#f0efec",
    "#86b6ef",
    "#2a78d6",
    "#184f95",
]

NEUTRAL = "#f0efec"


def hex_to_rgb255(color: str) -> Tuple[int, int, int]:
    c = color.lstrip("#")
    if len(c) == 3:
        c = "".join(ch * 2 for ch in c)
    return int(c[0:2], 16), int(c[2:4], 16), int(c[4:6], 16)


def rgb255_to_hex(rgb: Sequence[float]) -> str:
    return "#" + "".join(f"{int(round(max(0, min(255, v)))):02x}" for v in rgb)


def to_hex(color: str) -> str:
    """Normalise any colour accepted by the legacy renderer to ``#rrggbb``.

    Hex strings are handled natively; named colours (``"red"``) are resolved
    through matplotlib when it is available so existing configs keep working.
    """
    if isinstance(color, str) and color.startswith("#"):
        return rgb255_to_hex(hex_to_rgb255(color))
    try:
        from matplotlib.colors import to_hex as _mpl_to_hex

        return _mpl_to_hex(color)
    except Exception:  # pragma: no cover - fall back to the colour as given
        return str(color)


def mix(c1: str, c2: str, t: float) -> str:
    """Linear interpolation between two hex colours (t=0 -> c1, t=1 -> c2)."""
    a, b = hex_to_rgb255(c1), hex_to_rgb255(c2)
    return rgb255_to_hex([x + (y - x) * t for x, y in zip(a, b)])


def ramp(stops: Sequence[str], t: float) -> str:
    """Sample a piecewise-linear colour ramp at t in [0, 1]."""
    if t != t:  # NaN
        return NEUTRAL
    t = min(1.0, max(0.0, float(t)))
    pos = t * (len(stops) - 1)
    i = min(int(pos), len(stops) - 2)
    return mix(stops[i], stops[i + 1], pos - i)


def colormap_stops(name: str, half: bool = False, n: int = 11) -> List[str]:
    """Return colour stops for a (matplotlib-named) colormap.

    Args:
        name: colormap name from the config.
        half: keep only the upper half (used for coefficients in [0, 1]).
        n: number of stops to sample.
    """
    if name == "RdBu":
        stops = DIVERGING_RDBU
        return stops[len(stops) // 2 :] if half else list(stops)
    try:
        import numpy as np
        from matplotlib import colormaps

        cmap = colormaps[name]
        lo = 0.5 if half else 0.0
        return [
            rgb255_to_hex([v * 255 for v in cmap(x)[:3]])
            for x in np.linspace(lo, 1.0, n)
        ]
    except Exception:
        stops = DIVERGING_RDBU
        return stops[len(stops) // 2 :] if half else list(stops)


def sequential_stops(color: str) -> List[str]:
    """One-hue ramp from a light tint to the given colour (and a bit darker)."""
    color = to_hex(color)
    return [mix("#ffffff", color, 0.12), color, mix(color, "#000000", 0.35)]


def categorical(n: int, colors: Sequence[str] = ()) -> List[str]:
    """First n categorical colours; configured colours take precedence."""
    base = [to_hex(c) for c in colors] if colors else list(CATEGORICAL)
    return [base[i % len(base)] for i in range(n)]


def text_on(fill: str) -> str:
    """Pick readable label ink for text drawn on top of a filled mark."""
    r, g, b = hex_to_rgb255(to_hex(fill))
    luminance = 0.2126 * r + 0.7152 * g + 0.0722 * b
    return "#0b0b0b" if luminance > 150 else "#ffffff"
