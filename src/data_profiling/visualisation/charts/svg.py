"""Native SVG renderer for chart specs.

Produces self-contained inline ``<svg>`` markup: no images, no external
resources, no JavaScript and no element ids (so any number of charts can be
inlined in one HTML document). Text uses ``currentColor`` so charts follow the
surrounding text colour; per-mark ``<title>`` elements provide hover tooltips.
"""

import json
import math
from html import escape
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from data_profiling.visualisation.charts import palette as pal
from data_profiling.visualisation.charts.scales import (
    Linear,
    date_ticks,
    date_tooltip_format,
    fmt_count,
    fmt_date,
    fmt_num,
    nice_max,
    nice_ticks,
)
from data_profiling.visualisation.charts.specs import (
    BarSpec,
    ChartSpec,
    HeatmapSpec,
    HistogramSpec,
    LineSpec,
    MissingMatrixSpec,
    PieSpec,
    ScatterSpec,
    StackedBarSpec,
    StemGridSpec,
)

FONT = 11
SMALL = 10
CHAR_W = 0.58  # average glyph width relative to the font size
SURFACE = "#ffffff"
MUTED = 'fill="currentColor" fill-opacity="0.68"'
GRID = 'stroke="currentColor" stroke-opacity="0.12"'
AXIS = 'stroke="currentColor" stroke-opacity="0.35"'


# Tooltip lines are joined with this separator: it survives HTML minification
# (which collapses newlines) and reads naturally in native <title> tooltips.
# The report's script splits on it to render a multi-line tooltip; a first line
# without ": " is shown as the tooltip heading.
SEP = " · "


def tips(*lines: Optional[str]) -> str:
    return SEP.join(str(x) for x in lines if x)


def pct(part: float, whole: float, digits: int = 1) -> str:
    return f"{(part / whole * 100) if whole else 0:.{digits}f}%"


def _f(v: float) -> str:
    """Compact coordinate formatting (1 decimal, no trailing zeros)."""
    s = f"{v:.1f}"
    return s[:-2] if s.endswith(".0") else s


def text_width(s: str, size: float = FONT) -> float:
    return len(s) * size * CHAR_W


def truncate(s: str, n: int = 24) -> str:
    return s if len(s) <= n else s[: n - 1] + "…"


class Canvas:
    def __init__(self, width: float, height: float, kind: str, label: str = ""):
        self.width, self.height, self.kind, self.label = width, height, kind, label
        self.parts: List[str] = []
        self.data: Dict[str, str] = {}

    def add(self, s: str) -> None:
        self.parts.append(s)

    def rect(
        self,
        x: float,
        y: float,
        w: float,
        h: float,
        fill: str,
        tip: Optional[str] = None,
        extra: str = "",
        mark: bool = True,
    ) -> None:
        if w <= 0 or h <= 0:
            return
        cls = ' class="dp-mark"' if mark else ""
        body = f"<title>{escape(tip)}</title></rect>" if tip else "/>"
        self.add(
            f'<rect{cls} x="{_f(x)}" y="{_f(y)}" width="{_f(w)}" height="{_f(h)}" '
            f'fill="{fill}"{(" " + extra) if extra else ""}'
            + (">" + body if tip else body)
        )

    def hit(self, x: float, y: float, w: float, h: float, tip: str) -> None:
        """Invisible hover target, larger than the mark it describes."""
        if w <= 0 or h <= 0:
            return
        self.add(
            f'<rect class="dp-hit" x="{_f(x)}" y="{_f(y)}" width="{_f(w)}" '
            f'height="{_f(h)}" fill="currentColor" fill-opacity="0">'
            f"<title>{escape(tip)}</title></rect>"
        )

    def path(self, d: str, attrs: str, tip: Optional[str] = None) -> None:
        if tip:
            self.add(f'<path d="{d}" {attrs}><title>{escape(tip)}</title></path>')
        else:
            self.add(f'<path d="{d}" {attrs}/>')

    def line(
        self, x1: float, y1: float, x2: float, y2: float, attrs: str = GRID
    ) -> None:
        self.add(
            f'<line x1="{_f(x1)}" y1="{_f(y1)}" x2="{_f(x2)}" y2="{_f(y2)}" {attrs}/>'
        )

    def text(
        self,
        x: float,
        y: float,
        s: str,
        anchor: str = "start",
        size: float = FONT,
        attrs: str = MUTED,
        rotate: Optional[float] = None,
        weight: Optional[str] = None,
        tip: Optional[str] = None,
    ) -> None:
        a = f' text-anchor="{anchor}"' if anchor != "start" else ""
        fs = f' font-size="{_f(size)}"' if size != FONT else ""
        fw = f' font-weight="{weight}"' if weight else ""
        tr = (
            f' transform="rotate({_f(rotate)} {_f(x)} {_f(y)})"'
            if rotate is not None
            else ""
        )
        inner = escape(s) + (f"<title>{escape(tip)}</title>" if tip else "")
        cls = ' class="dp-tip-text"' if tip else ""
        self.add(
            f'<text{cls} x="{_f(x)}" y="{_f(y)}"{a}{fs}{fw}{tr} {attrs}>{inner}</text>'
        )

    def render(self) -> str:
        w, h = math.ceil(self.width), math.ceil(self.height)
        label = escape(self.label or self.kind.replace("_", " "))
        data = "".join(
            f' data-dp-{k}="{escape(v, quote=True)}"' for k, v in self.data.items()
        )
        return (
            f'<svg xmlns="http://www.w3.org/2000/svg" '
            f'class="dp-chart dp-chart-{self.kind} img-responsive center-img" '
            f'viewBox="0 0 {w} {h}" width="{w}" height="{h}" role="img" '
            f'aria-label="{label}"{data} '
            f'style="max-width:100%;height:auto;font-family:inherit;font-size:{FONT}px">'
            + "".join(self.parts)
            + "</svg>"
        )


# ----------------------------------------------------------------------------
# Shared pieces
# ----------------------------------------------------------------------------


def bar_path(x: float, y: float, w: float, h: float, end: str = "top") -> str:
    """Bar with a 4px rounded data-end and a square baseline."""
    r = min(4.0, w / 2, h) if end == "top" else min(4.0, h / 2, w)
    if r < 1:
        return f"M{_f(x)} {_f(y)}h{_f(w)}v{_f(h)}h{_f(-w)}Z"
    if end == "top":
        return (
            f"M{_f(x)} {_f(y + h)}V{_f(y + r)}Q{_f(x)} {_f(y)} {_f(x + r)} {_f(y)}"
            f"H{_f(x + w - r)}Q{_f(x + w)} {_f(y)} {_f(x + w)} {_f(y + r)}"
            f"V{_f(y + h)}Z"
        )
    return (
        f"M{_f(x)} {_f(y)}H{_f(x + w - r)}Q{_f(x + w)} {_f(y)} {_f(x + w)} {_f(y + r)}"
        f"V{_f(y + h - r)}Q{_f(x + w)} {_f(y + h)} {_f(x + w - r)} {_f(y + h)}"
        f"H{_f(x)}Z"
    )


def legend(
    c: Canvas,
    items: Sequence[Tuple[str, str, str]],
    x: float,
    y: float,
    max_width: float,
    size: float = SMALL,
) -> float:
    """Draw a wrapping legend; items are (label, colour, shape). Returns height."""
    row_h = size + 8
    cx, cy = x, y
    for label, color, shape in items:
        label = truncate(label, 32)
        w = 14 + 6 + text_width(label, size) + 14
        if cx > x and cx + w > x + max_width:
            cx, cy = x, cy + row_h
        mid = cy + row_h / 2
        if shape == "rect":
            c.rect(cx, mid - 5, 10, 10, color, extra='rx="2"', mark=False)
        else:
            dash = ' stroke-dasharray="4 3"' if shape == "dash" else ""
            c.line(
                cx,
                mid,
                cx + 14,
                mid,
                f'stroke="{color}" stroke-width="2" stroke-linecap="round"{dash}',
            )
        c.text(cx + 20, mid + size * 0.35, label, size=size)
        cx += w
    return cy + row_h - y


def x_axis_ticks(
    lo: float, hi: float, plot_w: float, is_date: bool, size: float
) -> List[Tuple[float, str]]:
    """Ticks for a horizontal axis, thinned so labels do not collide."""
    target = max(2, min(8, int(plot_w / 55)))
    for count in range(target, 1, -1):
        if is_date:
            ticks = date_ticks(lo, hi, count)
        else:
            vals = nice_ticks(lo, hi, count)
            step = vals[1] - vals[0] if len(vals) > 1 else 0
            ticks = [(v, fmt_num(v, step)) for v in vals]
        widest = max((text_width(t, size) for _, t in ticks), default=0)
        if len(ticks) <= 1 or widest * len(ticks) * 1.25 < plot_w:
            return ticks
    return ticks[:: max(1, len(ticks) // 2)]


def y_axis(
    c: Canvas,
    sy: Linear,
    ticks: Sequence[float],
    x0: float,
    x1: float,
    fmt: Callable[[float], str],
    size: float = SMALL,
    grid: bool = True,
) -> None:
    for t in ticks:
        y = sy(t)
        if grid:
            c.line(x0, y, x1, y)
        c.text(x0 - 6, y + size * 0.35, fmt(t), anchor="end", size=size)


def y_label_width(labels: Sequence[str], size: float = SMALL) -> float:
    return max((text_width(s, size) for s in labels), default=0) + 10


# ----------------------------------------------------------------------------
# Histogram
# ----------------------------------------------------------------------------


def render_histogram(spec: HistogramSpec) -> str:
    W, H = spec.width, spec.height
    size = 9 if spec.mini else SMALL
    multi = len(spec.series) > 1
    edges_all = [e for s in spec.series for e in s.edges]
    counts_all = [v for s in spec.series for v in s.counts]
    lo, hi = float(min(edges_all)), float(max(edges_all))
    ymax = nice_max(float(max(counts_all)) if counts_all else 1.0)
    yticks = nice_ticks(0, ymax, 3 if spec.mini else 4)
    ystep = yticks[1] - yticks[0] if len(yticks) > 1 else 0

    def yfmt(v: float) -> str:
        return fmt_num(v, ystep)

    left = 6 if spec.mini else 18 + y_label_width([yfmt(t) for t in yticks], size)
    top = 8.0
    c = Canvas(W, H, "histogram" if not spec.mini else "mini-histogram", spec.title)
    legend_h = 0.0
    if multi and not spec.mini:
        legend_h = legend(
            c,
            [(s.label, s.color, "rect") for s in spec.series],
            left,
            4,
            W - left - 8,
            size,
        )
        top += legend_h
    bottom = (size + 14) if spec.x_labels else 8
    plot_w, plot_h = W - left - 8, H - top - bottom
    sx = Linear(lo, hi, left, left + plot_w)
    sy = Linear(0, ymax, top + plot_h, top)

    if not spec.mini:
        y_axis(c, sy, yticks, left, left + plot_w, yfmt, size)
        c.text(
            12,
            top + plot_h / 2,
            spec.y_label,
            anchor="middle",
            size=size,
            rotate=-90,
        )

    span = hi - lo
    dfmt = date_tooltip_format(span)

    def edge_label(v: float) -> str:
        return fmt_date(v, dfmt) if spec.date else fmt_num(v)

    def bin_label(edges: Sequence[float], i: int) -> str:
        close = "]" if i == len(edges) - 2 else ")"
        return f"[{edge_label(edges[i])}, {edge_label(edges[i + 1])}{close}"

    density = spec.y_label == "Density"

    def shares(s: Any) -> Tuple[np.ndarray, np.ndarray]:
        counts = np.nan_to_num(np.asarray(s.counts, float))
        mass = counts * np.diff(s.edges) if density else counts
        total = mass.sum() or 1.0
        return mass / total, np.cumsum(mass) / total

    stats = [shares(s) for s in spec.series]
    shared_edges = all(
        len(s.edges) == len(spec.series[0].edges)
        and np.allclose(s.edges, spec.series[0].edges)
        for s in spec.series
    )
    value_name = spec.y_label or "Count"

    def series_lines(k: int, i: int) -> List[str]:
        s = spec.series[k]
        share, cum = stats[k]
        n = s.counts[i]
        name = f"{s.label} {value_name.lower()}" if multi else value_name
        prefix = f"{s.label} " if multi else ""
        return [
            f"{name}: {fmt_num(n) if density else fmt_count(n)}",
            f"{prefix}{'share' if multi else 'Share'}: {share[i] * 100:.1f}%",
            f"{prefix}{'cumulative' if multi else 'Cumulative'}: {cum[i] * 100:.1f}%",
        ]

    indexed = list(enumerate(spec.series))
    for k_idx, s in reversed(indexed) if multi else indexed:
        opacity = ' fill-opacity="0.6"' if multi else ""
        for i, n in enumerate(s.counts):
            if n is None or not np.isfinite(n) or n <= 0:
                continue
            x0, x1 = sx(s.edges[i]), sx(s.edges[i + 1])
            bw = x1 - x0
            gap = min(2.0, bw * 0.2) if bw >= 3 else 0.0
            y = sy(n)
            h = sy(0) - y
            # Shared bins get one combined, full-height hover column (below);
            # otherwise each bar carries its own tooltip.
            tip = (
                None
                if shared_edges
                else tips(bin_label(s.edges, i), *series_lines(k_idx, i))
            )
            if bw - gap >= 8:
                c.path(
                    bar_path(x0 + gap / 2, y, bw - gap, h),
                    f'class="dp-mark" fill="{s.color}"{opacity}',
                    tip,
                )
            else:
                c.rect(x0 + gap / 2, y, bw - gap, h, s.color, tip, opacity.strip())

    if shared_edges and spec.series:
        edges = spec.series[0].edges
        for i in range(len(edges) - 1):
            lines: List[str] = []
            for k_idx in range(len(spec.series)):
                lines += series_lines(k_idx, i)
            x0, x1 = sx(edges[i]), sx(edges[i + 1])
            c.hit(x0, top, x1 - x0, plot_h, tips(bin_label(edges, i), *lines))

    c.line(left, sy(0), left + plot_w, sy(0), AXIS)
    if spec.x_labels:
        for v, label in x_axis_ticks(lo, hi, plot_w, spec.date, size):
            x = sx(v)
            if x < left - 1 or x > left + plot_w + 1:
                continue
            c.line(x, sy(0), x, sy(0) + 4, AXIS)
            anchor = "middle"
            if x - text_width(label, size) / 2 < 0:
                anchor = "start"
            elif x + text_width(label, size) / 2 > W:
                anchor = "end"
            c.text(x, sy(0) + size + 6, label, anchor=anchor, size=size)
    return c.render()


# ----------------------------------------------------------------------------
# Bars
# ----------------------------------------------------------------------------


def _value_fmt(spec: BarSpec) -> Callable[[float], str]:
    if spec.value_format == "percent":
        return lambda v: f"{v * 100:.0f}%"
    return lambda v: fmt_count(v)


def render_bar(spec: BarSpec) -> str:
    if spec.horizontal:
        return _render_hbar(spec)
    return _render_vbar(spec)


def _render_hbar(spec: BarSpec) -> str:
    n_series = max(1, len(spec.series))
    multi = n_series > 1
    bar_h = 14 if multi else 16
    band = n_series * bar_h + (n_series - 1) * 2 + 8
    labels = [truncate(str(x)) for x in spec.categories]
    left = min(180, y_label_width(labels, SMALL)) + 4
    W = spec.width
    top = 6.0
    c = Canvas(W, 0, "bar", spec.title)
    if multi:
        top += legend(
            c, [(s.label, s.color, "rect") for s in spec.series], left, 2, W - left
        )
    vmax = spec.value_max or max(
        (v for s in spec.series for v in s.values if v is not None), default=1
    )
    vmax = vmax or 1
    fmt = _value_fmt(spec)
    right_pad = 64
    sx = Linear(0, vmax, left, W - right_pad)
    totals = [float(sum(v for v in s.values if v)) for s in spec.series]
    n_cat = len(spec.categories)
    for i, cat in enumerate(spec.categories):
        y0 = top + i * band + 4
        c.text(
            left - 8,
            y0 + (band - 8) / 2 + SMALL * 0.35,
            labels[i],
            anchor="end",
            size=SMALL,
            tip=str(cat) if labels[i] != str(cat) else None,
        )
        for j, s in enumerate(spec.series):
            v = s.values[i] if i < len(s.values) else None
            if v is None:
                continue
            y = y0 + j * (bar_h + 2)
            w = max(0.0, sx(v) - left)
            note = s.annotations[i] if s.annotations else fmt(v)
            tip = tips(
                str(cat),
                f"Series: {s.label}" if s.label else None,
                *(
                    s.details[i]
                    if s.details
                    else [
                        f"{spec.value_name}: {note}",
                        f"Share of shown: {pct(v, totals[j])}",
                        f"Rank: {i + 1} of {n_cat}",
                    ]
                ),
            )
            if w >= 1:
                c.path(
                    bar_path(left, y, w, bar_h, end="right"),
                    f'class="dp-mark" fill="{s.color}"',
                    tip,
                )
            c.text(left + w + 5, y + bar_h / 2 + SMALL * 0.35, note, size=SMALL)
    height = top + len(spec.categories) * band + 6
    c.line(left, top, left, height - 6, AXIS)
    c.height = height
    return c.render()


def _render_vbar(spec: BarSpec) -> str:
    """Vertical bars (missing-values count); one series."""
    s = spec.series[0]
    n = len(spec.categories)
    labels = [truncate(str(x), 20) for x in spec.categories]
    size = SMALL if n <= 20 else 9
    label_h = max((text_width(lbl, size) for lbl in labels), default=0) * 0.72 + 12
    W = spec.width
    yticks = [0, 0.25, 0.5, 0.75, 1.0] if spec.value_format == "percent" else None
    fmt = _value_fmt(spec)
    vmax: float = (
        float(spec.value_max)
        if spec.value_max
        else float(max((v for v in s.values if v is not None), default=1))
    )
    if yticks is None:
        vmax = nice_max(vmax)
        yticks = nice_ticks(0, vmax, 4)
    left = 16 + y_label_width([fmt(t) for t in yticks], size)
    top = 34 if n > 12 else 22
    plot_h = spec.height - top - label_h
    H = spec.height
    c = Canvas(W, H, "bar", spec.title)
    plot_w = W - left - 30
    band = plot_w / max(n, 1)
    bw = min(24.0, band * 0.72)
    sy = Linear(0, vmax, top + plot_h, top)
    y_axis(c, sy, yticks, left, left + plot_w, fmt, size)
    if spec.value_label:
        c.text(12, top + plot_h / 2, spec.value_label, "middle", size=size, rotate=-90)
    for i, cat in enumerate(spec.categories):
        v = s.values[i]
        cx = left + band * (i + 0.5)
        note = s.annotations[i] if s.annotations else fmt(v)
        y = sy(v)
        c.path(
            bar_path(cx - bw / 2, y, bw, sy(0) - y),
            f'class="dp-mark" fill="{s.color}"',
        )
        c.hit(
            left + band * i,
            top,
            band,
            plot_h,
            tips(
                str(cat),
                *(s.details[i] if s.details else [f"{spec.value_name}: {note}"]),
            ),
        )
        if n <= 12:
            c.text(cx, y - 5, note, "middle", size=size)
        else:
            c.text(cx, y - 5, note, size=size - 1, rotate=-45)
        c.text(
            cx + 3,
            sy(0) + 10,
            labels[i],
            anchor="end",
            size=size,
            rotate=-45,
            tip=str(cat) if labels[i] != str(cat) else None,
        )
    c.line(left, sy(0), left + plot_w, sy(0), AXIS)
    return c.render()


def render_stacked_bar(spec: StackedBarSpec) -> str:
    W = spec.width
    multi = len(spec.rows) > 1
    row_labels = [truncate(r.label, 18) for r in spec.rows]
    left = (min(140, y_label_width(row_labels)) + 4) if multi else 0
    bar_h = 34
    c = Canvas(W, 0, "stacked-bar", spec.title)
    y = 4.0
    for r, rlabel in zip(spec.rows, row_labels):
        total = float(sum(v for v in r.values if v)) or 1.0
        if multi:
            c.text(left - 8, y + bar_h / 2 + FONT * 0.35, rlabel, "end")
        x = float(left)
        avail = W - left
        order = sorted(range(len(r.values)), key=lambda k: -(r.values[k] or 0))
        rank = {k: pos + 1 for pos, k in enumerate(order)}
        n_present = sum(1 for v in r.values if v)
        for k, (cat, v, color) in enumerate(
            zip(spec.categories, r.values, spec.colors)
        ):
            if not v:
                continue
            w = v / total * avail
            pct_v = v / total * 100
            tip = tips(
                str(cat),
                f"Series: {r.label}" if multi and r.label else None,
                f"Count: {fmt_count(v)}",
                f"Share: {pct_v:.1f}%",
                f"Rank: {rank[k]} of {n_present}",
                f"Total: {fmt_count(total)}",
            )
            c.rect(x, y, max(0.0, w - 2), bar_h, color, tip, 'rx="3"')
            label = f"{pct_v:.1f}% ({fmt_count(v)})"
            if pct_v > 8 and text_width(label, SMALL) + 8 < w:
                c.text(
                    x + (w - 2) / 2,
                    y + bar_h / 2 + SMALL * 0.35,
                    label,
                    "middle",
                    size=SMALL,
                    attrs=f'fill="{pal.text_on(color)}"',
                    weight="600",
                )
            x += w
        y += bar_h + 8
    if spec.legend:
        y += legend(
            c,
            [(str(k), col, "rect") for k, col in zip(spec.categories, spec.colors)],
            left,
            y,
            W - left,
        )
    c.height = y + 4
    return c.render()


def render_pie(spec: PieSpec) -> str:
    R, r_in = 96.0, 54.0
    cx, cy = R + 8, R + 8
    W = spec.width if spec.legend else 2 * R + 16
    c = Canvas(W, 2 * R + 16, "pie", spec.title)
    total = float(sum(v for v in spec.values if v)) or 1.0
    a0 = -math.pi / 2
    order = sorted(range(len(spec.values)), key=lambda k: -(spec.values[k] or 0))
    rank = {k: pos + 1 for pos, k in enumerate(order)}
    n_present = sum(1 for v in spec.values if v)
    for k, (cat, v, color) in enumerate(zip(spec.categories, spec.values, spec.colors)):
        if not v:
            continue
        frac = v / total
        a1 = a0 + frac * 2 * math.pi
        if frac >= 0.9999:
            a1 = a0 + 2 * math.pi - 1e-4
        large = 1 if a1 - a0 > math.pi else 0

        def pt(a: float, rad: float) -> str:
            return f"{_f(cx + rad * math.cos(a))} {_f(cy + rad * math.sin(a))}"

        d = (
            f"M{pt(a0, R)}A{_f(R)} {_f(R)} 0 {large} 1 {pt(a1, R)}"
            f"L{pt(a1, r_in)}A{_f(r_in)} {_f(r_in)} 0 {large} 0 {pt(a0, r_in)}Z"
        )
        c.path(
            d,
            f'class="dp-mark" fill="{color}" stroke="{SURFACE}" stroke-width="2"',
            tips(
                str(cat),
                f"Count: {fmt_count(v)}",
                f"Share: {frac * 100:.1f}%",
                f"Rank: {rank[k]} of {n_present}",
                f"Total: {fmt_count(total)}",
            ),
        )
        if frac > 0.08:
            mid = (a0 + a1) / 2
            rad = (R + r_in) / 2
            c.text(
                cx + rad * math.cos(mid),
                cy + rad * math.sin(mid) + SMALL * 0.35,
                f"{frac * 100:.0f}%",
                "middle",
                size=SMALL,
                attrs=f'fill="{pal.text_on(color)}"',
                weight="600",
            )
        a0 = a1
    c.text(cx, cy + 5, fmt_count(total), "middle", size=14, attrs='fill="currentColor"')
    if spec.legend:
        legend(
            c,
            [(str(k), col, "rect") for k, col in zip(spec.categories, spec.colors)],
            2 * R + 32,
            cy - 60,
            W - 2 * R - 40,
        )
    return c.render()


# ----------------------------------------------------------------------------
# Heatmaps
# ----------------------------------------------------------------------------


def _coef_label(t: float) -> str:
    if 0.95 <= t < 1:
        return "<1"
    if -1 < t <= -0.95:
        return ">-1"
    if t == 1:
        return "1"
    if t == -1:
        return "-1"
    if -0.05 < t < 0.05:
        return ""
    return f"{t:.1f}"


def _strength(v: float) -> str:
    a = abs(v)
    if a < 0.1:
        return "negligible"
    if a < 0.3:
        return "weak"
    if a < 0.5:
        return "moderate"
    if a < 0.7:
        return "strong"
    return "very strong"


def _interpret(spec: HeatmapSpec, i: int, j: int, v: float) -> List[str]:
    if spec.interpret == "none":
        return []
    if spec.row_labels[i] == spec.col_labels[j]:
        return ["Same variable"]
    if spec.interpret == "nullity":
        if abs(v) < 0.1:
            return ["Missingness is unrelated"]
        meaning = (
            "tend to be missing together"
            if v > 0
            else "one tends to be missing when the other is not"
        )
        return [f"Strength: {_strength(v)}", f"Meaning: {meaning}"]
    signed = spec.vmin < 0 and abs(v) >= 0.1
    direction = (" positive" if v > 0 else " negative") if signed else ""
    return [f"Strength: {_strength(v)}{direction}"]


def render_heatmap(spec: HeatmapSpec) -> str:
    nr, nc = len(spec.row_labels), len(spec.col_labels)
    cell = max(5.0, min(spec.cell_max, 460.0 / max(nr, nc, 1)))
    size = max(6.0, min(SMALL, cell * 0.8))
    rl = [truncate(str(x)) for x in spec.row_labels]
    cl = [truncate(str(x)) for x in spec.col_labels]
    left = max((text_width(s, size) for s in rl), default=0) + 10
    label_h = max((text_width(s, size) for s in cl), default=0) * 0.72 + 12
    top = 8
    grid_w, grid_h = nc * cell, nr * cell
    cb_x = left + grid_w + 18
    W = cb_x + 12 + 40
    has_bad = False
    H = top + grid_h + label_h
    c = Canvas(W, max(H, 200 if spec.colorbar else 0), "heatmap", spec.title)
    with_tips = nr * nc <= 4000
    span = (spec.vmax - spec.vmin) or 1.0
    for i in range(nr):
        for j in range(nc):
            if spec.mask is not None and spec.mask[i][j]:
                continue
            raw = spec.values[i][j]
            bad = raw is None or not np.isfinite(raw)
            v = 0.0 if raw is None else float(raw)
            if bad:
                has_bad = True
                fill = spec.bad_color
            else:
                fill = pal.ramp(spec.stops, (v - spec.vmin) / span)
            x, y = left + j * cell, top + i * cell
            tip = None
            if with_tips:
                tip = tips(
                    f"{spec.row_labels[i]} × {spec.col_labels[j]}",
                    f"{spec.value_name}: "
                    + ("n/a (could not be computed)" if bad else f"{v:.3f}"),
                    *([] if bad else _interpret(spec, i, j, v)),
                )
            gap = 1.0 if cell >= 8 else 0.0
            c.rect(x, y, cell - gap, cell - gap, fill, tip)
            if spec.annotate and not bad and cell >= 16:
                label = (
                    _coef_label(v)
                    if spec.annotation_style == "coefficient"
                    else fmt_num(v)
                )
                if label:
                    c.text(
                        x + (cell - gap) / 2,
                        y + (cell - gap) / 2 + size * 0.35,
                        label,
                        "middle",
                        size=size,
                        attrs=f'fill="{pal.text_on(fill)}"',
                    )
    for i, s in enumerate(rl):
        c.text(
            left - 6,
            top + (i + 0.5) * cell + size * 0.35,
            s,
            "end",
            size=size,
            tip=str(spec.row_labels[i]) if s != str(spec.row_labels[i]) else None,
        )
    for j, s in enumerate(cl):
        x = left + (j + 0.5) * cell
        c.text(
            x + size * 0.3,
            top + grid_h + 8,
            s,
            "end",
            size=size,
            rotate=-45,
            tip=str(spec.col_labels[j]) if s != str(spec.col_labels[j]) else None,
        )
    if spec.colorbar:
        steps = 24
        cb_h = min(max(grid_h, 160.0), 260.0)
        for k in range(steps):
            t0 = 1 - (k + 1) / steps
            c.rect(
                cb_x,
                top + k * cb_h / steps,
                12,
                cb_h / steps + 0.5,
                pal.ramp(spec.stops, t0 + 0.5 / steps),
                mark=False,
            )
        for t in nice_ticks(spec.vmin, spec.vmax, 4):
            y = top + (1 - (t - spec.vmin) / span) * cb_h
            c.line(cb_x + 12, y, cb_x + 16, y, AXIS)
            c.text(cb_x + 19, y + SMALL * 0.35, fmt_num(t, 0.5), size=SMALL)
        if has_bad:
            y = top + cb_h + 14
            c.rect(cb_x, y, 12, 12, spec.bad_color, mark=False)
            c.text(cb_x - 2, y + 26, spec.bad_label, size=9)
            c.height = max(c.height, y + 32)
            c.width = max(c.width, cb_x + text_width(spec.bad_label, 9) + 4)
    return c.render()


def render_missing_matrix(spec: MissingMatrixSpec) -> str:
    n = len(spec.columns)
    labels = [truncate(str(x), 20) for x in spec.columns]
    col_w = max(4.0, min(28.0, 600.0 / max(n, 1)))
    size = max(7.0, min(SMALL, col_w * 0.75))
    show_labels = spec.labels or n <= 50
    label_h = (
        max((text_width(s, size) for s in labels), default=0) * 0.72 + 10
        if show_labels
        else 4
    )
    left = y_label_width([fmt_count(spec.nrows)], SMALL) + 4
    plot_h = 240.0
    top = label_h
    W = left + n * col_w + 8
    c = Canvas(max(W, 120), top + plot_h + 8, "missing-matrix", spec.title)
    sy = Linear(0, max(spec.nrows, 1), top, top + plot_h)
    background = pal.NEUTRAL
    for j, (col, runs) in enumerate(zip(spec.columns, spec.bands)):
        x = left + j * col_w
        w = col_w - (2 if col_w >= 6 else 0.5)
        present = sum((e - s) * f for s, e, f in runs)
        missing = spec.nrows - present
        overall = (
            f"Missing in column: {fmt_count(round(missing))} "
            f"({pct(missing, max(spec.nrows, 1))})"
        )
        c.rect(x, top, w, plot_h, background, tips(str(col), overall), mark=False)
        for s, e, frac in runs:
            if frac <= 0:
                continue
            fill = pal.mix(background, spec.color, frac)
            approx = "≈" if 0 < frac < 1 else ""
            band_tip = tips(
                str(col),
                f"Rows: {s + 1:,}–{e:,}",
                f"Present in these rows: {approx}{frac * 100:.0f}%",
                overall,
            )
            c.rect(x, sy(s), w, sy(e) - sy(s) + 0.3, fill, band_tip, mark=False)
        if show_labels:
            c.text(
                x + w / 2 - size * 0.3,
                top - 6,
                labels[j],
                size=size,
                rotate=-45,
                tip=str(col) if labels[j] != str(col) else None,
            )
    c.text(left - 6, top + SMALL, "1", "end", size=SMALL)
    c.text(left - 6, top + plot_h, fmt_count(spec.nrows), "end", size=SMALL)
    return c.render()


# ----------------------------------------------------------------------------
# Scatter / hexbin
# ----------------------------------------------------------------------------


def _extent(v: np.ndarray) -> Tuple[float, float]:
    v = v[np.isfinite(v)]
    if v.size == 0:
        return 0.0, 1.0
    lo, hi = float(v.min()), float(v.max())
    pad = (hi - lo) * 0.04 or (abs(lo) * 0.05 or 0.5)
    return lo - pad, hi + pad


def render_scatter(spec: ScatterSpec) -> str:
    W, H = spec.width, spec.height
    x = np.asarray(spec.x, dtype=float)
    y = np.asarray(spec.y, dtype=float)
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    xlo, xhi = _extent(x)
    ylo, yhi = _extent(y)
    yt = nice_ticks(ylo, yhi, 5)
    ystep = yt[1] - yt[0] if len(yt) > 1 else 0
    left = 22 + y_label_width([fmt_num(t, ystep) for t in yt])
    top, bottom = 10, 40
    plot_w, plot_h = W - left - 12, H - top - bottom
    sx = Linear(xlo, xhi, left, left + plot_w)
    sy = Linear(ylo, yhi, top + plot_h, top)
    c = Canvas(W, H, "hexbin" if spec.hexbin else "scatter", spec.title)
    y_axis(c, sy, yt, left, left + plot_w, lambda v: fmt_num(v, ystep))
    for v, label in x_axis_ticks(xlo, xhi, plot_w, False, SMALL):
        px = sx(v)
        c.line(px, top, px, top + plot_h)
        c.text(px, top + plot_h + SMALL + 6, label, "middle", size=SMALL)
    c.text(left + plot_w / 2, H - 6, spec.x_label, "middle")
    c.text(12, top + plot_h / 2, spec.y_label, "middle", rotate=-90)

    if spec.hexbin and x.size:
        px, py = sx.many(x), sy.many(y)
        r = plot_w / (spec.gridsize * math.sqrt(3))
        q = (math.sqrt(3) / 3 * (px - left) - (py - top) / 3) / r
        rr = (2 / 3 * (py - top)) / r
        # cube rounding
        cx_, cz_ = q, rr
        cy_ = -cx_ - cz_
        rx, ry, rz = np.round(cx_), np.round(cy_), np.round(cz_)
        dx, dy, dz = np.abs(rx - cx_), np.abs(ry - cy_), np.abs(rz - cz_)
        fix_x = (dx > dy) & (dx > dz)
        fix_y = ~fix_x & (dy > dz)
        rx = np.where(fix_x, -ry - rz, rx)
        ry = np.where(fix_y, -rx - rz, ry)
        rz = np.where(~fix_x & ~fix_y, -rx - ry, rz)
        keys, counts = np.unique(np.stack([rx, rz], axis=1), axis=0, return_counts=True)
        stops = pal.sequential_stops(spec.color)
        cmax = counts.max()
        corners = [
            (
                r * math.cos(math.radians(60 * k - 30)),
                r * math.sin(math.radians(60 * k - 30)),
            )
            for k in range(6)
        ]
        hexd = (
            "".join(
                ("m" if k == 0 else "l")
                + f"{_f(corners[k][0] - (corners[k - 1][0] if k else 0))} "
                f"{_f(corners[k][1] - (corners[k - 1][1] if k else 0))}"
                for k in range(6)
            )
            + "z"
        )
        c.add(
            f'<svg x="{_f(left)}" y="{_f(top)}" width="{_f(plot_w)}" '
            f'height="{_f(plot_h)}" overflow="hidden">'
            f'<g transform="translate({_f(-left)} {_f(-top)})">'
        )
        total = float(counts.sum())
        half_w = r * math.sqrt(3) / 2
        for (qq, zz), n in zip(keys, counts):
            hx = left + r * math.sqrt(3) * (qq + zz / 2)
            hy = top + r * 1.5 * zz
            fill = pal.ramp(stops, 0.15 + 0.85 * (n / cmax) ** 0.5)
            x_a, x_b = sx.invert(hx - half_w), sx.invert(hx + half_w)
            y_a, y_b = sorted((sy.invert(hy + r), sy.invert(hy - r)))
            c.path(
                f"M{_f(hx)} {_f(hy)}{hexd}",
                f'class="dp-mark" fill="{fill}"',
                tips(
                    f"{fmt_count(n)} point{'' if n == 1 else 's'}",
                    f"Share: {pct(n, total)}",
                    f"{spec.x_label or 'x'}: {fmt_num(x_a)} – {fmt_num(x_b)}",
                    f"{spec.y_label or 'y'}: {fmt_num(y_a)} – {fmt_num(y_b)}",
                ),
            )
        c.add("</g></svg>")
        c.text(
            left + plot_w - 2,
            top + 12,
            f"hexbin · {fmt_count(spec.n_points or x.size)} points",
            "end",
            size=9,
        )
    else:
        c.add(
            f'<g fill="{spec.color}" fill-opacity="0.65" stroke="{SURFACE}" '
            'stroke-width="0.75">'
        )
        for xv, yv in zip(x, y):
            c.add(f'<circle cx="{_f(sx(xv))}" cy="{_f(sy(yv))}" r="3.5"/>')
        c.add("</g>")
        # Nearest-point hover (report script); far smaller than a <title> per point.
        c.data["scatter"] = json.dumps(
            {
                "p": [
                    round(left, 1),
                    round(top, 1),
                    round(plot_w, 1),
                    round(plot_h, 1),
                ],
                "xd": [xlo, xhi],
                "yd": [ylo, yhi],
                "xl": spec.x_label or "x",
                "yl": spec.y_label or "y",
                "c": spec.color,
                "x": [_num(v, 5) for v in x],
                "y": [_num(v, 5) for v in y],
            },
            separators=(",", ":"),
        )
    c.line(left, top + plot_h, left + plot_w, top + plot_h, AXIS)
    return c.render()


# ----------------------------------------------------------------------------
# Lines
# ----------------------------------------------------------------------------


def _polyline(xs: np.ndarray, ys: np.ndarray) -> str:
    parts, pen = [], False
    for xv, yv in zip(xs, ys):
        if not (np.isfinite(xv) and np.isfinite(yv)):
            pen = False
            continue
        parts.append(f"{'L' if pen else 'M'}{_f(xv)} {_f(yv)}")
        pen = True
    return "".join(parts)


def _num(v: Optional[float], digits: int = 6) -> Optional[float]:
    if v is None or not np.isfinite(v):
        return None
    out = float(f"{v:.{digits}g}")
    return int(out) if out.is_integer() and abs(out) < 1e15 else out


def _duration(seconds: float) -> str:
    d = seconds / 86400
    if d >= 2:
        return f"{d:,.1f} days"
    h = seconds / 3600
    if h >= 2:
        return f"{h:,.1f} hours"
    return f"{seconds / 60:,.1f} minutes"


def render_line(spec: LineSpec) -> str:
    W, H = spec.width, spec.height
    size = 9 if spec.mini else SMALL
    xs_all = np.concatenate([np.asarray(s.x, float) for s in spec.series])
    ys_all = np.concatenate([np.asarray(s.y, float) for s in spec.series])
    xlo, xhi = float(np.nanmin(xs_all)), float(np.nanmax(xs_all))
    ylo, yhi = _extent(ys_all)
    yt = nice_ticks(ylo, yhi, 3 if spec.mini else 5)
    ystep = yt[1] - yt[0] if len(yt) > 1 else 0
    left = 8 + y_label_width([fmt_num(t, ystep) for t in yt], size)
    if spec.y_label:
        left += 14
    c = Canvas(W, H, "line" if not spec.mini else "mini-line", spec.title)
    top = 8.0
    if spec.legend:
        items = [(s.label, s.color, "dash" if s.dash else "line") for s in spec.series]
        top += legend(c, items, left, 2, W - left - 8, size)
    bottom = size + 14
    plot_w, plot_h = W - left - 10, H - top - bottom
    sx = Linear(xlo, xhi, left, left + plot_w)
    sy = Linear(ylo, yhi, top + plot_h, top)
    y_axis(c, sy, yt, left, left + plot_w, lambda v: fmt_num(v, ystep), size)
    if spec.y_label:
        c.text(12, top + plot_h / 2, spec.y_label, "middle", size=size, rotate=-90)
    span = xhi - xlo
    dfmt = date_tooltip_format(span)
    for x0, x1, color in spec.spans:
        a, b = sx(max(x0, xlo)), sx(min(x1, xhi))
        if spec.x_is_date:
            tip = tips(
                "Gap",
                f"From: {fmt_date(x0, dfmt)}",
                f"To: {fmt_date(x1, dfmt)}",
                f"Duration: {_duration(x1 - x0)}",
            )
        else:
            tip = tips(
                "Gap",
                f"From: {fmt_num(x0)}",
                f"To: {fmt_num(x1)}",
                f"Length: {fmt_num(x1 - x0)}",
            )
        c.rect(a, top, max(b - a, 1.5), plot_h, color, tip, 'fill-opacity="0.2"')

    # Data for the interactive crosshair (report script); already downsampled.
    # A shared x axis is stored once; dates are whole seconds.
    def xs_of(s: Any) -> List[Any]:
        if spec.x_is_date:
            return [int(v) if np.isfinite(v) else None for v in s.x]
        return [_num(v) for v in s.x]

    x_lists = [xs_of(s) for s in spec.series]
    shared_x = all(xl == x_lists[0] for xl in x_lists)
    payload: Dict[str, Any] = {
        "p": [round(left, 1), round(top, 1), round(plot_w, 1), round(plot_h, 1)],
        "xd": [xlo, xhi],
        "yd": [ylo, yhi],
        "d": spec.x_is_date,
        "s": [
            {
                "l": s.label,
                "c": s.color,
                **({} if shared_x else {"x": x_lists[k]}),
                "y": [_num(v, 5) for v in s.y],
            }
            for k, s in enumerate(spec.series)
        ],
        "g": [[a, b] for a, b, _ in spec.spans],
    }
    if shared_x:
        payload["x"] = x_lists[0]
    c.data["line"] = json.dumps(payload, separators=(",", ":"))
    for s in spec.series:
        d = _polyline(sx.many(np.asarray(s.x, float)), sy.many(np.asarray(s.y, float)))
        dash = ' stroke-dasharray="5 3"' if s.dash else ""
        opacity = ' stroke-opacity="0.85"' if len(spec.series) > 1 else ""
        c.path(
            d,
            f'fill="none" stroke="{s.color}" stroke-width="{1.5 if spec.mini else 2}" '
            f'stroke-linejoin="round" stroke-linecap="round"{dash}{opacity}',
            s.label or None,
        )
    c.line(left, top + plot_h, left + plot_w, top + plot_h, AXIS)
    for v, label in x_axis_ticks(xlo, xhi, plot_w, spec.x_is_date, size):
        px = sx(v)
        if px < left - 1 or px > left + plot_w + 1:
            continue
        c.line(px, top + plot_h, px, top + plot_h + 4, AXIS)
        anchor = "middle"
        if px - text_width(label, size) / 2 < 0:
            anchor = "start"
        elif px + text_width(label, size) / 2 > W:
            anchor = "end"
        c.text(px, top + plot_h + size + 6, label, anchor, size=size)
    return c.render()


def render_stem_grid(spec: StemGridSpec) -> str:
    W = spec.width
    n_rows = len(spec.rows)
    n_cols = max((len(r) for r in spec.rows), default=1)
    row_label_w = (
        min(120, y_label_width(spec.row_labels)) if len(spec.row_labels) > 1 else 0
    )
    panel_w = (W - row_label_w) / n_cols
    panel_h = 200.0
    c = Canvas(W, n_rows * (panel_h + 10), "stem-grid", spec.title)
    for i, row in enumerate(spec.rows):
        oy = i * (panel_h + 10)
        if row_label_w:
            c.text(
                row_label_w - 8,
                oy + panel_h / 2,
                truncate(spec.row_labels[i], 16),
                "end",
            )
        for j, p in enumerate(row):
            ox = row_label_w + j * panel_w
            left, top = ox + 40, oy + (22 if i == 0 else 8)
            plot_w, plot_h = panel_w - 56, panel_h - (top - oy) - 24
            nlags = max(len(p.values) - 1, 1)
            sx = Linear(-0.5, nlags + 0.5, left, left + plot_w)
            sy = Linear(-1.05, 1.05, top + plot_h, top)
            if i == 0:
                c.text(
                    left + plot_w / 2,
                    oy + 14,
                    p.title,
                    "middle",
                    weight="600",
                    attrs='fill="currentColor"',
                )
            y_axis(
                c,
                sy,
                [-1, -0.5, 0, 0.5, 1],
                left,
                left + plot_w,
                lambda v: fmt_num(v, 0.5),
            )
            lags = np.arange(len(p.values))
            band = (
                _polyline(sx.many(lags), sy.many(np.asarray(p.conf_high, float)))
                + _polyline(
                    sx.many(lags[::-1]), sy.many(np.asarray(p.conf_low, float)[::-1])
                ).replace("M", "L", 1)
                + "Z"
            )
            c.path(band, f'fill="{p.color}" fill-opacity="0.18"')
            c.line(left, sy(0), left + plot_w, sy(0), AXIS)
            r = max(1.5, min(4.0, plot_w / (nlags + 1) / 2.6))
            for k, v in enumerate(p.values):
                if v is None or not np.isfinite(v):
                    continue
                px, py = sx(k), sy(v)
                c.line(px, sy(0), px, py, f'stroke="{p.color}" stroke-width="1.5"')
                c.add(
                    f'<circle class="dp-mark" cx="{_f(px)}" cy="{_f(py)}" r="{_f(r)}" '
                    f'fill="{p.color}" stroke="{SURFACE}" stroke-width="1"/>'
                )
                lo_b, hi_b = p.conf_low[k], p.conf_high[k]
                significant = k > 0 and (v < lo_b or v > hi_b)
                c.hit(
                    sx(k - 0.5),
                    top,
                    sx(k + 0.5) - sx(k - 0.5),
                    plot_h,
                    tips(
                        f"Lag {k}",
                        (
                            f"Series: {spec.row_labels[i]}"
                            if len(spec.row_labels) > 1
                            else None
                        ),
                        f"{p.title}: {v:.3f}",
                        None if k == 0 else f"95% band: {lo_b:+.3f} to {hi_b:+.3f}",
                        (
                            None
                            if k == 0
                            else f"Significant: {'yes' if significant else 'no'}"
                        ),
                    ),
                )
            for v, label in x_axis_ticks(0, nlags, plot_w, False, SMALL):
                if float(v).is_integer():
                    c.text(sx(v), top + plot_h + SMALL + 6, label, "middle", size=SMALL)
    return c.render()


_RENDERERS: Dict[type, Callable] = {
    HistogramSpec: render_histogram,
    BarSpec: render_bar,
    StackedBarSpec: render_stacked_bar,
    PieSpec: render_pie,
    HeatmapSpec: render_heatmap,
    MissingMatrixSpec: render_missing_matrix,
    ScatterSpec: render_scatter,
    LineSpec: render_line,
    StemGridSpec: render_stem_grid,
}


def render(spec: ChartSpec) -> str:
    """Render a chart spec to inline SVG markup."""
    try:
        fn = _RENDERERS[type(spec)]
    except KeyError as exc:
        raise TypeError(f"No SVG renderer for {type(spec).__name__}") from exc
    return fn(spec)
