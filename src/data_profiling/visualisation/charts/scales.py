"""Scales, tick generation and number/date formatting for SVG charts."""

import math
from datetime import datetime, timezone
from typing import List, Sequence, Tuple

import numpy as np
import pandas as pd


class Linear:
    """Map a numeric domain onto a pixel range."""

    def __init__(self, d0: float, d1: float, r0: float, r1: float):
        if not math.isfinite(d0) or not math.isfinite(d1):
            d0, d1 = 0.0, 1.0
        if d0 == d1:
            pad = abs(d0) * 0.05 or 0.5
            d0, d1 = d0 - pad, d1 + pad
        self.d0, self.d1, self.r0, self.r1 = d0, d1, r0, r1

    def __call__(self, v: float) -> float:
        return self.r0 + (v - self.d0) / (self.d1 - self.d0) * (self.r1 - self.r0)

    def invert(self, px: float) -> float:
        return self.d0 + (px - self.r0) / (self.r1 - self.r0) * (self.d1 - self.d0)

    def many(self, values: np.ndarray) -> np.ndarray:
        return self.r0 + (values - self.d0) / (self.d1 - self.d0) * (self.r1 - self.r0)


def _nice_step(span: float, count: int) -> float:
    raw = span / max(count, 1)
    if raw <= 0 or not math.isfinite(raw):
        return 1.0
    mag = 10 ** math.floor(math.log10(raw))
    for m in (1, 2, 2.5, 5, 10):
        if raw <= m * mag:
            return m * mag
    return 10 * mag


def nice_ticks(lo: float, hi: float, count: int = 5) -> List[float]:
    """Round tick values inside [lo, hi]."""
    if not (math.isfinite(lo) and math.isfinite(hi)):
        return []
    if lo == hi:
        return [lo]
    step = _nice_step(hi - lo, count)
    # At large magnitudes the step can be below float resolution (e.g. values
    # around 1e17 spanning 100): fall back to the end points.
    if step <= np.spacing(max(abs(lo), abs(hi))) * 4:
        return [lo, hi]
    start = math.ceil(lo / step - 1e-9) * step
    n = int(math.floor((hi - start) / step + 1e-9)) + 1
    ticks = []
    for k in range(max(0, min(n, 50))):  # bounded: never loops on float error
        v = start + k * step
        ticks.append(0.0 if abs(v) < step * 1e-9 else v)
    return ticks


def nice_max(hi: float, count: int = 4) -> float:
    """Extend an upper bound to the next round tick (for zero-based axes)."""
    if hi <= 0 or not math.isfinite(hi):
        return 1.0
    step = _nice_step(hi, count)
    return math.ceil(hi / step - 1e-9) * step


def fmt_num(v: float, step: float = 0.0) -> str:
    """Compact numeric label: 1,234 / 12.5k / 3.4M / 0.25 / 1.2e-05."""
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return "–"
    a = abs(v)
    if a >= 1e4:
        for div, suf in ((1e12, "T"), (1e9, "B"), (1e6, "M"), (1e3, "k")):
            if a >= div:
                s = f"{v / div:.3g}"
                return f"{s}{suf}"
    if a != 0 and a < 1e-3:
        return f"{v:.1e}"
    if step:
        decimals = max(0, -int(math.floor(math.log10(step))) if step < 1 else 0)
        if step < 1 and round(step * 10**decimals, 6) % 1:
            decimals += 1
        return f"{v:,.{decimals}f}"
    if float(v).is_integer():
        return f"{int(v):,}"
    return f"{v:.4g}"


def fmt_count(v: float) -> str:
    if float(v).is_integer():
        return f"{int(v):,}"
    return f"{v:.3g}"


# ----------------------------------------------------------------------------
# Dates. Values are handled as POSIX seconds (UTC), which is how the summary
# stores date histograms.
# ----------------------------------------------------------------------------

_DATE_STEPS: Sequence[Tuple[str, float, str]] = (
    ("10YS", 10 * 365.25 * 86400, "%Y"),
    ("5YS", 5 * 365.25 * 86400, "%Y"),
    ("2YS", 2 * 365.25 * 86400, "%Y"),
    ("YS", 365.25 * 86400, "%Y"),
    ("QS", 91.3 * 86400, "%Y-%m"),
    ("MS", 30.4 * 86400, "%Y-%m"),
    ("7D", 7 * 86400, "%Y-%m-%d"),
    ("D", 86400, "%m-%d"),
    ("6h", 6 * 3600, "%m-%d %H:%M"),
    ("h", 3600, "%H:%M"),
    ("15min", 900, "%H:%M"),
    ("min", 60, "%H:%M"),
    ("10s", 10, "%H:%M:%S"),
    ("s", 1, "%H:%M:%S"),
)


def to_seconds(values: Sequence) -> np.ndarray:
    """Convert datetimes / timestamps / numbers to float POSIX seconds."""
    arr = np.asarray(values)
    if np.issubdtype(arr.dtype, np.datetime64):
        out = arr.astype("datetime64[ns]").astype("int64").astype(float) / 1e9
        out[np.isnat(arr)] = np.nan
        return out
    if arr.dtype == object:
        return np.array(
            [
                pd.Timestamp(v).value / 1e9 if v is not None and v == v else np.nan
                for v in arr
            ]
        )
    return arr.astype(float)


def fmt_date(sec: float, fmt: str = "%Y-%m-%d") -> str:
    try:
        return datetime.fromtimestamp(sec, tz=timezone.utc).strftime(fmt)
    except (OverflowError, OSError, ValueError):
        return ""


def date_ticks(lo: float, hi: float, count: int = 6) -> List[Tuple[float, str]]:
    """Calendar-aligned ticks between lo and hi (POSIX seconds)."""
    if not (math.isfinite(lo) and math.isfinite(hi)) or hi <= lo:
        return [(lo, fmt_date(lo))] if math.isfinite(lo) else []
    span = hi - lo
    chosen = _DATE_STEPS[-1]
    for step in reversed(_DATE_STEPS):
        if span / step[1] <= count:
            chosen = step
            break
    freq, _, fmt = chosen
    try:
        rng = pd.date_range(
            pd.Timestamp(lo, unit="s"), pd.Timestamp(hi, unit="s"), freq=freq
        )
    except Exception:
        rng = pd.DatetimeIndex([])
    ticks = [(t.value / 1e9, t.strftime(fmt)) for t in rng]
    if len(ticks) < 2:
        ticks = [(v, fmt_date(v, "%Y-%m-%d")) for v in (lo, hi)]
    return ticks


def date_tooltip_format(span: float) -> str:
    return "%Y-%m-%d %H:%M" if span < 3 * 86400 else "%Y-%m-%d"
