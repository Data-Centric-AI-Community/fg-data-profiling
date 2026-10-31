"""Build chart specs from the inputs of the public plot functions.

Each builder takes exactly what the corresponding function in
``visualisation.plot`` / ``visualisation.missing`` receives, so those functions
can dispatch to the native renderer without any change at their call sites.
"""

from typing import Any, List, Optional, Sequence, Union

import numpy as np
import pandas as pd

from data_profiling.config import Settings
from data_profiling.visualisation.charts import palette as pal
from data_profiling.visualisation.charts.scales import to_seconds
from data_profiling.visualisation.charts.specs import (
    BarSeries,
    BarSpec,
    HeatmapSpec,
    HistogramSpec,
    HistSeries,
    LineSeries,
    LineSpec,
    MissingMatrixSpec,
    PieSpec,
    ScatterSpec,
    StackedBarSpec,
    StackedRow,
    StemGridSpec,
    StemPanel,
)

MAX_LINE_POINTS = 1200
MAX_MATRIX_BANDS = 200
OTHER: Any = object()  # sentinel: folded tail of categories
OTHER_LABEL = "Other"


def primary(config: Settings, idx: int = 0) -> str:
    colors = config.html.style.primary_colors
    return pal.to_hex(colors[idx % len(colors)])


def comparison_colors(config: Settings) -> List[str]:
    """Same semantics as ``plot.create_comparison_color_list``."""
    colors = [pal.to_hex(c) for c in config.html.style.primary_colors]
    labels = config.html.style._labels
    if len(colors) < len(labels):
        end = colors[1] if len(colors) >= 2 else "#000000"
        n = len(labels)
        colors = [pal.mix(colors[0], end, i / max(n - 1, 1)) for i in range(n)]
    return colors


# --- Histograms ---------------------------------------------------------------


def histogram(
    config: Settings,
    series: Any,
    bins: Any,
    date: bool = False,
    mini: bool = False,
) -> HistogramSpec:
    if isinstance(bins, list):
        labels = config.html.style._labels
        colors = comparison_colors(config)
        hs = [
            HistSeries(
                counts=np.asarray(c, float).tolist(),
                edges=np.asarray(e, float).tolist(),
                color=colors[i],
                label=str(labels[i]) if i < len(labels) else "",
            )
            for i, (c, e) in enumerate(zip(series, bins))
            if len(e) > 1
        ]
    else:
        hs = [
            HistSeries(
                counts=np.asarray(series, float).tolist(),
                edges=np.asarray(bins, float).tolist(),
                color=primary(config),
            )
        ]
    return HistogramSpec(
        title="Histogram",
        series=hs,
        date=date,
        mini=mini,
        x_labels=config.plot.histogram.x_axis_labels,
        y_label="Density" if config.plot.histogram.density else "Frequency",
        width=240 if mini else 640,
        height=170 if mini else 280,
    )


# --- Categorical frequencies ---------------------------------------------------


def _fold_categories(data: Sequence[pd.Series], max_slots: int) -> List[str]:
    """Union of categories in order of first appearance, folded to max_slots."""
    cats: List[str] = []
    for s in data:
        for k in s.index:
            if k not in cats:
                cats.append(k)
    if len(cats) > max_slots:
        cats = cats[: max_slots - 1] + [OTHER]
    return cats


def _row_values(s: pd.Series, cats: List[Any]) -> List[float]:
    out = []
    keep = [c for c in cats if c is not OTHER]
    for c in cats:
        if c is OTHER:
            out.append(float(s[[k for k in s.index if k not in keep]].sum()))
        else:
            out.append(float(s.get(c, 0)))
    return out


def cat_frequency(config: Settings, data: Union[pd.Series, List[pd.Series]]) -> Any:
    rows = data if isinstance(data, list) else [data]
    configured = config.plot.cat_freq.colors
    # Configured colours are honoured as-is (cycled, like the legacy renderer);
    # otherwise fold into a fixed-order 8-colour palette + "Other".
    max_slots = 10**6 if configured else len(pal.CATEGORICAL)
    cats = _fold_categories(rows, max_slots)
    colors = pal.categorical(len(cats), configured or ())
    if cats and cats[-1] is OTHER and not configured:
        colors[-1] = "#8a8984"
    hide_legend = config.vars.cat.redact
    names = [OTHER_LABEL if c is OTHER else str(c) for c in cats]
    if hide_legend:  # sensitive mode: category names must not reach the markup
        names = [f"Category {i + 1}" for i in range(len(cats))]
    plot_type = config.plot.cat_freq.type
    if plot_type == "bar":
        labels = config.html.style._labels
        return StackedBarSpec(
            title="Category frequency",
            categories=names,
            colors=colors,
            rows=[
                StackedRow(
                    label=str(labels[i]) if len(rows) > 1 and i < len(labels) else "",
                    values=_row_values(s, cats),
                )
                for i, s in enumerate(rows)
            ],
            legend=not hide_legend,
            redact=hide_legend,
        )
    if plot_type == "pie":
        return PieSpec(
            title="Category frequency",
            categories=names,
            values=_row_values(rows[0], cats),
            colors=colors,
            legend=not hide_legend,
            redact=hide_legend,
            width=460,
        )
    raise ValueError(
        f"'{plot_type}' is not a valid plot type! Expected values are ['bar', 'pie']"
    )


def word_frequency(
    config: Settings, word_counts: Union[pd.Series, List[pd.Series]], top: int = 20
) -> BarSpec:
    """Top-word frequency bars (replaces the rasterised word cloud)."""
    series = word_counts if isinstance(word_counts, list) else [word_counts]
    multi = len(series) > 1
    order: List[str] = []
    for s in series:
        for w in s.sort_values(ascending=False).index[:top]:
            if w not in order:
                order.append(w)
    order = order[:top]
    labels = config.html.style._labels
    colors = comparison_colors(config) if multi else [primary(config)]
    return BarSpec(
        title="Most frequent words",
        categories=[str(w) for w in order],
        series=[
            BarSeries(
                values=[float(s.get(w, 0)) for w in order],
                color=colors[i % len(colors)],
                label=str(labels[i]) if multi and i < len(labels) else "",
            )
            for i, s in enumerate(series)
        ],
        horizontal=True,
        value_name="Occurrences",
        width=520,
    )


# --- Correlations / heatmaps --------------------------------------------------


def _matrix(data: pd.DataFrame) -> np.ndarray:
    try:
        return np.asarray(data, dtype=float)
    except Exception:
        n = len(data)
        return np.full((n, n), np.nan, dtype=float)


def _nested(m: np.ndarray) -> List[List[Optional[float]]]:
    return [[None if not np.isfinite(v) else float(v) for v in row] for row in m]


def correlation_matrix(
    config: Settings, data: pd.DataFrame, vmin: int = -1
) -> HeatmapSpec:
    m = _matrix(data)
    labels = [str(x) for x in data.columns]
    return HeatmapSpec(
        title="Correlation matrix",
        row_labels=labels,
        col_labels=labels,
        values=_nested(m),
        vmin=float(vmin),
        vmax=1.0,
        stops=pal.colormap_stops(config.plot.correlation.cmap, half=vmin == 0),
        bad_color=pal.to_hex(config.plot.correlation.bad),
        annotate=len(labels) <= 12,
    )


def missing_heatmap(
    config: Settings, corr_mat: pd.DataFrame, mask: Any, columns: List[str]
) -> HeatmapSpec:
    m = _matrix(corr_mat)
    mask_arr = np.asarray(mask).astype(bool)
    labels = [str(x) for x in corr_mat.columns]
    # Drop the fully-masked first row / last column, like the classic layout.
    keep_r = [i for i in range(len(labels)) if not mask_arr[i].all()]
    keep_c = [j for j in range(len(labels)) if not mask_arr[:, j].all()]
    return HeatmapSpec(
        title="Missing values correlation",
        row_labels=[labels[i] for i in keep_r],
        col_labels=[labels[j] for j in keep_c],
        values=_nested(m[np.ix_(keep_r, keep_c)]),
        mask=mask_arr[np.ix_(keep_r, keep_c)].tolist(),
        value_name="Nullity correlation",
        interpret="nullity",
        vmin=-1.0,
        vmax=1.0,
        stops=pal.colormap_stops(config.plot.missing.cmap),
        bad_color=pal.NEUTRAL,
        annotate=config.plot.missing.force_labels,
        cell_max=40.0,
    )


# --- Missing values -----------------------------------------------------------


def missing_bar(
    config: Settings, notnull_counts: pd.Series, nrows: int, columns: List[str]
) -> BarSpec:
    counts = pd.Series(notnull_counts)
    frac = (counts / max(nrows, 1)).astype(float).tolist()
    n = len(counts)
    details = []
    for present in counts.tolist():
        missing = nrows - present
        share = present / nrows * 100 if nrows else 0.0
        details.append(
            [
                f"Present: {int(present):,} ({share:.1f}%)",
                f"Missing: {int(missing):,} ({100 - share:.1f}%)",
                f"Rows: {int(nrows):,}",
            ]
        )
    return BarSpec(
        title="Count of non-missing values",
        categories=[str(c) for c in counts.index],
        series=[
            BarSeries(
                values=frac,
                color=primary(config),
                annotations=[f"{int(v):,}" for v in counts.tolist()],
                details=details,
            )
        ],
        horizontal=n > 50,
        value_format="percent",
        value_max=1.0,
        value_label="Share of rows present",
        width=max(480, min(1100, 60 + n * 30)) if n <= 50 else 640,
        height=340,
    )


def missing_matrix(
    config: Settings, notnull: Any, columns: List[str], nrows: int
) -> MissingMatrixSpec:
    arr = np.asarray(notnull, dtype=bool)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    height = arr.shape[0]
    n_bands = max(1, min(MAX_MATRIX_BANDS, height))
    edges = np.linspace(0, height, n_bands + 1).round().astype(int)
    bands: List[List[Any]] = []
    for j in range(arr.shape[1]):
        col = arr[:, j]
        runs: List[Any] = []
        for b in range(n_bands):
            s, e = int(edges[b]), int(edges[b + 1])
            if e <= s:
                continue
            f = float(col[s:e].mean())
            # Quantise to 10% steps: readable and keeps the markup small.
            fq = (
                1.0
                if f == 1
                else (0.0 if f == 0 else round(min(max(f, 0.05), 0.95), 1))
            )
            if runs and runs[-1][2] == fq:
                runs[-1] = (runs[-1][0], e, fq)
            else:
                runs.append((s, e, fq))
        bands.append(runs)
    return MissingMatrixSpec(
        title="Missing values matrix",
        columns=[str(c) for c in columns],
        nrows=int(nrows if nrows else height),
        bands=bands,
        color=primary(config),
        labels=config.plot.missing.force_labels,
    )


# --- Scatter ------------------------------------------------------------------


def scatter(
    config: Settings,
    x: Any,
    y: Any,
    x_label: str,
    y_label: str,
    n_total: int,
    gridsize: int = 30,
) -> ScatterSpec:
    xs = np.asarray(x, dtype=float)
    ys = np.asarray(y, dtype=float)
    hexbin = n_total > config.plot.scatter_threshold
    return ScatterSpec(
        title=f"{y_label} vs {x_label}" if x_label or y_label else "Scatter",
        x=xs if hexbin else xs.tolist(),
        y=ys if hexbin else ys.tolist(),
        color=primary(config),
        x_label=str(x_label),
        y_label=str(y_label),
        hexbin=hexbin,
        gridsize=gridsize,
        n_points=int(xs.size),
        width=560,
        height=380,
    )


# --- Time series --------------------------------------------------------------


def _xy(series: pd.Series) -> tuple:
    """Index (as seconds when datetime) and values, downsampled for display."""
    s = series
    is_date = isinstance(s.index, pd.DatetimeIndex)
    x = to_seconds(s.index.values) if is_date else np.asarray(s.index, dtype=float)
    y = np.asarray(s.values, dtype=float)
    n = len(y)
    if n > MAX_LINE_POINTS:
        # min/max per bucket keeps peaks and troughs visible.
        buckets = MAX_LINE_POINTS // 2
        edges = np.linspace(0, n, buckets + 1).astype(int)
        idx: List[int] = []
        for a, b in zip(edges[:-1], edges[1:]):
            if b <= a:
                continue
            seg = y[a:b]
            if np.isfinite(seg).any():
                lo, hi = int(np.nanargmin(seg)) + a, int(np.nanargmax(seg)) + a
                idx.extend(sorted({lo, hi}))
            else:
                idx.append(a)
        x, y = x[idx], y[idx]
    return x, y, is_date


def timeseries(
    config: Settings, series: Union[pd.Series, List[pd.Series]], mini: bool
) -> LineSpec:
    if isinstance(series, list):
        labels = config.html.style._labels
        colors = comparison_colors(config)
        lines, is_date = [], False
        for i, s in enumerate(series):
            x, y, is_date = _xy(s)
            lines.append(LineSeries(x.tolist(), y.tolist(), colors[i], str(labels[i])))
    else:
        x, y, is_date = _xy(series)
        lines = [
            LineSeries(x.tolist(), y.tolist(), primary(config), str(series.name or ""))
        ]
    return LineSpec(
        title="Time series",
        series=lines,
        x_is_date=is_date,
        legend=len(lines) > 1,
        mini=mini,
        width=260 if mini else 640,
        height=170 if mini else 280,
    )


def _gap_spans(gaps: Sequence[Any], color: str) -> List[Any]:
    spans = []
    for g in gaps:
        g = np.asarray(g)
        if g.size < 2:
            continue
        if np.issubdtype(g.dtype, np.number):
            a, b = g[:2].astype(float)
        else:  # datetime64 or Timestamp objects
            a, b = to_seconds(g[:2])
        spans.append((float(a), float(b), color))
    return spans


def timeseries_gaps(
    config: Settings,
    series: Union[pd.Series, List[pd.Series]],
    gaps: Union[list, List[list]],
) -> LineSpec:
    spec = timeseries(config, series, mini=False)
    if isinstance(series, list):
        colors = comparison_colors(config)
        for i, g in enumerate(gaps):
            spec.spans.extend(_gap_spans(g, colors[i]))
    else:
        spec.spans = _gap_spans(gaps, comparison_colors(config)[0])
    spec.title = "Gap analysis"
    return spec


def overview_timeseries(config: Settings, variables: Any, scale: bool) -> LineSpec:
    lines: List[LineSeries] = []
    is_date = False
    col = next(iter(variables))
    comparison = isinstance(variables[col]["type"], list)
    colors = pal.categorical(len(variables))

    def prep(s: pd.Series) -> pd.Series:
        if scale:
            rng = s.max() - s.min()
            return (s - s.min()) / (rng if rng else 1)
        return s

    for k, (name, data) in enumerate(variables.items()):
        if comparison:
            if not all(t == "TimeSeries" for t in data["type"]):
                continue
            for i, s in enumerate(data["series"]):
                x, y, is_date = _xy(prep(s))
                lines.append(
                    LineSeries(
                        x.tolist(),
                        y.tolist(),
                        colors[k % len(colors)],
                        f"{name} ({config.html.style._labels[i]})",
                        dash=i > 0,
                    )
                )
        elif data["type"] == "TimeSeries":
            x, y, is_date = _xy(prep(data["series"]))
            lines.append(
                LineSeries(
                    x.tolist(), y.tolist(), colors[len(lines) % len(colors)], str(name)
                )
            )
    return LineSpec(
        title="Time series" + (" (scaled)" if scale else ""),
        series=lines or [LineSeries([0.0], [0.0], primary(config))],
        x_is_date=is_date,
        legend=True,
        width=720,
        height=320,
    )


def acf_pacf(
    config: Settings, series: Union[pd.Series, List[pd.Series]]
) -> StemGridSpec:
    from statsmodels.tsa.stattools import acf, pacf

    from data_profiling.visualisation.plot import _get_ts_lag

    many = isinstance(series, list)
    items = series if many else [series]
    colors = comparison_colors(config) if many else [primary(config)]
    rows: List[List[StemPanel]] = []
    for s, color in zip(items, colors):
        data = s.dropna()
        lag = int(_get_ts_lag(config, data))
        panels = []
        for title, fn, kw in (
            ("ACF", acf, {"fft": True}),
            ("PACF", pacf, {"method": "ywm"}),
        ):
            vals, conf = fn(data, nlags=lag, alpha=0.05, **kw)
            vals = np.asarray(vals, float)
            conf = np.asarray(conf, float)
            lo = (conf[:, 0] - vals).tolist()
            hi = (conf[:, 1] - vals).tolist()
            # The lag-0 band is degenerate; mirror statsmodels and hide it.
            lo[0], hi[0] = 0.0, 0.0
            panels.append(StemPanel(title, vals.tolist(), lo, hi, color))
        rows.append(panels)
    return StemGridSpec(
        title="Autocorrelation",
        rows=rows,
        row_labels=[str(x) for x in config.html.style._labels] if many else [],
        width=900,
    )
