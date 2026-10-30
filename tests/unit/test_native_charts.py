"""Tests for the native (inline SVG) chart renderer and its compatibility."""

import json
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd
import pytest

from data_profiling import ProfileReport
from data_profiling.config import Settings
from data_profiling.visualisation import charts
from data_profiling.visualisation import missing as missing_plots
from data_profiling.visualisation import plot

SVG_NS = "{http://www.w3.org/2000/svg}"


def _assert_native(markup: str) -> ET.Element:
    assert charts.is_inline_svg(markup)
    assert "data:image" not in markup and "<image" not in markup
    root = ET.fromstring(markup)  # well-formed XML
    assert root.tag == f"{SVG_NS}svg"
    assert root.get("viewBox")
    assert "id=" not in markup  # safe to inline many charts in one document
    return root


@pytest.fixture
def config() -> Settings:
    return Settings()


@pytest.mark.parametrize(
    "plot_cfg, expected",
    [
        ({}, True),
        ({"image_format": "png"}, False),
        ({"image_format": "svg"}, True),
        ({"renderer": "matplotlib"}, False),
        ({"renderer": "native", "image_format": "png"}, True),
    ],
)
def test_renderer_resolution(plot_cfg, expected):
    assert charts.use_native(Settings(plot=plot_cfg)) is expected


def test_histograms(config):
    counts, edges = np.histogram(np.random.default_rng(0).normal(size=500), bins=20)
    _assert_native(plot.histogram(config, counts, edges))
    _assert_native(plot.mini_histogram(config, counts, edges))
    dates = edges * 86400 * 365 + 1.6e9
    _assert_native(plot.histogram(config, counts, dates, date=True))
    assert plot.histogram(config, np.array([]), edges) is None


def test_comparison_histogram(config):
    config.html.style._labels = ["a", "b"]
    c1, e1 = np.histogram(np.arange(10), bins=5)
    c2, e2 = np.histogram(np.arange(10) + 2, bins=5)
    root = _assert_native(plot.histogram(config, [c1, c2], [e1, e2]))
    labels = [t.text for t in root.iter(f"{SVG_NS}text")]
    assert "a" in labels and "b" in labels


@pytest.mark.parametrize("plot_type", ["bar", "pie"])
def test_cat_frequency(config, plot_type):
    config.plot.cat_freq.type = plot_type
    data = pd.Series({"x": 5, "y": 3, "z": 1})
    markup = plot.cat_frequency_plot(config, data)
    _assert_native(markup)
    assert "x · Count: 5" in markup


def test_cat_frequency_folds_long_tail(config):
    data = pd.Series({f"c{i}": 20 - i for i in range(12)})
    markup = plot.cat_frequency_plot(config, data)
    assert "Other" in markup and "c11" not in markup


def test_cat_frequency_custom_colors(config):
    config.plot.cat_freq.colors = ["gold", "#123456"]
    markup = plot.cat_frequency_plot(config, pd.Series({"x": 5, "y": 3}))
    assert "#ffd700" in markup and "#123456" in markup


def test_cat_frequency_invalid_type(config):
    config.plot.cat_freq.type = "donut"
    with pytest.raises(ValueError):
        plot.cat_frequency_plot(config, pd.Series({"x": 1}))


def test_cat_frequency_redacted(config):
    config.vars.cat.redact = True
    markup = plot.cat_frequency_plot(config, pd.Series({"secret": 5, "hidden": 3}))
    _assert_native(markup)
    assert "secret" not in markup and "hidden" not in markup


@pytest.mark.parametrize("vmin", [-1, 0])
def test_correlation_matrix(config, vmin):
    df = pd.DataFrame(
        np.random.default_rng(1).normal(size=(50, 4)), columns=list("abcd")
    )
    corr = df.corr()
    corr.iloc[0, 1] = np.nan
    markup = plot.correlation_matrix(config, corr, vmin=vmin)
    _assert_native(markup)
    assert "invalid coefficient" in markup


def test_scatter_and_hexbin(config):
    rng = np.random.default_rng(2)
    small = pd.Series(rng.normal(size=50))
    root = _assert_native(plot.scatter_pairwise(config, small, small * 2, "x", "y"))
    assert len(list(root.iter(f"{SVG_NS}circle"))) == 50
    big = pd.Series(rng.normal(size=5000))
    markup = plot.scatter_pairwise(config, big, big + rng.normal(size=5000), "x", "y")
    _assert_native(markup)
    assert "hexbin" in markup and "<circle" not in markup
    _assert_native(plot.scatter_complex(config, pd.Series(rng.normal(size=20) + 1j)))
    dims = pd.Series([(800, 600), (1024, 768), (640, 480)])
    _assert_native(plot.scatter_series(config, dims))


def test_word_frequency(config):
    words = pd.Series({"alpha": 10, "beta": 4, "gamma": 2})
    markup = plot.plot_word_cloud(config, words)
    _assert_native(markup)
    assert "alpha" in markup


def test_timeseries(config):
    idx = pd.date_range("2024-01-01", periods=5000, freq="h")
    s = pd.Series(np.sin(np.arange(5000) / 50), index=idx, name="s")
    root = _assert_native(plot.mini_ts_plot(config, s))
    path = max((p.get("d", "") for p in root.iter(f"{SVG_NS}path")), key=len)
    assert path.count("L") < 1500  # downsampled
    _assert_native(plot.mini_ts_plot(config, s, figsize=(7, 3)))
    gaps = [np.array([idx[10], idx[200]])]
    _assert_native(plot.plot_timeseries_gap_analysis(config, s, gaps))
    _assert_native(plot.plot_acf_pacf(config, s))
    variables = {"s": {"type": "TimeSeries", "series": s}}
    _assert_native(plot.plot_overview_timeseries(config, variables, scale=True))


def test_missing_plots(config):
    df = pd.DataFrame(
        np.random.default_rng(3).normal(size=(1000, 4)), columns=list("abcd")
    )
    df[df > 1.2] = np.nan
    _assert_native(
        missing_plots.plot_missing_bar(
            config, len(df) - df.isnull().sum(), len(df), list(df)
        )
    )
    root = _assert_native(
        missing_plots.plot_missing_matrix(
            config, df.notnull().values, list(df), len(df)
        )
    )
    # rows are bucketed: markup size does not scale with the number of rows
    assert len(list(root.iter(f"{SVG_NS}rect"))) < 4 * 210
    corr = df.isnull().corr()
    mask = np.zeros_like(corr)
    mask[np.triu_indices_from(mask)] = True
    _assert_native(missing_plots.plot_missing_heatmap(config, corr, mask, list(df)))


def test_matplotlib_renderer_unchanged():
    config = Settings(plot={"renderer": "matplotlib"})
    counts, edges = np.histogram(np.arange(100), bins=10)
    out = plot.histogram(config, counts, edges)
    assert not charts.is_inline_svg(out)
    assert "matplotlib" in out
    config = Settings(plot={"image_format": "png"})
    assert plot.histogram(config, counts, edges).startswith("data:image/png")


@pytest.fixture(scope="module")
def report_df() -> pd.DataFrame:
    rng = np.random.default_rng(4)
    df = pd.DataFrame(
        {
            "num": rng.normal(size=300),
            "cat": rng.choice(list("abc"), 300),
            "flag": rng.choice([True, False], 300),
            "when": pd.date_range("2024-01-01", periods=300, freq="D"),
        }
    )
    df.loc[::7, "num"] = np.nan
    return df


def test_report_has_no_matplotlib_images(report_df):
    html = ProfileReport(report_df, explorative=True, progress_bar=False).to_html()
    assert "dp-chart" in html
    assert "matplotlib" not in html


def test_report_json_still_serialises(report_df):
    data = json.loads(
        ProfileReport(report_df, minimal=True, progress_bar=False).to_json()
    )
    assert "variables" in data


def test_widget_flavour_renders_native(report_df):
    pytest.importorskip("ipywidgets")
    from data_profiling.report.presentation.flavours.widget.image import WidgetImage

    markup = plot.histogram(Settings(), *np.histogram(np.arange(10), bins=3))
    widget = WidgetImage(markup, image_format=Settings().plot.image_format, alt="h")
    assert widget.render().value.startswith("<svg")


# --- Hover details -------------------------------------------------------------


def _data_attr(markup: str, name: str) -> dict:
    attr = ET.fromstring(markup).get(f"data-dp-{name}")
    assert attr is not None
    return json.loads(attr)


def test_histogram_tooltips(config):
    counts, edges = np.histogram(np.arange(100), bins=4)
    markup = plot.histogram(config, counts, edges)
    assert "Share: 25.0%" in markup and "Cumulative: 100.0%" in markup
    assert 'class="dp-hit"' in markup  # full-height hover columns


def test_comparison_histogram_single_combined_tooltip(config):
    config.html.style._labels = ["a", "b"]
    edges = np.linspace(0, 10, 6)
    c1, _ = np.histogram(np.arange(10), bins=edges)
    c2, _ = np.histogram(np.arange(10) // 2, bins=edges)
    root = ET.fromstring(plot.histogram(config, [c1, c2], [edges, edges]))
    hit_tips = [
        r.find(f"{SVG_NS}title").text
        for r in root.iter(f"{SVG_NS}rect")
        if r.get("class") == "dp-hit"
    ]
    assert len(hit_tips) == 5
    assert all("a frequency" in t and "b frequency" in t for t in hit_tips)


def test_missing_bar_tooltip(config):
    counts = pd.Series({"x": 8, "y": 10})
    markup = missing_plots.plot_missing_bar(config, counts, 10, ["x", "y"])
    assert "Missing: 2 (20.0%)" in markup and "Present: 8 (80.0%)" in markup


def test_heatmap_interpretation(config):
    corr = pd.DataFrame(
        [[1.0, 0.82], [0.82, 1.0]], columns=["a", "b"], index=["a", "b"]
    )
    markup = plot.correlation_matrix(config, corr)
    assert "Strength: very strong positive" in markup and "Same variable" in markup


def test_stacked_bar_tooltip(config):
    markup = plot.cat_frequency_plot(config, pd.Series({"x": 6, "y": 4}))
    assert "Rank: 1 of 2" in markup and "Share: 60.0%" in markup


def test_line_crosshair_data(config):
    idx = pd.date_range("2024-01-01", periods=100, freq="D")
    s = pd.Series(np.arange(100.0), index=idx, name="s")
    data = _data_attr(plot.mini_ts_plot(config, s, figsize=(7, 3)), "line")
    assert data["d"] is True and len(data["x"]) == 100
    assert data["s"][0]["y"][:3] == [0, 1, 2]


def test_scatter_hover_data(config):
    s = pd.Series(np.arange(20.0))
    data = _data_attr(plot.scatter_pairwise(config, s, s * 2, "u", "v"), "scatter")
    assert (data["xl"], data["yl"]) == ("u", "v") and len(data["x"]) == 20


def test_acf_tooltips(config):
    s = pd.Series(np.sin(np.arange(200) / 5))
    markup = plot.plot_acf_pacf(config, s)
    assert "Significant: yes" in markup and "95% band" in markup


def test_report_script_is_jinja_safe():
    from pathlib import Path

    import data_profiling

    js = (
        Path(data_profiling.__file__).parent
        / "report/presentation/flavours/html/templates/wrapper/assets/script.js"
    ).read_text()
    assert "dp-tooltip" in js
    assert not any(tok in js for tok in ("{{", "{%", "{#"))


def test_histogram_large_integers_terminates(config):
    # issue 915: values ~7e17 spanning 100 are below float resolution per step
    edges = 716277643516076032.0 + np.array([0.0, 128.0, 256.0])
    counts = np.array([50, 50])
    _assert_native(plot.histogram(config, counts, edges))
    _assert_native(plot.mini_histogram(config, counts, edges))


@pytest.mark.parametrize(
    "start, end", [("1900-01-01", "2000-01-01"), ("1700-01-01", "2250-01-01")]
)
def test_date_ticks_long_spans_stay_small(start, end):
    # A span beyond the largest calendar step must not fall back to 1-second
    # ticks (which would materialise billions of timestamps).
    from data_profiling.visualisation.charts.scales import date_ticks

    lo, hi = pd.Timestamp(start).value / 1e9, pd.Timestamp(end).value / 1e9
    ticks = date_ticks(lo, hi, 6)
    assert 2 <= len(ticks) <= 24
    assert all(label[:4].isdigit() for _, label in ticks)


def test_date_histogram_century(config):
    lo, hi = (
        pd.Timestamp("1900-01-01").value / 1e9,
        pd.Timestamp("2000-01-01").value / 1e9,
    )
    edges = np.linspace(lo, hi, 11)
    counts = np.arange(10)
    _assert_native(plot.histogram(config, counts, edges, date=True))
