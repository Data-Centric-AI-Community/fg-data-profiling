"""HTML report shell: colour modes, accent, transparency, overview tiles,
variable rows and the native word cloud."""
import re
import warnings

import numpy as np
import pandas as pd
import pytest

from data_profiling import ProfileReport
from data_profiling.config import Settings, Theme
from data_profiling.report.presentation.flavours.html.theme import (
    DARK_SURFACE,
    LIGHT_SURFACE,
    contrast,
    theme_tokens,
)
from data_profiling.visualisation import plot


@pytest.fixture(scope="module")
def shell_df():
    rng = np.random.default_rng(0)
    words = [
        "data",
        "quality",
        "report",
        "profile",
        "missing",
        "value",
        "column",
        "table",
    ]
    return pd.DataFrame(
        {
            "num": rng.normal(size=200),
            "cat": rng.choice(["a", "b", "c"], 200),
            "text": [" ".join(rng.choice(words, size=12)) for _ in range(200)],
        }
    )


def _html(df, **kwargs):
    return ProfileReport(df, progress_bar=False, **kwargs).to_html()


def test_default_is_light_with_top_bar(shell_df):
    html = _html(shell_df, minimal=True)
    assert "data-bs-theme=light" in html
    assert "data-dp-mode=light" in html
    assert re.search(r"<header class=\"?dp-topbar", html)
    assert not re.search(r"<body class=\"?dp-root dp-transparent", html)
    # Bootswatch themes are gone: only the base Bootstrap sheet is inlined.
    assert "Bootswatch" not in html


@pytest.mark.parametrize("mode", ["dark", "auto"])
def test_colour_modes(shell_df, mode):
    html = _html(shell_df, minimal=True, html={"style": {"theme": mode}})
    assert f"data-dp-mode={mode}" in html
    if mode == "dark":
        assert "data-bs-theme=dark" in html


def test_transparent_background(shell_df):
    html = _html(
        shell_df, minimal=True, html={"style": {"transparent_background": True}}
    )
    assert re.search(r"<body class=\"?dp-root dp-transparent", html)


def test_navbar_can_be_removed_without_losing_sections(shell_df):
    html = _html(shell_df, minimal=True, html={"navbar_show": False})
    assert not re.search(r"<header class=\"?dp-topbar", html)
    assert "Overview</h1>" in html
    assert "Variables</h1>" in html


@pytest.mark.parametrize("legacy", ["united", "flatly", "cosmo", "simplex"])
def test_legacy_themes_warn_and_become_accents(legacy):
    with pytest.warns(FutureWarning, match=legacy):
        settings = Settings(html={"style": {"theme": legacy}})
    style = settings.html.style
    assert style.theme == Theme(legacy)
    assert style.theme_mode == "light"
    assert style.accent_color != style.primary_colors[0]


def test_legacy_theme_keeps_custom_primary_colours():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        settings = Settings(
            html={"style": {"theme": "united", "primary_colors": ["#123456"]}}
        )
    assert settings.html.style.accent_color == "#123456"


def test_new_themes_do_not_warn():
    with warnings.catch_warnings():
        warnings.simplefilter("error", FutureWarning)
        for mode in ("light", "dark", "auto"):
            Settings(html={"style": {"theme": mode}})


@pytest.mark.parametrize("accent", ["#0d6efd", "#ffd400", "#1b1b1b", "#18bc9c"])
def test_accent_shades_are_readable(accent):
    tokens = theme_tokens(
        Settings(html={"style": {"primary_colors": [accent]}}).html.style
    )
    assert contrast(tokens["light"]["link"], LIGHT_SURFACE) >= 4.5
    assert contrast(tokens["dark"]["link"], DARK_SURFACE) >= 4.5


def test_accent_reaches_the_stylesheet(shell_df):
    html = _html(
        shell_df, minimal=True, html={"style": {"primary_colors": ["#7a3e8e"]}}
    )
    light = theme_tokens(
        Settings(html={"style": {"primary_colors": ["#7a3e8e"]}}).html.style
    )["light"]["link"]
    assert f"--dp-accent:{light}" in html.replace(" ", "")


def test_overview_tiles_and_cards(shell_df):
    html = _html(shell_df, minimal=True)
    assert "dp-kpis" in html
    for label in ("Rows", "Variables", "Missing cells", "Alerts"):
        assert f">{label}<" in html
    # Overview blocks are cards, not tabs.
    assert "dp-overview-grid" in html


def test_variable_rows_carry_filter_metadata(shell_df):
    html = _html(shell_df, minimal=True)
    assert re.search(r"data-name=\"?num\"?", html)
    assert re.search(r"data-type=\"?Numeric\"?", html)
    assert "data-dp-var-filter" in html


def test_more_details_is_a_native_disclosure(shell_df):
    html = _html(shell_df, minimal=True)
    assert "<details" in html
    assert "More details</summary>" in html
    assert "dp-panels" in html


def test_native_word_cloud_is_svg_text():
    counts = pd.Series({"data": 40, "quality": 25, "report": 10, "profile": 4})
    svg = plot.plot_word_cloud(Settings(), counts)
    assert svg.lstrip().startswith("<svg")
    assert "<image" not in svg and "data:image" not in svg
    words = re.findall(r'<text class="dp-wc dp-wc-\d+"[^>]*>([^<]+)<title>', svg)
    assert set(words) == set(counts.index)
    # Larger counts are drawn larger.
    sizes = dict(
        re.findall(r'font-size="([\d.]+)"[^>]*>([^<]+)<title>', svg)[i][::-1]
        for i in range(len(words))
    )
    assert float(sizes["data"]) > float(sizes["profile"])


def test_word_cloud_compare_has_one_panel_per_report():
    a = pd.Series({"data": 5, "quality": 3})
    b = pd.Series({"data": 2, "report": 4})
    config = Settings()
    config.html.style._labels = ["A", "B"]
    svg = plot.plot_word_cloud(config, [a, b])
    assert ">A</text>" in svg and ">B</text>" in svg


def test_text_variable_shows_cloud_and_top_words(shell_df):
    html = _html(shell_df, minimal=False, vars={"text": {"words": True}})
    start = html.index("data-name=text")
    row = html[start : html.index("</article>", start)]
    assert "dp-chart-word-cloud" in row
    # Top five words, without an "Other values" or "(Missing)" remainder.
    top = row[row.index("minifreqtable") :]
    top = top[: top.index("</table>")]
    assert "Other values" not in top and "(Missing)" not in top


@pytest.fixture(scope="module")
def sections_html(shell_df):
    df = shell_df.assign(
        num2=shell_df["num"] * 2 + 1, num3=-shell_df["num"], cat2=shell_df["cat"]
    )
    df.loc[::7, "num2"] = np.nan
    df.loc[::5, "num3"] = np.nan
    return _html(
        df,
        correlations={
            k: {"calculate": True}
            for k in ("auto", "pearson", "spearman", "kendall", "phi_k")
        },
        missing_diagrams={"bar": True, "matrix": True, "heatmap": True},
    )


def test_interactions_explorer_holds_every_pair(sections_html):
    assert "data-dp-explorer" in sections_html
    panes = re.findall(r'class="?dp-pane"? [^>]*data-x=', sections_html)
    assert len(panes) >= 4
    # Opens on the strongest pair of distinct variables.
    default = re.search(
        r"data-default-x=\"?([\w]+)\"? data-default-y=\"?([\w]+)", sections_html
    )
    assert default and default.group(1) != default.group(2)
    assert "data-stat-label" in sections_html


def test_correlations_method_switch_and_pairs(sections_html):
    assert "data-dp-switch" in sections_html
    for label in ("Pearson&#39;s r", "Spearman&#39;s ρ", "Kendall&#39;s τ"):
        assert label in sections_html or label.replace("&#39;", "'") in sections_html
    assert "Strongest pairs" in sections_html
    assert "Show the values as a table" in sections_html


def test_strongest_pairs_are_ranked():
    from data_profiling.report.structure.correlations import get_strongest_pairs

    m = pd.DataFrame(
        [[1, 0.2, -0.9], [0.2, 1, 0.5], [-0.9, 0.5, 1]],
        index=list("abc"),
        columns=list("abc"),
    )
    pairs = get_strongest_pairs(Settings(), {"pearson": m}, top=2)
    assert [(p["x"], p["y"]) for p in pairs] == [("a", "c"), ("b", "c")]
    assert pairs[0]["negative"]


def test_missing_values_are_cards(sections_html):
    assert "dp-cards" in sections_html
    assert "Count by variable" in sections_html
    assert "Where values are missing" in sections_html
    assert "Missing together" in sections_html


def test_no_vendor_footer(shell_df):
    html = _html(shell_df, minimal=True)
    assert "ydata.ai" not in html
    assert "Report generated by" not in html
    assert "<footer" not in html
