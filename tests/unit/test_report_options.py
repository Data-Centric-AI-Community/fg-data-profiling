import pandas as pd
import pytest

from data_profiling import ProfileReport


# Enhanced fixture with more diverse data types
@pytest.fixture
def sample_categorical_data():
    return pd.DataFrame(
        {
            "dummy_cat": [
                "Amadeou_plus",
                "Amadeou_plus",
                "Beta_front",
                "Calciumus",
                "Dimitrius",
                "esperagus_anonymoliumus",
                "FrigaTTTBrigde_Writap",
                "galgarartiy",
                "He",
                "I",
                "JimISGODDOT",
            ]
            * 10
        }
    )


@pytest.fixture
def sample_boolean_data():
    return pd.DataFrame({"dummy_bool": [True] * 82 + [False] * 36})


def generate_cat_data_series(categories):
    """Helper function to generate categorical data"""
    dummy_data = []
    for cat, i in categories.items():
        dummy_data.extend([cat] * i)
    return pd.DataFrame({"dummy_cat": dummy_data})


def generate_report(data, **kwargs):
    """Helper function to generate report with common settings"""
    default_settings = {
        "progress_bar": False,
        "samples": None,
        "correlations": None,
        "missing_diagrams": None,
        "duplicates": None,
        "interactions": None,
    }
    default_settings.update(kwargs)
    return ProfileReport(df=data, **default_settings)


# Test category frequency plots general options
@pytest.mark.parametrize(
    "data_fixture",
    ["sample_boolean_data", "sample_categorical_data"],
    ids=["boolean", "categorical"],
)
@pytest.mark.parametrize("plot_type", ["bar", "pie"])
def test_deactivated_cat_frequency_plot(data_fixture, plot_type, request):
    data = request.getfixturevalue(data_fixture)
    profile = generate_report(data)
    profile.config.plot.cat_freq.show = False
    profile.config.plot.cat_freq.type = plot_type
    html_report = profile.to_html()
    assert "Common Values (Plot)" not in html_report


@pytest.mark.parametrize(
    "data_fixture",
    ["sample_boolean_data", "sample_categorical_data"],
    ids=["boolean", "categorical"],
)
def test_cat_frequency_default_barh_plot(data_fixture, request):
    data = request.getfixturevalue(data_fixture)
    profile = generate_report(data)
    html_report = profile.to_html()
    assert "Common Values (Plot)" in html_report


@pytest.mark.parametrize(
    "data_fixture",
    ["sample_boolean_data", "sample_categorical_data"],
    ids=["boolean", "categorical"],
)
def test_cat_frequency_pie_plot(data_fixture, request):
    data = request.getfixturevalue(data_fixture)
    profile = generate_report(data)
    profile.config.plot.cat_freq.type = "pie"
    html_report = profile.to_html()
    assert "pie" in html_report


@pytest.mark.parametrize("plot_type", ["bar", "pie"])
def test_max_unique_categories(plot_type):
    # Test with different numbers of unique categories
    categories = {f"cat_{i}": 5 for i in range(10)}
    data = generate_cat_data_series(categories)
    profile = generate_report(data)
    profile.config.plot.cat_freq.max_unique = 5
    profile.config.plot.cat_freq.type = plot_type
    html_report = profile.to_html()

    # Should not show plot when unique categories exceed max_unique
    assert "Common Values (Plot)" not in html_report


def test_more_categories_than_colors():
    # Test handling when there are more categories than defined colors
    test_data = generate_cat_data_series({f"cat_{i}": 10 for i in range(5)})
    custom_colors = ["gold", "blue", "coral"]

    profile = generate_report(test_data)
    profile.config.plot.cat_freq.colors = custom_colors
    html_report = profile.to_html()

    # Should still generate plot without errors
    assert "Common Values (Plot)" in html_report


@pytest.mark.skip("Skipping empty color list test. Code needs to be updated.")
def test_empty_color_list():
    # Test behavior with empty color list
    test_data = generate_cat_data_series({"A": 10, "B": 10})
    profile = generate_report(test_data)
    profile.config.plot.cat_freq.colors = []
    html_report = profile.to_html()

    # Should use default colors
    assert "Common Values (Plot)" in html_report


@pytest.mark.parametrize("invalid_type", ["scatter", "box", "invalid"])
def test_invalid_plot_types(invalid_type):
    test_data = generate_cat_data_series({"A": 10, "B": 10})

    with pytest.raises(ValueError):
        profile = generate_report(test_data)
        profile.config.plot.cat_freq.type = invalid_type
        profile.to_html()


def test_config_persistence():
    # Test that plot configuration persists after cache invalidation
    test_data = generate_cat_data_series({"A": 10, "B": 10})
    profile = generate_report(test_data)
    profile.config.plot.cat_freq.type = "pie"
    profile.config.plot.cat_freq.colors = ["gold", "blue"]

    # Cache invalidation shouldn't affect config
    profile.invalidate_cache()
    html_report = profile.to_html()
    assert "pie" in html_report
    # native renderer: colour attribute on the SVG mark
    assert "#ffd700" in html_report


def test_config_persistence_matplotlib():
    # Same as above, legacy renderer
    test_data = generate_cat_data_series({"A": 10, "B": 10})
    profile = generate_report(test_data)
    profile.config.plot.renderer = "matplotlib"
    profile.config.plot.cat_freq.type = "pie"
    profile.config.plot.cat_freq.colors = ["gold", "blue"]

    profile.invalidate_cache()
    html_report = profile.to_html()
    assert "pie" in html_report
    assert "fill: #ffd700" in html_report
