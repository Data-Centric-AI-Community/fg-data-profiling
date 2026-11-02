"""Behavioral and report integration coverage for missingness associations."""

import importlib
import json
import re
from html.parser import HTMLParser
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from data_profiling import ProfileReport, compare
from data_profiling.config import Settings
from data_profiling.model.missing import get_missing_associations


def association_config(**kwargs):
    return Settings(missing_diagrams={"associations": True}, **kwargs)


def summaries(df, **types):
    """Supply the already computed summaries consumed by the missing pipeline."""
    return {
        name: {
            "type": types.get(name, "Numeric"),
            "n_missing": int(series.isna().sum()),
            "n_distinct": int(series.nunique()),
        }
        for name, series in df.items()
    }


def analyze(df, **types):
    return get_missing_associations(association_config(), df, summaries(df, **types))


def matching(result, kind, target=None, predictor=None):
    return [
        finding
        for finding in result["findings"]
        if finding["kind"] == kind
        and (target is None or finding["target"] == target)
        and (predictor is None or finding["predictor"] == predictor)
    ]


@pytest.fixture
def planted():
    return pd.DataFrame(
        {
            "target": [np.nan] * 100 + list(range(100)),
            "twin": [np.nan] * 100 + list(range(100)),
            "category": ["affected"] * 100 + ["control"] * 100,
            "boolean": [True] * 100 + [False] * 100,
        }
    )


def test_missing_together_is_one_unordered_pair(planted):
    result = analyze(planted)
    pairs = matching(result, "missingness")
    assert len(pairs) == 1
    assert {pairs[0]["target"], pairs[0]["predictor"]} == {"target", "twin"}
    assert pairs[0]["strength"] == pytest.approx(1.0)
    assert pairs[0]["group_missing_rate"] == pytest.approx(1.0)
    assert pairs[0]["complement_missing_rate"] == pytest.approx(0.0)


@pytest.mark.parametrize(
    "predictor,type_name", [("category", "Categorical"), ("boolean", "Boolean")]
)
def test_categorical_and_boolean_predictors(planted, predictor, type_name):
    df = planted[["target", predictor]]
    result = analyze(df, **{predictor: type_name})
    findings = matching(result, "categorical", "target", predictor)
    assert len(findings) == 1
    assert findings[0]["group_missing_rate"] == pytest.approx(1.0)
    assert findings[0]["complement_missing_rate"] == pytest.approx(0.0)
    assert isinstance(findings[0]["value"], str)


def test_independent_missing_at_random_control():
    rng = np.random.default_rng(8123)
    df = pd.DataFrame(rng.normal(size=(5000, 4)), columns=list("abcd"))
    df = df.mask(rng.random(df.shape) < 0.25)
    df["category"] = rng.choice(["x", "y", "z"], len(df))
    df["boolean"] = rng.choice([True, False], len(df))
    assert analyze(df, category="Categorical", boolean="Boolean")["findings"] == []


def test_custom_numeric_schema_excludes_categorical_testing(planted):
    df = planted[["target", "boolean"]].copy()
    df["boolean"] = df["boolean"].astype(int)
    assert analyze(df, boolean="Numeric")["findings"] == []
    assert matching(analyze(df, boolean="Categorical"), "categorical")


def test_constant_observed_values_still_have_missingness_signal():
    df = pd.DataFrame({"a": [None] * 100 + [1] * 100, "b": [None] * 100 + [2] * 100})
    assert len(matching(analyze(df), "missingness")) == 1


def test_partial_comissingness_is_detected():
    # Joint missing/present table: [[80, 20], [20, 80]], phi = 0.6.
    df = pd.DataFrame(
        {
            "a": [None] * 100 + [1] * 100,
            "b": [None] * 80 + [1] * 20 + [None] * 20 + [1] * 80,
        }
    )
    findings = matching(analyze(df), "missingness")
    assert len(findings) == 1
    assert findings[0]["strength"] == pytest.approx(0.6)
    assert findings[0]["group_missing_rate"] == pytest.approx(0.8)
    assert findings[0]["complement_missing_rate"] == pytest.approx(0.2)


def test_pair_detection_is_invariant_to_column_name_order():
    # A minority of the broad missing group is a narrower missing group. The
    # conditional rate uplift differs by direction, while the pair is the same.
    df = pd.DataFrame({"a": [None] * 50 + [1] * 950, "b": [None] * 300 + [1] * 700})
    original = matching(analyze(df), "missingness")
    renamed = matching(analyze(df.rename(columns={"a": "z"})), "missingness")
    assert len(original) == len(renamed) == 1
    assert original[0]["strength"] == pytest.approx(renamed[0]["strength"])


def test_multiple_testing_correction_rejects_borderline_category_signal():
    df = pd.DataFrame(
        {
            "target": [None] * 40 + list(range(40)),
            "signal": ["affected"] * 27
            + ["control"] * 13
            + ["affected"] * 13
            + ["control"] * 27,
        }
    )
    assert len(matching(analyze(df, signal="Categorical"), "categorical")) == 1
    for index in range(30):
        # Exactly equal missingness in these groups; each contributes a true
        # null hypothesis to the same predeclared testing family.
        df[f"control_{index}"] = ["x", "y"] * 40
    types = {name: "Categorical" for name in df if name != "target"}
    assert analyze(df, **types)["findings"] == []


def test_opposite_missingness_is_not_reported_as_missing_together():
    df = pd.DataFrame({"a": [None] * 100 + [1] * 100, "b": [1] * 100 + [None] * 100})
    assert matching(analyze(df), "missingness") == []


@pytest.mark.parametrize("dtype", ["boolean", "category"])
def test_nullable_and_categorical_predictor_dtypes(dtype):
    df = pd.DataFrame(
        {
            "target": [None] * 100 + list(range(100)),
            "predictor": pd.Series(
                [True] * 100 + [False] * 80 + [None] * 20, dtype=dtype
            ),
        }
    )
    type_name = "Boolean" if dtype == "boolean" else "Categorical"
    findings = matching(
        analyze(df, predictor=type_name), "categorical", "target", "predictor"
    )
    assert len(findings) == 1
    assert findings[0]["group_missing_rate"] == 1
    assert findings[0]["complement_missing_rate"] == 0


def test_column_limit_returns_explicit_skip():
    df = pd.DataFrame({f"x{i}": [None] * 100 + [1] * 100 for i in range(101)})
    result = analyze(df)
    assert result["status"] == "skipped_limits"
    assert result["findings"] == []


def test_findings_are_bounded_and_ties_are_deterministic():
    df = pd.DataFrame({f"x{i:02d}": [None] * 100 + [1] * 100 for i in range(15)})
    first = analyze(df)
    assert len(first["findings"]) == 50
    assert first == analyze(df)


def test_unsupported_predictors_are_not_grouped(planted):
    df = planted[["target", "category"]]
    assert analyze(df, category="Unsupported")["findings"] == []


@pytest.mark.parametrize(
    "df",
    [
        pd.DataFrame(),
        pd.DataFrame({"a": []}),
        pd.DataFrame({"a": [None] * 100}),
        pd.DataFrame({"a": [1] * 100, "b": ["x"] * 100}),
    ],
)
def test_empty_and_constant_inputs(df):
    assert analyze(df)["findings"] == []


def test_rare_groups_and_high_cardinality_are_not_reported():
    df = pd.DataFrame(
        {
            "target": [None] + list(range(199)),
            "rare": ["rare"] + ["common"] * 199,
            "identifier": [f"id-{i}" for i in range(200)],
        }
    )
    assert analyze(df, rare="Categorical", identifier="Categorical")["findings"] == []


@pytest.mark.parametrize("support,expected", [(19, False), (20, True)])
def test_minimum_group_support_boundary(support, expected):
    df = pd.DataFrame(
        {
            "target": [None] * support + list(range(200 - support)),
            "predictor": ["affected"] * support + ["control"] * (200 - support),
        }
    )
    assert (
        bool(matching(analyze(df, predictor="Categorical"), "categorical")) is expected
    )


def test_minimum_support_scales_with_dataset_size():
    df = pd.DataFrame(
        {
            "target": [None] * 30 + list(range(4970)),
            "predictor": ["affected"] * 30 + ["control"] * 4970,
        }
    )
    assert analyze(df, predictor="Categorical")["findings"] == []


def test_result_is_deterministic_native_json_and_does_not_mutate(planted):
    original = planted.copy(deep=True)
    first = analyze(planted, category="Categorical", boolean="Boolean")
    second = analyze(planted, category="Categorical", boolean="Boolean")
    assert first == second
    assert json.loads(json.dumps(first, allow_nan=False)) == first
    pd.testing.assert_frame_equal(planted, original)
    identities = [
        (f["kind"], f["target"], f["predictor"], f["value"]) for f in first["findings"]
    ]
    assert len(identities) == len(set(identities))


def test_disabled_does_not_touch_dataframe():
    class Unreadable:
        def __getattr__(self, name):
            raise AssertionError(f"Disabled analysis accessed {name}")

    assert get_missing_associations(Settings(), Unreadable(), {})["findings"] == []


def test_spark_is_explicitly_skipped_without_collecting():
    class SparkLike:
        def __getattr__(self, name):
            raise AssertionError(f"Unsupported backend accessed {name}")

    with pytest.warns(UserWarning, match="(?i)spark|pandas|unsupported"):
        result = get_missing_associations(association_config(), SparkLike(), {})
    assert result["findings"] == []
    assert (
        "unsupported" in result["status"].lower() or "skip" in result["status"].lower()
    )


def report(df, **kwargs):
    return ProfileReport(
        df,
        progress_bar=False,
        pool_size=1,
        correlations=None,
        interactions={"continuous": False},
        duplicates={"head": 0},
        missing_diagrams={
            "bar": False,
            "matrix": False,
            "heatmap": False,
            "associations": True,
        },
        **kwargs,
    )


def test_minimal_bypasses_entrypoint_even_with_explicit_enable(planted):
    describe = importlib.import_module("data_profiling.model.describe")
    with patch.object(
        describe, "get_missing_associations", side_effect=AssertionError("minimal scan")
    ):
        description = report(planted, minimal=True).description_set
    assert not description.missing.get("associations", {}).get("findings", [])


def test_report_reuses_findings_for_alerts_json_and_html(planted):
    describe = importlib.import_module("data_profiling.model.describe")
    with patch.object(
        describe, "get_missing_associations", wraps=get_missing_associations
    ) as compute:
        profile = report(planted)
        description = profile.description_set
        result = description.missing["associations"]
        assert result["findings"]
        payload = json.loads(profile.to_json())
        assert payload["missing"]["associations"] == result
        assert "association" in profile.to_html().lower()
        alerts = [
            alert
            for alert in description.alerts
            if "ASSOCIATION" in alert.alert_type.name
        ]
        assert {alert.column_name for alert in alerts} == {
            f["target"] for f in result["findings"]
        }
        assert len(alerts) == len({alert.column_name for alert in alerts})
        assert re.search(r'href=["\']?#missing(?:["\'\s>])', profile.to_html())
        assert compute.call_count == 1


def test_comparison_with_empty_findings_renders(planted):
    positive = report(planted)
    control = report(planted.fillna(0))
    assert not control.description_set.missing["associations"]["findings"]
    compared = positive.compare(control)
    assert "association" in compared.to_html().lower()
    assert "associations" in json.loads(compared.to_json())["missing"]


def test_comparison_with_disabled_dataset_preserves_dataset_boundaries(planted):
    enabled = report(planted)
    disabled = ProfileReport(planted, minimal=True, progress_bar=False, pool_size=1)
    # The module function retains the individual compute configurations. The
    # instance shortcut explicitly reapplies its caller's config to both inputs.
    comparison = compare([enabled, disabled])
    datasets = comparison.description_set.missing["associations"]["datasets"]
    assert len(datasets) == 2
    assert datasets[0]["findings"]
    assert not datasets[1]["findings"]
    assert "association" in comparison.to_html().lower()


def test_comparison_alias_cannot_enable_analysis_for_minimal_source(planted):
    enabled = report(planted)
    disabled = ProfileReport(planted, minimal=True, progress_bar=False, pool_size=1)
    comparison = enabled.compare(disabled)
    datasets = comparison.description_set.missing["associations"]["datasets"]
    assert datasets[0]["findings"]
    assert not datasets[1]["findings"]


def test_html_escapes_category_labels(planted):
    df = planted[["target", "category"]].replace("affected", "<script>secret</script>")
    html = report(df).to_html()

    class Scripts(HTMLParser):
        def __init__(self):
            super().__init__()
            self.in_script = False
            self.contents = []

        def handle_starttag(self, tag, attrs):
            self.in_script = tag == "script"

        def handle_endtag(self, tag):
            if tag == "script":
                self.in_script = False

        def handle_data(self, data):
            if self.in_script:
                self.contents.append(data)

    scripts = Scripts()
    scripts.feed(html)
    assert "secret" not in scripts.contents


def test_comparison_config_can_hide_associations_and_summary_alerts(planted):
    from data_profiling import compare

    descriptions = [report(planted).description_set, report(planted).description_set]
    config = Settings(missing_diagrams={"associations": False})
    comparison = compare(descriptions, config=config)
    assert "associations" not in comparison.description_set.missing
    assert all(
        "ASSOCIATION" not in alert.alert_type.name
        for alerts in comparison.description_set.alerts
        for alert in alerts
    )


def test_redacted_categories_are_not_exposed(planted):
    df = planted[["target", "category"]].replace(
        "affected", "private-association-label"
    )
    profile = report(df, vars={"cat": {"redact": True}})
    payload = profile.description_set.missing["associations"]
    assert "private-association-label" not in json.dumps(payload)


@pytest.mark.parametrize("type_schema", [None, {"flag": "boolean"}])
def test_inferred_boolean_spellings_are_normalized(type_schema):
    df = pd.DataFrame(
        {
            "target": [None] * 100 + list(range(100)),
            "flag": ["yes", "YES"] * 50 + ["no", "NO"] * 50,
        }
    )
    profile = report(df, type_schema=type_schema)
    assert profile.description_set.variables["flag"]["type"] == "Boolean"
    findings = matching(
        profile.description_set.missing["associations"], "categorical", "target", "flag"
    )
    assert len(findings) == 1
    assert findings[0]["group_missing_rate"] == pytest.approx(1.0)


def test_custom_numeric_schema_is_respected_by_report():
    df = pd.DataFrame(
        {
            "target": [None] * 100 + list(range(100)),
            "numeric_code": [1] * 100 + [2] * 100,
        }
    )
    profile = report(df, type_schema={"numeric_code": "numeric"})
    assert profile.description_set.variables["numeric_code"]["type"] == "Numeric"
    assert not matching(profile.description_set.missing["associations"], "categorical")


def test_mixed_boolean_representations_do_not_create_false_predictor_groups():
    df = pd.DataFrame(
        {
            "target": [None] * 100 + list(range(100)),
            "flag": [True] * 100 + ["yes"] * 100,
        }
    )
    profile = report(df, type_schema={"flag": "boolean"})
    assert not matching(profile.description_set.missing["associations"], "categorical")
