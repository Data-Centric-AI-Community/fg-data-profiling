"""Tests for the agent-oriented JSON output (data-profiling.agent/1)."""

import json

import numpy as np
import pandas as pd
import pytest

from data_profiling import ProfileReport, agent
from data_profiling.controller import console
from data_profiling.model.agent import estimate_tokens


@pytest.fixture(scope="module")
def df():
    rng = np.random.default_rng(42)
    n = 300
    data = pd.DataFrame(
        {
            "age": rng.integers(18, 90, n),
            "income": rng.lognormal(10, 1, n).round(2),
            "city": rng.choice(["Lisbon", "Porto", "Faro"], n),
            "id": np.arange(n),
            "signup": pd.date_range("2020-01-01", periods=n, freq="D"),
            "flag": rng.choice([True, False], n),
            "review": [
                " ".join(
                    rng.choice(["good", "bad", "slow", "fast"], int(rng.integers(2, 6)))
                )
                for _ in range(n)
            ],
            "const": 1,
        }
    )
    data["spend"] = data["income"] * 0.3 + rng.normal(0, 100, n)
    data.loc[rng.choice(n, 30, replace=False), ["income", "spend"]] = np.nan
    return data


@pytest.fixture(scope="module")
def report(df):
    return ProfileReport(df, title="fixture", progress_bar=False)


def _validate(payload):
    jsonschema = pytest.importorskip("jsonschema")
    jsonschema.validate(payload, agent.json_schema())


@pytest.mark.parametrize("detail", ["brief", "standard", "full"])
def test_default_output_matches_schema(report, detail):
    payload = json.loads(report.to_json(detail=detail))
    assert payload["schema"] == agent.SCHEMA_ID
    assert payload["detail"] == detail
    _validate(payload)


def test_header_and_order(report):
    payload = json.loads(report.to_json())
    assert list(payload)[:5] == ["schema", "detail", "dataset", "findings", "columns"]
    ds = payload["dataset"]
    assert ds["rows"] == 300 and ds["columns"] == 9
    assert ds["types"]["numeric"] >= 3
    assert sum(ds["roles"].values()) == 9


def test_findings_sorted_by_severity(report):
    findings = json.loads(report.to_json())["findings"]
    order = [agent.SEVERITIES.index(f["sev"]) for f in findings]
    assert order == sorted(order)
    constant = next(f for f in findings if f["tag"] == "constant")
    assert constant["cols"] == ["const"] and constant["sev"] == "high"
    missing = next(
        f for f in findings if f["tag"] == "missing" and f["cols"] == ["income"]
    )
    assert missing["value"] == 10
    assert not missing["msg"].startswith("[")


def test_severity_override(df):
    report = ProfileReport(
        df, progress_bar=False, json_output={"severity": {"constant": "info"}}
    )
    findings = json.loads(report.to_json())["findings"]
    assert next(f for f in findings if f["tag"] == "constant")["sev"] == "info"


def test_no_charts_or_value_tables(report):
    s = report.to_json(detail="full")
    assert "<svg" not in s
    assert "value_counts" not in json.loads(s)["columns"]["city"]


def test_size_is_bounded(report):
    n_cols = 9
    standard = estimate_tokens(report.to_json())
    brief = estimate_tokens(report.to_json(detail="brief"))
    full = estimate_tokens(report.to_json(detail="full"))
    assert brief < standard < full
    assert standard / n_cols < 150
    assert full / n_cols < 400
    # Legacy dump of the same profile is far larger
    with pytest.warns(FutureWarning):
        assert len(report.to_json(legacy=True)) > 20 * len(report.to_json())


def test_size_does_not_grow_with_rows():
    rng = np.random.default_rng(0)

    def size(n):
        frame = pd.DataFrame(
            {
                "x": rng.normal(size=n),
                "id": np.arange(n),
                "t": pd.date_range("2020", periods=n, freq="min"),
            }
        )
        return len(ProfileReport(frame, minimal=True, progress_bar=False).to_json())

    assert size(20_000) < 1.2 * size(500)


@pytest.mark.parametrize("budget", [150, 250, 400, 800])
def test_token_budget(report, budget):
    payload = json.loads(report.to_json(max_tokens=budget))
    used = estimate_tokens(
        json.dumps(payload, separators=(",", ":"), ensure_ascii=False)
    )
    assert used <= budget or payload["omitted"].get("budget") == "exceeded"
    assert payload["dataset"]["rows"] == 300
    _validate(payload)


def test_token_budget_keeps_columns_with_findings(report):
    payload = json.loads(report.to_json(max_tokens=400))
    assert "income" in payload["columns"]
    dropped = payload["omitted"].get("columns", [])
    assert all(c not in payload["columns"] for c in dropped)


def test_custom_tokenizer(report):
    calls = []

    def tokenizer(text):
        calls.append(len(text))
        return len(text.split(","))

    report.to_json(max_tokens=50, tokenizer=tokenizer)
    assert calls


def test_include_values_false_drops_raw_values(report, df):
    s = report.to_json(include_values=False, detail="full")
    for value in ("Lisbon", "Porto", "good", "slow", "2020-01-01"):
        assert value not in s
    payload = json.loads(s)
    assert payload["omitted"]["values"] == "include_values=False"
    assert "stats" in payload["columns"]["age"]


def test_sensitive_mode_redacts(df):
    report = ProfileReport(df, sensitive=True, progress_bar=False)
    s = report.to_json(detail="full")
    assert "Lisbon" not in s and "good" not in s


def test_samples_opt_in(report):
    assert "sample" not in json.loads(report.to_json())
    payload = json.loads(report.to_json(include_samples=True))
    assert 0 < len(payload["sample"]) <= 5
    assert "sample" not in json.loads(
        report.to_json(include_samples=True, include_values=False)
    )


def test_columns_filter(report):
    payload = json.loads(report.to_json(columns=["income", "city"]))
    assert list(payload["columns"]) == ["income", "city"]
    assert all(
        not f["cols"] or set(f["cols"]) & {"income", "city"}
        for f in payload["findings"]
    )
    with pytest.raises(ValueError):
        report.to_json(columns=["nope"])


def test_correlation_pairs_are_labelled(report):
    corr = json.loads(report.to_json())["correlations"]
    assert corr["threshold"] == 0.5
    pairs = {tuple(sorted(p[:2])) for p in corr["pairs"]}
    assert ("income", "spend") in pairs
    assert all(abs(p[2]) >= 0.5 for p in corr["pairs"])


def test_roles_in_columns(report):
    cols = json.loads(report.to_json())["columns"]
    assert cols["id"]["role"] == "identifier"
    assert "stats" not in cols["id"] and cols["id"]["range"] == [0, 299]
    assert cols["city"]["role"] == "dimension" and cols["city"]["top"][0][1] > 0
    assert cols["review"]["role"] == "free_text" and cols["review"]["top_terms"]
    assert cols["const"]["role"] == "constant"
    assert cols["signup"]["role"] == "timestamp"


def test_floats_are_compact(report):
    s = report.to_json()
    assert not any(
        len(tok) > 12
        for tok in s.replace("[", ",").replace("]", ",").split(",")
        if tok.replace(".", "").replace("-", "").isdigit()
    )


def test_metadata_and_descriptions_flow_through(df):
    report = ProfileReport(
        df,
        progress_bar=False,
        dataset={"description": "Customer sample", "url": "https://example.com"},
        variables={
            "descriptions": {
                "income": "Yearly income in EUR",
                "city": {"description": "City of residence", "role": "dimension"},
            }
        },
    )
    payload = json.loads(report.to_json())
    assert payload["dataset"]["description"] == "Customer sample"
    assert payload["columns"]["income"]["desc"] == "Yearly income in EUR"
    assert payload["columns"]["city"]["desc"] == "City of residence"
    assert payload["columns"]["city"]["role_basis"] == ["user"]


def test_to_dict_and_cache(report):
    a = report.to_dict()
    a["dataset"]["rows"] = -1
    assert report.to_dict()["dataset"]["rows"] == 300
    assert report.to_dict(detail="brief") != report.to_dict(detail="full")
    assert report.json == report.to_json()


def test_markdown(report):
    md = report.to_markdown()
    assert md.startswith("# Data profile: fixture")
    for section in ("## Findings", "## Columns", "## Correlations"):
        assert section in md
    assert "| income | numeric | measure" in md


def test_describe_column(report):
    col = report.describe_column("income")
    assert col["column"] == "income" and col["schema"] == agent.SCHEMA_ID
    assert "histogram" in col and len(col["histogram"]["counts"]) <= 10
    assert {f["tag"] for f in col["findings"]} >= {"missing"}
    with pytest.raises(ValueError):
        report.describe_column("nope")


def test_legacy_flag(report):
    with pytest.warns(FutureWarning, match="deprecated"):
        legacy = json.loads(report.to_json(legacy=True))
    assert "variables" in legacy and "scatter" in legacy
    with pytest.raises(ValueError, match="cannot be combined"):
        report.to_json(legacy=True, detail="brief")


def test_legacy_from_config(df):
    report = ProfileReport(
        df, minimal=True, progress_bar=False, json_output={"legacy": True}
    )
    with pytest.warns(FutureWarning):
        assert "variables" in json.loads(report.to_json())
    assert "columns" in json.loads(report.to_json(legacy=False))


def test_to_file(report, tmp_path):
    report.to_file(tmp_path / "p.json")
    assert json.loads((tmp_path / "p.json").read_text())["schema"] == agent.SCHEMA_ID
    with pytest.warns(FutureWarning):
        report.to_file(tmp_path / "legacy.json", legacy=True)
    assert "variables" in json.loads((tmp_path / "legacy.json").read_text())
    report.to_file(tmp_path / "p.md")
    assert (tmp_path / "p.md").read_text().startswith("# Data profile")
    with pytest.raises(ValueError):
        report.to_file(tmp_path / "p.html", legacy=True)


def test_minimal_defaults_to_brief(df):
    report = ProfileReport(df, minimal=True, progress_bar=False)
    assert json.loads(report.to_json())["detail"] == "brief"


def test_comparison(df):
    left = ProfileReport(df.iloc[:150], title="Left", progress_bar=False)
    right_df = df.iloc[150:].copy()
    right_df.loc[right_df.index[:60], "age"] = np.nan
    right = ProfileReport(right_df, title="Right", progress_bar=False)
    payload = json.loads(left.compare(right).to_json())
    _validate(payload)
    assert payload["kind"] == "comparison" and payload["labels"] == ["Left", "Right"]
    delta = payload["delta"][0]
    assert delta["columns"]["age"]["missing_pct"][0] == 0
    assert delta["columns"]["age"]["missing_pct"][1] == 40
    assert any(
        f["tag"] == "missing" and f["cols"] == ["age"] for f in delta["findings"]["new"]
    )
    assert "Changes from Left to Right" in left.compare(right).to_markdown()


def test_timeseries_index():
    n = 120
    frame = pd.DataFrame(
        {"value": np.sin(np.arange(n) / 5)},
        index=pd.date_range("2021-01-01", periods=n, freq="D"),
    )
    payload = json.loads(
        ProfileReport(frame, tsmode=True, progress_bar=False).to_json()
    )
    _validate(payload)
    assert payload["dataset"]["time_index"]["start"].startswith("2021-01-01")


def test_console_legacy_flag(tmp_path, df):
    csv = tmp_path / "data.csv"
    df.drop(columns=["signup"]).head(50).to_csv(csv, index=False)
    out = tmp_path / "out.json"
    console.main(["-s", "--minimal", str(csv), str(out)])
    assert json.loads(out.read_text())["schema"] == agent.SCHEMA_ID
    with pytest.warns(FutureWarning):
        console.main(["-s", "--minimal", "--legacy-json", str(csv), str(out)])
    assert "variables" in json.loads(out.read_text())


# --- Regressions found in review -------------------------------------------


def test_redacted_columns_not_in_sample(df):
    report = ProfileReport(
        df, progress_bar=False, vars={"cat": {"redact": True}, "text": {"redact": True}}
    )
    sample = report.to_dict(include_samples=True)["sample"]
    assert all("city" not in row and "review" not in row for row in sample)
    assert "Lisbon" not in json.dumps(sample)


def test_include_values_false_drops_identifier_ranges_and_time_bounds():
    frame = pd.DataFrame(
        {
            "id": np.arange(900000, 900200),
            "x": np.random.default_rng(0).normal(size=200),
        }
    )
    cols = json.loads(
        ProfileReport(frame, progress_bar=False).to_json(include_values=False)
    )["columns"]
    assert "range" not in cols["id"]
    ts = pd.DataFrame(
        {"v": np.arange(60.0)}, index=pd.date_range("2021-03-01", periods=60, freq="D")
    )
    s = ProfileReport(ts, tsmode=True, progress_bar=False).to_json(include_values=False)
    assert "2021-03" not in s


def test_large_integers_stay_exact():
    frame = pd.DataFrame({"id": np.arange(10**17, 10**17 + 100)})
    col = json.loads(ProfileReport(frame, progress_bar=False).to_json())["columns"][
        "id"
    ]
    assert col["range"] == [10**17, 10**17 + 99]


def test_comparison_respects_budget_and_validates_roles(df):
    left = ProfileReport(df.iloc[:150], title="L", progress_bar=False)
    right = ProfileReport(df.iloc[150:], title="R", progress_bar=False)
    comparison = left.compare(right)
    full = comparison.to_json()
    small = comparison.to_json(max_tokens=200)
    assert len(small) < len(full)
    bad = ProfileReport(
        df.iloc[:150], progress_bar=False, variables={"roles": {"age": "metric"}}
    )
    with pytest.raises(ValueError):
        bad.compare(ProfileReport(df.iloc[150:], progress_bar=False)).to_json()


def test_cache_tracks_config_and_tokenizer(df):
    report = ProfileReport(df, progress_bar=False)
    assert len(report.to_dict()["columns"]["city"]["top"]) == 3
    report.config.json_output.top_k = 1
    assert len(report.to_dict()["columns"]["city"]["top"]) == 1
    a = report.to_json(max_tokens=300, tokenizer=lambda s: len(s))
    b = report.to_json(max_tokens=300, tokenizer=lambda s: len(s) // 100)
    assert len(b) > len(a)


def test_roles_and_comparison_survive_dumps(df):
    frame = df.copy()
    frame["acct"] = [f"{i:05d}" for i in range(10000, 10000 + len(frame))]
    report = ProfileReport(frame, progress_bar=False)
    before = report.to_dict()
    loaded = ProfileReport(None, progress_bar=False).loads(report.dumps())
    assert loaded.get_roles()["acct"] == report.get_roles()["acct"]
    assert loaded.to_dict() == before

    comparison = ProfileReport(df.iloc[:150], progress_bar=False).compare(
        ProfileReport(df.iloc[150:], progress_bar=False)
    )
    reloaded = ProfileReport(None, progress_bar=False).loads(comparison.dumps())
    assert reloaded.to_dict()["kind"] == "comparison"


def test_delta_ignores_findings_on_removed_columns():
    from data_profiling.model.agent.comparison import build_delta

    a = {
        "dataset": {"rows": 1},
        "columns": {"x": {}, "gone": {}},
        "findings": [{"tag": "unique", "cols": ["gone"], "sev": "info", "msg": ""}],
    }
    b = {"dataset": {"rows": 1}, "columns": {"x": {}}, "findings": []}
    delta = build_delta(a, b, "A", "B")
    assert delta["columns_removed"] == ["gone"]
    assert delta["findings"]["resolved"] == []


def test_messages_with_brackets_in_column_names():
    frame = pd.DataFrame({"weight [kg]": [1] * 20, "y": range(20)})
    findings = json.loads(ProfileReport(frame, progress_bar=False).to_json())[
        "findings"
    ]
    constant = next(f for f in findings if f["tag"] == "constant")
    assert constant["msg"] == "has a constant value"


def test_spark_numeric_dtypes_are_numeric():
    from data_profiling.model.roles import _is_numeric_dtype

    for t in ("bigint", "smallint", "tinyint", "decimal(10,2)", "int", "double"):
        assert _is_numeric_dtype(t)
    assert not _is_numeric_dtype("string")
