"""Tests for semantic column roles."""

import uuid

import numpy as np
import pandas as pd
import pytest

from data_profiling import ProfileReport
from data_profiling.model.roles import ROLES, infer_roles, source_dtypes_of


@pytest.fixture(scope="module")
def tricky():
    rng = np.random.default_rng(1)
    n = 600
    return pd.DataFrame(
        {
            "customer_id": rng.permutation(np.arange(100000, 100000 + n)),
            "row_id": np.arange(n),
            "order_uuid": [
                str(uuid.UUID(int=int(rng.integers(0, 2**63)))) for _ in range(n)
            ],
            "zip_str": rng.integers(1000, 9999, n).astype(str),
            "zip_int": rng.integers(10000, 99999, n),
            "category_id": rng.integers(1, 21, n),
            "year": rng.integers(2018, 2025, n),
            "is_churned": rng.choice([0, 1], n),
            "price": rng.lognormal(3, 1, n).round(2),
            "quantity": rng.poisson(3, n),
            "rating": rng.integers(1, 6, n),
            "country": rng.choice(["PT", "ES", "FR", "DE", "US"], n),
            "email": [f"user{i}@example.com" for i in rng.permutation(n)],
            "review": [
                " ".join(
                    rng.choice(
                        ["good", "bad", "slow", "late", "love"],
                        int(rng.integers(3, 20)),
                    )
                )
                for _ in range(n)
            ],
            "created_at": pd.date_range("2023-01-01", periods=n, freq="37min"),
            "source": "web",
            "notes": [np.nan] * n,
        }
    )


EXPECTED = {
    "customer_id": "identifier",
    "row_id": "identifier",
    "order_uuid": "identifier",
    "zip_str": "code",
    "zip_int": "code",
    "category_id": "dimension",
    "year": "unknown",
    "is_churned": "dimension",
    "price": "measure",
    "quantity": "measure",
    "rating": "unknown",
    "country": "dimension",
    "email": "identifier",
    "review": "free_text",
    "created_at": "timestamp",
    "source": "constant",
    "notes": "empty",
}


@pytest.mark.parametrize("minimal", [True, False])
def test_inferred_roles(tricky, minimal):
    report = ProfileReport(tricky, minimal=minimal, progress_bar=False)
    roles = report.get_roles()
    assert {c: r["role"] for c, r in roles.items()} == EXPECTED
    assert roles["year"]["role_candidates"] == ["dimension", "measure"]
    assert "year_range" in roles["year"]["role_basis"]
    assert roles["customer_id"]["role_basis"][:3] == ["unique", "integer", "unordered"]
    assert all(r["role"] in ROLES for r in roles.values())


def test_unresolved(tricky):
    report = ProfileReport(tricky, minimal=True, progress_bar=False)
    assert set(report.get_roles(unresolved=True)) == {"year", "rating"}


def test_without_name_hints_falls_back_honestly():
    rng = np.random.default_rng(3)
    frame = pd.DataFrame(
        {"a": rng.integers(10000, 99999, 500), "b": rng.normal(size=500)}
    )
    roles = ProfileReport(frame, minimal=True, progress_bar=False).get_roles()
    assert roles["a"] == {"role": "measure", "role_basis": ["default"]}
    assert roles["b"]["role_basis"] == ["default"]


def test_config_overrides(tricky):
    report = ProfileReport(
        tricky,
        minimal=True,
        progress_bar=False,
        variables={
            "roles": {"year": "dimension"},
            "descriptions": {"rating": {"description": "1-5 stars", "role": "measure"}},
        },
    )
    roles = report.get_roles()
    assert roles["year"] == {"role": "dimension", "role_basis": ["user"]}
    assert roles["rating"] == {"role": "measure", "role_basis": ["user"]}
    assert report.get_roles(unresolved=True) == {}


def test_roles_setting_wins_over_description(tricky):
    report = ProfileReport(
        tricky,
        minimal=True,
        progress_bar=False,
        variables={
            "roles": {"rating": "dimension"},
            "descriptions": {"rating": {"role": "measure"}},
        },
    )
    assert report.get_roles()["rating"]["role"] == "dimension"


def test_set_roles_does_not_recompute(tricky):
    report = ProfileReport(tricky, minimal=True, progress_bar=False)
    description = report.get_description()
    html_before = report.to_html()
    report.set_roles({"year": "dimension", "rating": "measure"})
    assert report.get_description() is description
    assert report.get_roles()["year"]["role_basis"] == ["user"]
    assert report.to_dict()["columns"]["year"]["role"] == "dimension"
    assert report.to_html() != html_before


def test_set_roles_validation(tricky):
    report = ProfileReport(tricky, minimal=True, progress_bar=False)
    with pytest.raises(ValueError, match="Valid roles"):
        report.set_roles({"year": "category"})
    with pytest.warns(UserWarning, match="not in the profile"):
        report.set_roles({"not_a_column": "dimension"})
    with pytest.warns(UserWarning, match="unusual"):
        report.set_roles({"review": "measure"})


def test_invalid_role_in_config(tricky):
    report = ProfileReport(
        tricky, minimal=True, progress_bar=False, variables={"roles": {"year": "bogus"}}
    )
    with pytest.raises(ValueError):
        report.get_roles()


def test_roles_survive_dump_and_load(tricky):
    report = ProfileReport(tricky, minimal=True, progress_bar=False)
    report.set_roles({"year": "dimension"})
    data = report.dumps()
    loaded = ProfileReport(tricky, minimal=True, progress_bar=False).loads(data)
    assert loaded.get_roles()["year"]["role_basis"] == ["user"]


def test_roles_apply_to_comparison(tricky):
    left = ProfileReport(
        tricky.iloc[:300],
        minimal=True,
        progress_bar=False,
        variables={"roles": {"year": "dimension"}},
    )
    right = ProfileReport(tricky.iloc[300:], minimal=True, progress_bar=False)
    payload = left.compare(right).to_dict()
    assert all(r["columns"]["year"]["role"] == "dimension" for r in payload["reports"])


def test_html_role_badges_and_card(tricky):
    report = ProfileReport(
        tricky,
        minimal=True,
        progress_bar=False,
        variables={
            "descriptions": {"price": {"description": "Unit price", "role": "measure"}}
        },
    )
    html = report.to_html()
    assert html.count("dp-role dp-role-") == len(EXPECTED)
    assert "dp-role-unknown" in html and "dp-role-user" in html
    assert "dp-ov-column_roles" in html
    assert "Unit price" in html
    assert (
        "{&#34;year&#34;: &#34;dimension&#34;" in html or '{"year": "dimension"' in html
    )


def test_html_escapes_column_names():
    frame = pd.DataFrame({"<b>x</b>": [1, 2, 3, 4, 5] * 4, "y": list(range(20))})
    report = ProfileReport(frame, minimal=True, progress_bar=False)
    report.set_roles({"<b>x</b>": "dimension"})
    html = report.to_html()
    assert "<td><b>x</b>" not in html
    assert "<b>x</b>" not in report.to_markdown().split("## Columns")[0]


def test_source_dtypes():
    frame = pd.DataFrame({"a": ["001", "002"], "b": [1, 2]})
    dtypes = source_dtypes_of(frame)
    assert dtypes["b"] == "int64"
    assert dtypes["a"] in ("object", "str", "string")
    assert source_dtypes_of(None) == {}


def test_comparison_summaries_are_unknown():
    roles = infer_roles({"a": {"type": ["Numeric", "Numeric"]}})
    assert roles["a"].role == "unknown" and roles["a"].basis == ["comparison"]
