"""Agent output for comparison reports: one agent/1 payload per report plus deltas."""

from typing import Any, Dict, List, Mapping, Sequence, Tuple

from data_profiling.model.agent.builder import AgentOptions, build_profile, num
from data_profiling.model.agent.vocab import SCHEMA_ID
from data_profiling.model.description import BaseDescription
from data_profiling.model.roles import ColumnRole

#: Minimum changes reported in a delta.
MISSING_PCT_POINTS = 1.0
DISTINCT_RELATIVE = 0.10
MEAN_SHIFT_STD = 0.10


def _column_delta(a: Mapping[str, Any], b: Mapping[str, Any]) -> Dict[str, List[Any]]:
    change: Dict[str, List[Any]] = {}
    for key in ("type", "role"):
        if a.get(key) != b.get(key):
            change[key] = [a.get(key), b.get(key)]
    ma, mb = a.get("missing_pct", 0) or 0, b.get("missing_pct", 0) or 0
    if abs(mb - ma) >= MISSING_PCT_POINTS:
        change["missing_pct"] = [ma, mb]
    da, db = a.get("distinct"), b.get("distinct")
    if isinstance(da, (int, float)) and isinstance(db, (int, float)) and da:
        if abs(db - da) / da >= DISTINCT_RELATIVE:
            change["distinct"] = [da, db]
    sa, sb = a.get("stats") or {}, b.get("stats") or {}
    if (
        "mean" in sa
        and "mean" in sb
        and isinstance(sa["mean"], (int, float))
        and isinstance(sb["mean"], (int, float))
    ):
        scale = sa.get("std") or abs(sa["mean"]) or 1
        if (
            isinstance(scale, (int, float))
            and abs(sb["mean"] - sa["mean"]) / scale >= MEAN_SHIFT_STD
        ):
            change["mean"] = [sa["mean"], sb["mean"]]
    return change


def _finding_key(f: Mapping[str, Any]) -> Tuple[str, Tuple[str, ...]]:
    return f["tag"], tuple(f.get("cols", []))


def build_delta(
    a: Dict[str, Any], b: Dict[str, Any], label_a: str, label_b: str
) -> Dict[str, Any]:
    cols_a, cols_b = a.get("columns", {}), b.get("columns", {})
    delta: Dict[str, Any] = {"from": label_a, "to": label_b, "columns": {}}
    for col in cols_a:
        if col in cols_b:
            change = _column_delta(cols_a[col], cols_b[col])
            if change:
                delta["columns"][col] = change
    only_a = [c for c in cols_a if c not in cols_b]
    only_b = [c for c in cols_b if c not in cols_a]
    if only_a:
        delta["columns_removed"] = only_a
    if only_b:
        delta["columns_added"] = only_b
    changed_cols = set(only_a) | set(only_b)

    def _kept(f: Mapping[str, Any]) -> bool:
        return not changed_cols.intersection(f.get("cols", []))

    fa = {_finding_key(f): f for f in a.get("findings", []) if _kept(f)}
    fb = {_finding_key(f): f for f in b.get("findings", []) if _kept(f)}
    delta["findings"] = {
        "new": [fb[k] for k in fb if k not in fa],
        "resolved": [fa[k] for k in fa if k not in fb],
    }
    ra, rb = a["dataset"].get("rows"), b["dataset"].get("rows")
    if ra != rb:
        delta["rows"] = [num(ra), num(rb)]
    return delta


def build_comparison(
    parts: Sequence[Tuple[str, BaseDescription, Mapping[str, ColumnRole]]],
    config: Any,
    opts: AgentOptions,
) -> Dict[str, Any]:
    reports = [build_profile(desc, config, roles, opts) for _, desc, roles in parts]
    labels = [label for label, _, _ in parts]
    for label, report in zip(labels, reports):
        report["dataset"]["label"] = label
        report.pop("schema", None)
        report.pop("detail", None)
    return {
        "schema": SCHEMA_ID,
        "kind": "comparison",
        "detail": opts.detail,
        "labels": labels,
        "delta": [
            build_delta(reports[0], reports[i], labels[0], labels[i])
            for i in range(1, len(reports))
        ],
        "reports": reports,
    }
