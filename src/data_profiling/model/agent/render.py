"""Render an agent payload as compact JSON or as Markdown for prompt injection."""

import json
from typing import Any, Dict, List


def render_json(payload: Dict[str, Any]) -> str:
    return json.dumps(payload, separators=(",", ":"), ensure_ascii=False)


def _cell(value: Any) -> str:
    s = "" if value is None else str(value)
    return s.replace("|", "\\|").replace("\n", " ")


def _fmt_pairs(pairs: List[List[Any]]) -> str:
    return ", ".join(f"{_cell(v)} {p}%" for v, p in pairs)


def _summary(col: Dict[str, Any]) -> str:
    parts: List[str] = []
    stats = col.get("stats")
    if stats:
        order = ("min", "p5", "p25", "p50", "p75", "p95", "max", "mean", "std", "skew")
        parts.append(" · ".join(f"{k} {stats[k]}" for k in order if k in stats))
    if "range" in col:
        parts.append(f"range {col['range'][0]} … {col['range'][1]}")
    if "top" in col:
        other = f", other {col['other_pct']}%" if "other_pct" in col else ""
        parts.append("top: " + _fmt_pairs(col["top"]) + other)
    if "len" in col:
        ln = col["len"]
        parts.append(f"len {ln.get('min')}/{ln.get('p50')}/{ln.get('max')}")
    if "top_terms" in col:
        parts.append("terms: " + ", ".join(col["top_terms"]))
    if "value" in col:
        parts.append(f"value {col['value']}")
    for key, label in (
        ("zeros_pct", "zeros"),
        ("negative_pct", "negative"),
        ("infinite_pct", "inf"),
    ):
        if key in col:
            parts.append(f"{label} {col[key]}%")
    if col.get("desc"):
        parts.append(f"“{col['desc']}”")
    return "; ".join(parts)


def _role(col: Dict[str, Any]) -> str:
    role = col.get("role", "")
    if role == "unknown" and col.get("role_candidates"):
        return "unknown (" + " or ".join(col["role_candidates"]) + ")"
    if col.get("role_basis") == ["default"]:
        return f"{role} (default)"
    if col.get("role_basis") == ["user"]:
        return f"{role} (user)"
    return role


def _profile_markdown(p: Dict[str, Any], heading: str = "#") -> List[str]:
    ds = p.get("dataset", {})
    lines = [f"{heading} Data profile: {ds.get('title', 'dataset')}", ""]
    head = [
        f"{ds.get('rows')} rows",
        f"{ds.get('columns')} columns",
        f"missing cells {ds.get('missing_cells_pct')}%",
    ]
    if "duplicate_rows_pct" in ds:
        head.append(f"duplicate rows {ds['duplicate_rows_pct']}%")
    lines.append(" · ".join(head))
    if ds.get("types"):
        lines.append("Types: " + ", ".join(f"{k} {v}" for k, v in ds["types"].items()))
    if ds.get("roles"):
        lines.append("Roles: " + ", ".join(f"{k} {v}" for k, v in ds["roles"].items()))
    for key in ("description", "creator", "author", "url"):
        if ds.get(key):
            lines.append(f"{key.capitalize()}: {ds[key]}")
    if ds.get("time_index"):
        ti = ds["time_index"]
        lines.append(
            f"Time index: {ti.get('start')} … {ti.get('end')}, frequency {ti.get('frequency')}, {ti.get('n_series')} series"
        )
    lines.append("")

    findings = p.get("findings", [])
    lines += [f"{heading}# Findings", ""]
    if findings:
        lines += ["| sev | tag | columns | detail |", "|---|---|---|---|"]
        for f in findings:
            lines.append(
                f"| {f['sev']} | {f['tag']} | {_cell(', '.join(f.get('cols', [])))} | {_cell(f.get('msg'))} |"
            )
    else:
        lines.append("None.")
    lines.append("")

    cols = p.get("columns", {})
    lines += [
        f"{heading}# Columns",
        "",
        "| column | type | role | missing % | distinct | tags | summary |",
        "|---|---|---|---|---|---|---|",
    ]
    for name, c in cols.items():
        lines.append(
            "| "
            + " | ".join(
                [
                    _cell(name),
                    c.get("type", ""),
                    _cell(_role(c)),
                    _cell(c.get("missing_pct", 0)),
                    _cell(c.get("distinct", "")),
                    _cell(", ".join(c.get("tags", []))),
                    _cell(_summary(c)),
                ]
            )
            + " |"
        )
    lines.append("")

    corr = p.get("correlations")
    if corr and corr.get("pairs"):
        lines += [f"{heading}# Correlations (|r| ≥ {corr.get('threshold')})", ""]
        for pair in corr["pairs"]:
            method = (
                f" ({pair[3]})"
                if len(pair) > 3
                else f" ({corr['methods'][0]})" if corr.get("methods") else ""
            )
            lines.append(f"- {pair[0]} ~ {pair[1]}: {pair[2]}{method}")
        lines.append("")

    if p.get("sample"):
        rows = p["sample"]
        keys = list(rows[0].keys())
        lines += [
            f"{heading}# Sample rows",
            "",
            "| " + " | ".join(_cell(k) for k in keys) + " |",
            "|" + "---|" * len(keys),
        ]
        for r in rows:
            lines.append("| " + " | ".join(_cell(r.get(k)) for k in keys) + " |")
        lines.append("")

    if p.get("omitted"):
        lines.append(
            "Omitted: " + "; ".join(f"{k}: {v}" for k, v in p["omitted"].items())
        )
    return lines


def render_markdown(payload: Dict[str, Any]) -> str:
    if payload.get("kind") == "comparison":
        lines = [
            f"# Data profile comparison: {' vs '.join(payload.get('labels', []))}",
            "",
        ]
        for delta in payload.get("delta", []):
            lines += [f"## Changes from {delta['from']} to {delta['to']}", ""]
            changes: List[str] = []
            if "rows" in delta:
                changes.append(f"- rows: {delta['rows'][0]} → {delta['rows'][1]}")
            for key, label in (
                ("columns_added", "added"),
                ("columns_removed", "removed"),
            ):
                if delta.get(key):
                    changes.append(f"- columns {label}: {', '.join(delta[key])}")
            for f in delta.get("findings", {}).get("new", []):
                changes.append(
                    f"- new {f['sev']} finding: {f['tag']} on {', '.join(f.get('cols', []))} ({f.get('msg')})"
                )
            for f in delta.get("findings", {}).get("resolved", []):
                changes.append(
                    f"- resolved: {f['tag']} on {', '.join(f.get('cols', []))}"
                )
            for col, change in delta.get("columns", {}).items():
                changes.append(
                    f"- {col}: "
                    + "; ".join(f"{k} {v[0]} → {v[1]}" for k, v in change.items())
                )
            lines += changes or ["No changes above thresholds."]
            lines.append("")
        for report in payload.get("reports", []):
            lines += _profile_markdown(report, heading="##")
            lines.append("")
        return "\n".join(lines).rstrip() + "\n"
    return "\n".join(_profile_markdown(payload)).rstrip() + "\n"
