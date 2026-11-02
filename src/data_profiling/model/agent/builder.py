"""Build the agent-oriented profile (``data-profiling.agent/1``) from a description.

The builder reads the same ``BaseDescription`` that drives the HTML report, so it
works for every backend. Output size scales with the number of columns, not
rows: every list is capped and no chart or raw value-count table is included.
"""

import html
import math
import re
from dataclasses import dataclass
from datetime import date, datetime
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from data_profiling.model.agent.vocab import (
    DETAIL_LEVELS,
    EXCLUDED_ALERTS,
    SCHEMA_ID,
    SEVERITIES,
    normalize_type,
    severity_of,
)
from data_profiling.model.alerts import Alert, AlertType
from data_profiling.model.description import BaseDescription
from data_profiling.model.roles import ColumnRole, column_description

_SEV_ORDER = {s: i for i, s in enumerate(SEVERITIES)}
_TAG_ORDER = {t: i for i, t in enumerate(AlertType)}
_HTML_TAG = re.compile(r"<[^>]+>")
_MAX_STRING = 80
_SAMPLE_ROWS = 5
_FULL_BINS = 10


@dataclass
class AgentOptions:
    detail: str = "standard"
    top_k: int = 5
    top_terms: int = 8
    corr_threshold: float = 0.5
    max_correlation_pairs: int = 50
    float_precision: int = 3
    include_samples: bool = False
    include_values: bool = True
    columns: Optional[Sequence[str]] = None
    severity: Optional[Mapping[str, str]] = None

    def validate(self) -> None:
        if self.detail not in DETAIL_LEVELS:
            raise ValueError(
                f"Invalid detail {self.detail!r}. Valid: {', '.join(DETAIL_LEVELS)}."
            )


# --------------------------------------------------------------------------- #
# Value encoding
# --------------------------------------------------------------------------- #


def num(value: Any, precision: int = 3) -> Any:
    """Compact number: integers stay exact, floats keep `precision` significant digits."""
    if value is None:
        return None
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, np.integer)):
        return int(value)
    try:
        x = float(value)
    except (TypeError, ValueError):
        return text(value)
    if math.isnan(x):
        return None
    if math.isinf(x):
        return "inf" if x > 0 else "-inf"
    if x.is_integer() and abs(x) < 1e15:
        return int(x)
    rounded = float(f"{x:.{precision}g}")
    return int(rounded) if rounded.is_integer() and abs(rounded) < 1e15 else rounded


def pct(ratio: Any, precision: int = 3) -> Any:
    """A ratio in [0, 1] as a percentage in [0, 100]."""
    if ratio is None:
        return None
    try:
        return num(float(ratio) * 100, precision)
    except (TypeError, ValueError):
        return None


def text(value: Any, limit: int = _MAX_STRING) -> str:
    if isinstance(value, (datetime, date, pd.Timestamp)):
        s = value.isoformat()
    else:
        s = str(value)
    s = html.unescape(_HTML_TAG.sub("", s))
    return s if len(s) <= limit else s[: limit - 1] + "…"


def _plain(value: Any) -> Any:
    """Make a scalar JSON-safe without changing its meaning."""
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return num(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, (str, int, bool)) or value is None:
        return value
    return text(value)


# --------------------------------------------------------------------------- #
# Findings
# --------------------------------------------------------------------------- #

_FINDING_VALUE = {
    AlertType.MISSING: ("p_missing", True),
    AlertType.ZEROS: ("p_zeros", True),
    AlertType.INFINITE: ("p_infinite", True),
    AlertType.DUPLICATES: ("p_duplicates", True),
    AlertType.NEAR_DUPLICATES: ("p_near_dups", True),
    AlertType.DIRTY_CATEGORY: ("p_fuzzy_vals", True),
    AlertType.HIGH_CARDINALITY: ("n_distinct", False),
    AlertType.SKEWED: ("skewness", False),
    AlertType.IMBALANCE: ("imbalance", False),
}


def _alert_message(alert: Alert) -> str:
    try:
        msg = alert._get_description()
    except Exception:  # pragma: no cover - a broken message must not break export
        msg = alert.alert_type.name.lower()
    if msg.startswith(f"[{alert.alert_type.name}] alert on column"):
        msg = alert.alert_type_name.lower()
    prefix = f"[{alert.column_name}]"
    if alert.column_name is not None and msg.startswith(prefix):
        msg = msg[len(prefix) :].lstrip()
    # Round long floats inside messages, e.g. skewness 3.8612345 -> 3.86
    msg = re.sub(r"-?\d+\.\d{4,}", lambda m: str(num(float(m.group()))), msg)
    return msg


def _finding(alert: Alert, opts: AgentOptions) -> Dict[str, Any]:
    cols: List[str] = [alert.column_name] if alert.column_name is not None else []
    values = alert.values or {}
    if alert.alert_type == AlertType.HIGH_CORRELATION and "fields" in values:
        cols += [str(f) for f in values["fields"] if f not in cols]
    out: Dict[str, Any] = {
        "sev": severity_of(alert.alert_type, opts.severity),
        "tag": alert.alert_type.name.lower(),
        "cols": cols,
        "msg": _alert_message(alert),
    }
    key = _FINDING_VALUE.get(alert.alert_type)
    if key is not None and key[0] in values:
        value = values[key[0]]
        out["value"] = pct(value) if key[1] else num(value)
    return out


def build_findings(
    alerts: Sequence[Alert], opts: AgentOptions, columns: Optional[set] = None
) -> List[Dict[str, Any]]:
    selected = [
        a
        for a in alerts
        if a.alert_type not in EXCLUDED_ALERTS
        and (columns is None or a.column_name is None or a.column_name in columns)
    ]
    selected.sort(
        key=lambda a: (
            _SEV_ORDER[severity_of(a.alert_type, opts.severity)],
            _TAG_ORDER.get(a.alert_type, 99),
            str(a.column_name),
        )
    )
    return [_finding(a, opts) for a in selected]


def _column_tags(alerts: Sequence[Alert]) -> Dict[str, List[str]]:
    tags: Dict[str, List[str]] = {}
    for a in alerts:
        if a.alert_type in EXCLUDED_ALERTS or a.column_name is None:
            continue
        tag = a.alert_type.name.lower()
        if tag not in tags.setdefault(a.column_name, []):
            tags[a.column_name].append(tag)
    return tags


# --------------------------------------------------------------------------- #
# Columns
# --------------------------------------------------------------------------- #


def _top_values(
    summary: Mapping[str, Any], k: int, count: int
) -> Tuple[List[List[Any]], Optional[Any]]:
    vc = summary.get("value_counts_without_nan")
    if vc is None or len(vc) == 0 or not count:
        return [], None
    head = vc.sort_values(ascending=False).head(k)
    top = [
        [_plain(idx) if not isinstance(idx, str) else text(idx), pct(c / count)]
        for idx, c in head.items()
    ]
    other = None
    if len(vc) > k:
        other = pct((count - float(head.sum())) / count)
    return top, other


def _quantile_stats(summary: Mapping[str, Any], full: bool) -> Dict[str, Any]:
    keys: List[Tuple[str, str]] = [("min", "min")]
    for q in ("5%", "25%", "50%", "75%", "95%"):
        if q in summary:
            keys.append((q, "p" + q[:-1]))
    keys += [("max", "max"), ("mean", "mean"), ("std", "std"), ("skewness", "skew")]
    if full:
        keys += [("kurtosis", "kurtosis"), ("mad", "mad"), ("cv", "cv")]
    out = {}
    for src, dst in keys:
        if src in summary:
            v = num(summary[src])
            if v is not None:
                out[dst] = v
    return out


def _coarse_histogram(
    summary: Mapping[str, Any], bins: int = _FULL_BINS
) -> Optional[Dict[str, Any]]:
    hist = summary.get("histogram")
    if hist is None:
        return None
    if isinstance(hist, list):  # merged comparison or multi-histogram output
        hist = hist[0] if hist else None
    if not (isinstance(hist, tuple) and len(hist) == 2):
        return None
    counts, edges = np.asarray(hist[0]), np.asarray(hist[1])
    if len(counts) == 0 or len(edges) != len(counts) + 1:
        return None
    if len(counts) > bins:
        groups = np.array_split(np.arange(len(counts)), bins)
        new_counts = [int(counts[g].sum()) for g in groups]
        new_edges = [edges[g[0]] for g in groups] + [edges[-1]]
        counts, edges = np.asarray(new_counts), np.asarray(new_edges)
    if np.issubdtype(edges.dtype, np.datetime64) or edges.dtype == object:
        edge_values = [text(e) for e in edges]
    else:
        edge_values = [num(e) for e in edges]
    return {"edges": edge_values, "counts": [int(c) for c in counts]}


def _role_fields(role: ColumnRole, detail: str) -> Dict[str, Any]:
    out: Dict[str, Any] = {"role": role.role}
    if detail != "brief" or role.needs_review:
        out["role_basis"] = list(role.basis)
    if role.candidates:
        out["role_candidates"] = list(role.candidates)
    return out


def build_column(
    name: str,
    summary: Mapping[str, Any],
    role: ColumnRole,
    tags: Sequence[str],
    opts: AgentOptions,
    description: str = "",
    redact: bool = False,
) -> Dict[str, Any]:
    detail = opts.detail
    full = detail == "full"
    vtype = normalize_type(summary.get("type"))
    count = int(summary.get("count", 0) or 0)
    include_values = opts.include_values and not redact

    col: Dict[str, Any] = {"type": vtype}
    col.update(_role_fields(role, detail))
    if description:
        col["desc"] = text(description, 200)
    missing = pct(summary.get("p_missing"))
    if missing:
        col["missing_pct"] = missing
    if "n_distinct" in summary:
        col["distinct"] = num(summary["n_distinct"])
    if tags:
        col["tags"] = list(tags)

    if detail == "brief":
        return col

    r = role.role
    if r in ("constant",):
        if include_values:
            top, _ = _top_values(summary, 1, count)
            if top:
                col["value"] = top[0][0]
        return col
    if r == "empty":
        return col

    if vtype == "numeric" or vtype == "timeseries":
        if r == "identifier":
            if include_values:
                col["range"] = [num(summary.get("min")), num(summary.get("max"))]
            mono = summary.get("monotonic")
            if mono:
                col["monotonic"] = {
                    2: "increasing_strict",
                    1: "increasing",
                    -1: "decreasing",
                    -2: "decreasing_strict",
                }.get(int(mono), "yes")
        elif r in ("code", "dimension") or (
            r == "unknown"
            and role.candidates
            and "dimension" in role.candidates
            and (summary.get("n_distinct") or 0) <= opts.top_k * 4
        ):
            if include_values:
                top, other = _top_values(summary, opts.top_k, count)
                if top:
                    col["top"] = top
                if other:
                    col["other_pct"] = other
            if r != "code":
                col["stats"] = _quantile_stats(summary, full)
        else:
            col["stats"] = _quantile_stats(summary, full)
            for key, dst in (
                ("p_zeros", "zeros_pct"),
                ("p_negative", "negative_pct"),
                ("p_infinite", "infinite_pct"),
            ):
                v = pct(summary.get(key))
                if v:
                    col[dst] = v
        if full:
            hist = _coarse_histogram(summary)
            if hist and r not in ("identifier", "code"):
                col["histogram"] = hist
        return col

    if vtype == "datetime":
        if include_values:
            col["range"] = [text(summary.get("min")), text(summary.get("max"))]
        if full:
            hist = _coarse_histogram(summary)
            if hist and include_values:
                col["histogram"] = hist
            v = pct(summary.get("p_invalid_dates"))
            if v:
                col["invalid_pct"] = v
        return col

    if vtype in (
        "categorical",
        "boolean",
        "text",
        "url",
        "path",
        "file",
        "image",
        "unsupported",
    ):
        if r in ("dimension", "code") or vtype in ("categorical", "boolean"):
            if include_values:
                top, other = _top_values(summary, opts.top_k, count)
                if top:
                    col["top"] = top
                if other:
                    col["other_pct"] = other
        if "max_length" in summary and r != "dimension":
            col["len"] = {
                "min": num(summary.get("min_length")),
                "p50": num(summary.get("median_length")),
                "max": num(summary.get("max_length")),
            }
        if r == "free_text" and include_values:
            wc = summary.get("word_counts")
            if wc is not None and len(wc) > 0:
                col["top_terms"] = [
                    text(w, 40)
                    for w in list(
                        wc.sort_values(ascending=False).index[: opts.top_terms]
                    )
                ]
        if full:
            for key, dst in (("imbalance", "imbalance"),):
                if key in summary:
                    col[dst] = num(summary[key])
            chi = summary.get("chi_squared")
            if isinstance(chi, Mapping) and "pvalue" in chi:
                col["chi_squared_p"] = num(chi["pvalue"])
            if "n_characters_distinct" in summary:
                col["chars_distinct"] = num(summary["n_characters_distinct"])
            if "script_counts" in summary and include_values:
                sc = summary["script_counts"]
                try:
                    col["scripts"] = [text(s, 20) for s in list(sc.index[:3])]
                except AttributeError:
                    pass
    return col


# --------------------------------------------------------------------------- #
# Dataset, correlations, samples
# --------------------------------------------------------------------------- #


def build_dataset(
    description: BaseDescription,
    config: Any,
    roles: Mapping[str, ColumnRole],
    include_values: bool = True,
) -> Dict[str, Any]:
    table = description.table
    ds: Dict[str, Any] = {}
    title = getattr(description.analysis, "title", None)
    if title:
        ds["title"] = text(title, 120)
    ds["rows"] = num(table.get("n"))
    ds["columns"] = num(table.get("n_var"))
    v = pct(table.get("p_cells_missing"))
    ds["missing_cells_pct"] = v if v is not None else 0
    if "p_duplicates" in table:
        ds["duplicate_rows_pct"] = pct(table.get("p_duplicates"))
    types = table.get("types") or {}
    ds["types"] = {normalize_type(k): int(c) for k, c in types.items()}
    role_counts: Dict[str, int] = {}
    for r in roles.values():
        role_counts[r.role] = role_counts.get(r.role, 0) + 1
    if role_counts:
        ds["roles"] = role_counts
    meta = config.dataset
    for key in ("description", "creator", "author", "url"):
        value = getattr(meta, key, "")
        if value:
            ds[key] = text(value, 300)
    tia = description.time_index_analysis
    if tia is not None:
        ds["time_index"] = {
            "n_series": num(tia.n_series),
            "length": num(tia.length),
            "frequency": text(tia.frequency) if tia.frequency else None,
        }
        if include_values:
            ds["time_index"]["start"] = text(tia.start)
            ds["time_index"]["end"] = text(tia.end)
    return ds


def build_correlations(
    description: BaseDescription, opts: AgentOptions, columns: Optional[set] = None
) -> Optional[Dict[str, Any]]:
    methods = {
        k: m
        for k, m in (description.correlations or {}).items()
        if isinstance(m, pd.DataFrame)
    }
    if not methods:
        return None
    pairs: List[List[Any]] = []
    for method, matrix in methods.items():
        cols = [c for c in matrix.columns if c in matrix.index]
        for i, a in enumerate(cols):
            for b in cols[i + 1 :]:
                if columns is not None and (a not in columns and b not in columns):
                    continue
                try:
                    value = float(matrix.loc[a, b])
                except (TypeError, ValueError):
                    continue
                if math.isnan(value) or abs(value) < opts.corr_threshold:
                    continue
                pair: List[Any] = [str(a), str(b), num(value, 2)]
                if len(methods) > 1:
                    pair.append(method)
                pairs.append(pair)
    pairs.sort(key=lambda p: -abs(p[2]))
    out: Dict[str, Any] = {
        "methods": sorted(methods),
        "threshold": opts.corr_threshold,
        "pairs": pairs[: opts.max_correlation_pairs],
    }
    if len(pairs) > opts.max_correlation_pairs:
        out["pairs_omitted"] = len(pairs) - opts.max_correlation_pairs
    return out


def build_sample(
    description: BaseDescription,
    columns: Optional[set] = None,
    exclude: Optional[set] = None,
) -> Optional[List[Dict[str, Any]]]:
    for sample in description.sample or []:
        data = getattr(sample, "data", None)
        if isinstance(data, pd.DataFrame) and len(data) > 0:
            frame = data.head(_SAMPLE_ROWS)
            keep = [
                c
                for c in frame.columns
                if (columns is None or c in columns) and c not in (exclude or set())
            ]
            frame = frame[keep]
            return [
                {
                    str(k): (
                        (None if pd.isna(v) else _plain(v))
                        if not isinstance(v, (list, dict))
                        else text(v)
                    )
                    for k, v in row.items()
                }
                for row in frame.to_dict(orient="records")
            ]
    return None


# --------------------------------------------------------------------------- #
# Profile
# --------------------------------------------------------------------------- #


def _redacted_types(config: Any) -> set:
    out = set()
    if getattr(config.vars.cat, "redact", False):
        out.add("Categorical")
    if getattr(config.vars.text, "redact", False):
        out.add("Text")
    return out


def build_profile(
    description: BaseDescription,
    config: Any,
    roles: Mapping[str, ColumnRole],
    opts: AgentOptions,
) -> Dict[str, Any]:
    """Build the full (un-budgeted) agent payload."""
    opts.validate()
    variables = description.variables
    if opts.columns is not None:
        unknown = [c for c in opts.columns if c not in variables]
        if unknown:
            raise ValueError(f"Unknown columns: {unknown}")
        selected = [c for c in variables if c in set(opts.columns)]
    else:
        selected = list(variables)
    selected_set = set(selected) if opts.columns is not None else None

    alerts = description.alerts if isinstance(description.alerts, list) else []
    tags = _column_tags(alerts)
    redacted = _redacted_types(config)

    payload: Dict[str, Any] = {
        "schema": SCHEMA_ID,
        "detail": opts.detail,
        "dataset": build_dataset(description, config, roles, opts.include_values),
        "findings": build_findings(alerts, opts, selected_set),
        "columns": {
            str(c): build_column(
                str(c),
                variables[c],
                roles[c],
                tags.get(c, []),
                opts,
                column_description(config, c),
                redact=str(variables[c].get("type")) in redacted,
            )
            for c in selected
        },
    }
    corr = build_correlations(description, opts, selected_set)
    if corr is not None:
        payload["correlations"] = corr

    omitted: Dict[str, Any] = {}
    if opts.include_samples and opts.include_values:
        redacted_cols = {
            c for c, v in variables.items() if str(v.get("type")) in redacted
        }
        sample = build_sample(description, selected_set, redacted_cols)
        if sample:
            payload["sample"] = sample
    elif opts.include_samples:
        omitted["sample_rows"] = "disabled by include_values=False"
    else:
        omitted["sample_rows"] = "include_samples=True"
    if opts.detail != "full":
        omitted["histograms"] = "detail=full"
    omitted["value_counts"] = f"top {opts.top_k} per column"
    if not opts.include_values:
        omitted["values"] = "include_values=False"
    elif redacted:
        omitted["values"] = (
            "redacted for "
            + ", ".join(sorted(normalize_type(t) for t in redacted))
            + " columns"
        )
    payload["omitted"] = omitted
    return payload
