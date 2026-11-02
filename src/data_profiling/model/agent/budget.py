"""Fit an agent payload into a token budget.

Priority, highest first: header (schema, dataset) > findings > columns that have
findings > other columns > correlations > sample. The payload degrades in this
order until it fits:

1. drop the sample rows;
2. downgrade columns without findings to ``brief``;
3. trim correlation pairs, then drop correlations;
4. downgrade columns with findings to ``brief``;
5. drop columns without findings (listed by name in ``omitted.columns``);
6. drop ``info`` then ``low`` findings;
7. drop columns with findings.

If the header alone does not fit, the payload is returned with
``omitted.budget = "exceeded"``.
"""

import json
from typing import Any, Callable, Dict, List, Optional

Tokenizer = Callable[[str], int]


def estimate_tokens(text: str) -> int:
    """Rough token estimate: ~4 characters per token for compact JSON."""
    return (len(text) + 3) // 4


def _size(obj: Any, tokenizer: Tokenizer) -> int:
    return tokenizer(json.dumps(obj, separators=(",", ":"), ensure_ascii=False))


def _brief(col: Dict[str, Any]) -> Dict[str, Any]:
    keep = (
        "type",
        "role",
        "role_candidates",
        "desc",
        "missing_pct",
        "distinct",
        "tags",
    )
    out = {k: col[k] for k in keep if k in col}
    if col.get("role") == "unknown" or col.get("role_basis") == ["default"]:
        if "role_basis" in col:
            out["role_basis"] = col["role_basis"]
    return out


def fit_to_budget(
    payload: Dict[str, Any],
    max_tokens: Optional[int],
    tokenizer: Optional[Tokenizer] = None,
) -> Dict[str, Any]:
    if not max_tokens:
        return payload
    tokenizer = tokenizer or estimate_tokens
    p = payload

    def fits() -> bool:
        return _size(p, tokenizer) <= max_tokens

    if fits():
        return p

    omitted = p.setdefault("omitted", {})
    with_findings = {c for f in p.get("findings", []) for c in f.get("cols", [])}
    columns: Dict[str, Dict[str, Any]] = p.get("columns", {})
    order = list(columns)

    # 1. sample
    if "sample" in p:
        del p["sample"]
        omitted["sample_rows"] = "dropped to fit max_tokens"
        if fits():
            return p

    # 2. downgrade columns without findings (largest first)
    def downgrade(names: List[str]) -> bool:
        for name in sorted(names, key=lambda n: -_size(columns[n], tokenizer)):
            brief = _brief(columns[name])
            if brief != columns[name]:
                columns[name] = brief
                omitted["downgraded_to_brief"] = (
                    omitted.get("downgraded_to_brief", 0) + 1
                )
                if fits():
                    return True
        return False

    if downgrade([c for c in order if c not in with_findings]):
        return p

    # 3. correlations
    corr = p.get("correlations")
    if corr:
        while corr["pairs"] and not fits():
            corr["pairs"].pop()
            corr["pairs_omitted"] = corr.get("pairs_omitted", 0) + 1
        if fits():
            return p
        del p["correlations"]
        omitted["correlations"] = "dropped to fit max_tokens"
        if fits():
            return p

    # 4. downgrade columns with findings
    if downgrade([c for c in order if c in with_findings]):
        return p

    # 5. drop columns without findings (from the end)
    dropped: List[str] = omitted.setdefault("columns", [])
    for name in reversed([c for c in order if c not in with_findings]):
        if name in columns:
            del columns[name]
            dropped.insert(0, name)
            if fits():
                return p

    # 6. drop low-priority findings
    for sev in ("info", "low"):
        before = len(p.get("findings", []))
        p["findings"] = [f for f in p.get("findings", []) if f.get("sev") != sev]
        if len(p["findings"]) < before:
            omitted.setdefault("findings", []).append(sev)
        if fits():
            return p

    # 7. drop remaining columns
    for name in reversed([c for c in order if c in columns]):
        del columns[name]
        dropped.insert(0, name)
        if fits():
            return p

    omitted["budget"] = "exceeded"
    return p
