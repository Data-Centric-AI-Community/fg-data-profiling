"""JSON Schema for ``data-profiling.agent/1``."""

import copy
from typing import Any, Dict

from data_profiling.model.agent.vocab import (
    DETAIL_LEVELS,
    ROLES,
    SCHEMA_ID,
    SEVERITIES,
    TAGS,
    TYPES,
)

_NUMBER_OR_NULL = {"type": ["number", "null"]}
_SCALAR = {"type": ["string", "number", "boolean", "null"]}
_TOP = {
    "type": "array",
    "items": {
        "type": "array",
        "prefixItems": [_SCALAR, _NUMBER_OR_NULL],
        "minItems": 2,
        "maxItems": 2,
    },
}

_COLUMN: Dict[str, Any] = {
    "type": "object",
    "required": ["type", "role"],
    "properties": {
        "type": {"enum": list(TYPES)},
        "role": {"enum": list(ROLES)},
        "role_basis": {"type": "array", "items": {"type": "string"}},
        "role_candidates": {"type": "array", "items": {"enum": list(ROLES)}},
        "desc": {"type": "string"},
        "missing_pct": {"type": "number", "minimum": 0, "maximum": 100},
        "distinct": {"type": "number", "minimum": 0},
        "tags": {"type": "array", "items": {"enum": list(TAGS)}},
        "stats": {
            "type": "object",
            "additionalProperties": {"type": ["number", "string", "null"]},
        },
        "range": {"type": "array", "minItems": 2, "maxItems": 2},
        "monotonic": {"type": "string"},
        "top": _TOP,
        "other_pct": {"type": "number"},
        "value": _SCALAR,
        "len": {"type": "object"},
        "top_terms": {"type": "array", "items": {"type": "string"}},
        "zeros_pct": {"type": "number"},
        "negative_pct": {"type": "number"},
        "infinite_pct": {"type": "number"},
        "invalid_pct": {"type": "number"},
        "histogram": {
            "type": "object",
            "required": ["edges", "counts"],
            "properties": {
                "edges": {"type": "array"},
                "counts": {"type": "array", "items": {"type": "integer"}},
            },
        },
        "imbalance": {"type": ["number", "null"]},
        "chi_squared_p": {"type": ["number", "null"]},
        "chars_distinct": {"type": ["number", "null"]},
        "scripts": {"type": "array", "items": {"type": "string"}},
    },
    "additionalProperties": False,
}

_FINDING: Dict[str, Any] = {
    "type": "object",
    "required": ["sev", "tag", "cols", "msg"],
    "properties": {
        "sev": {"enum": list(SEVERITIES)},
        "tag": {"enum": list(TAGS)},
        "cols": {"type": "array", "items": {"type": "string"}},
        "msg": {"type": "string"},
        "value": {"type": ["number", "null"]},
    },
    "additionalProperties": False,
}

_DATASET: Dict[str, Any] = {
    "type": "object",
    "required": ["rows", "columns", "missing_cells_pct", "types"],
    "properties": {
        "title": {"type": "string"},
        "label": {"type": "string"},
        "rows": {"type": "integer", "minimum": 0},
        "columns": {"type": "integer", "minimum": 0},
        "missing_cells_pct": {"type": "number"},
        "duplicate_rows_pct": {"type": ["number", "null"]},
        "types": {"type": "object", "additionalProperties": {"type": "integer"}},
        "roles": {"type": "object", "additionalProperties": {"type": "integer"}},
        "description": {"type": "string"},
        "creator": {"type": "string"},
        "author": {"type": "string"},
        "url": {"type": "string"},
        "time_index": {"type": "object"},
    },
    "additionalProperties": False,
}

_PROFILE_PROPERTIES: Dict[str, Any] = {
    "dataset": _DATASET,
    "findings": {"type": "array", "items": _FINDING},
    "columns": {"type": "object", "additionalProperties": _COLUMN},
    "correlations": {
        "type": "object",
        "required": ["threshold", "pairs"],
        "properties": {
            "methods": {"type": "array", "items": {"type": "string"}},
            "threshold": {"type": "number"},
            "pairs": {
                "type": "array",
                "items": {"type": "array", "minItems": 3, "maxItems": 4},
            },
            "pairs_omitted": {"type": "integer"},
        },
    },
    "sample": {"type": "array", "items": {"type": "object"}},
    "omitted": {"type": "object"},
}

_SINGLE: Dict[str, Any] = {
    "type": "object",
    "required": ["schema", "detail", "dataset", "findings", "columns"],
    "properties": {
        "schema": {"const": SCHEMA_ID},
        "detail": {"enum": list(DETAIL_LEVELS)},
        **_PROFILE_PROPERTIES,
    },
    "additionalProperties": False,
}

_REPORT_IN_COMPARISON: Dict[str, Any] = {
    "type": "object",
    "required": ["dataset", "findings", "columns"],
    "properties": _PROFILE_PROPERTIES,
    "additionalProperties": False,
}

_COMPARISON: Dict[str, Any] = {
    "type": "object",
    "required": ["schema", "kind", "labels", "delta", "reports"],
    "properties": {
        "schema": {"const": SCHEMA_ID},
        "kind": {"const": "comparison"},
        "detail": {"enum": list(DETAIL_LEVELS)},
        "labels": {"type": "array", "items": {"type": "string"}},
        "delta": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["from", "to", "columns", "findings"],
            },
        },
        "reports": {"type": "array", "items": _REPORT_IN_COMPARISON},
    },
    "additionalProperties": False,
}

_SCHEMA: Dict[str, Any] = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "$id": f"https://data-centric-ai-community.github.io/schemas/{SCHEMA_ID}.json",
    "title": "Data profile for AI agents (data-profiling.agent/1)",
    "oneOf": [_SINGLE, _COMPARISON],
}

#: Schema for the payload of ``ProfileReport.describe_column``.
COLUMN_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "required": ["schema", "column", "findings"],
    "properties": {
        "schema": {"const": SCHEMA_ID},
        "column": {"type": "string"},
        "findings": {"type": "array", "items": _FINDING},
    },
}


def json_schema() -> Dict[str, Any]:
    """JSON Schema (draft 2020-12) of the default ``to_json()`` output."""
    return copy.deepcopy(_SCHEMA)
