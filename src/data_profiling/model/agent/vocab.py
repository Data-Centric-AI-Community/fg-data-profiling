"""Closed vocabularies of the agent output: types, roles, tags and severities."""

from typing import Dict, Mapping, Optional

from data_profiling.model.alerts import AlertType
from data_profiling.model.roles import ROLES

SCHEMA_ID = "data-profiling.agent/1"

DETAIL_LEVELS = ("brief", "standard", "full")

TYPES = (
    "numeric",
    "categorical",
    "boolean",
    "datetime",
    "text",
    "url",
    "path",
    "file",
    "image",
    "timeseries",
    "unsupported",
)

_TYPE_MAP = {
    "Numeric": "numeric",
    "Categorical": "categorical",
    "Boolean": "boolean",
    "DateTime": "datetime",
    "Text": "text",
    "URL": "url",
    "Path": "path",
    "File": "file",
    "Image": "image",
    "TimeSeries": "timeseries",
    "Unsupported": "unsupported",
}

SEVERITIES = ("high", "medium", "low", "info")

#: Default severity per alert type. Overridable with `json_output.severity`.
DEFAULT_SEVERITY: Dict[str, str] = {
    "EMPTY": "high",
    "CONSTANT": "high",
    "INFINITE": "high",
    "UNSUPPORTED": "high",
    "MISSING": "medium",
    "DUPLICATES": "medium",
    "NEAR_DUPLICATES": "medium",
    "HIGH_CORRELATION": "medium",
    "DIRTY_CATEGORY": "medium",
    "TYPE_DATE": "medium",
    "SKEWED": "low",
    "IMBALANCE": "low",
    "ZEROS": "low",
    "HIGH_CARDINALITY": "low",
    "NON_STATIONARY": "low",
    "UNIQUE": "info",
    "UNIFORM": "info",
    "CONSTANT_LENGTH": "info",
    "SEASONAL": "info",
}

#: Alert types that are bookkeeping, not findings.
EXCLUDED_ALERTS = {AlertType.REJECTED}

TAGS = tuple(
    t.name.lower() for t in AlertType if t not in EXCLUDED_ALERTS
)  # type: ignore[attr-defined]


def normalize_type(vtype: object) -> str:
    return _TYPE_MAP.get(str(vtype), str(vtype).lower())


def severity_of(
    alert_type: AlertType, overrides: Optional[Mapping[str, str]] = None
) -> str:
    name = alert_type.name
    if overrides:
        for key in (name, name.lower()):
            if key in overrides:
                value = overrides[key]
                if value not in SEVERITIES:
                    raise ValueError(
                        f"Invalid severity {value!r} for {name}. Valid: {', '.join(SEVERITIES)}."
                    )
                return value
    return DEFAULT_SEVERITY.get(name, "low")


__all__ = [
    "SCHEMA_ID",
    "DETAIL_LEVELS",
    "TYPES",
    "ROLES",
    "TAGS",
    "SEVERITIES",
    "DEFAULT_SEVERITY",
    "EXCLUDED_ALERTS",
    "normalize_type",
    "severity_of",
]
