"""Legacy JSON export (``ProfileReport.to_json(legacy=True)``).

This module holds the pre-agent JSON serialisation, moved here unchanged so it
can be deleted in one step. It is frozen: do not fix or extend it. The output
is the full ``BaseDescription`` dump, including chart SVGs, unbounded value
counts, string-formatted alerts and an index-less correlation matrix.

Deprecated: scheduled for removal in the next major release. Use the default
``to_json()`` output (``data-profiling.agent/1``) or ``get_description()``.
"""

import json
from dataclasses import asdict, is_dataclass
from typing import Any, Dict, Union

import numpy as np
import pandas as pd

from data_profiling.config import Settings
from data_profiling.model.description import BaseDescription
from data_profiling.model.sample import Sample

LEGACY_REMOVAL_MESSAGE = (
    "`to_json(legacy=True)` returns the pre-agent JSON dump. It is deprecated and "
    "will be removed in the next major release. Use the default `to_json()` "
    "(schema `data-profiling.agent/1`) or `ProfileReport.get_description()` for "
    "the full statistics. See docs/migration/json-agent.md."
)


def format_summary(summary: Union[BaseDescription, dict]) -> dict:
    """Prepare summary for export to json file.

    Args:
        summary (Union[BaseDescription, dict]): summary to export

    Returns:
        dict: summary as dict
    """

    def fmt(v: Any) -> Any:
        if isinstance(v, dict):
            return {k: fmt(va) for k, va in v.items()}
        else:
            if isinstance(v, pd.Series):
                return fmt(v.to_dict())
            elif (
                isinstance(v, tuple)
                and len(v) == 2
                and all(isinstance(x, np.ndarray) for x in v)
            ):
                return {"counts": v[0].tolist(), "bin_edges": v[1].tolist()}
            else:
                return v

    if isinstance(summary, BaseDescription):
        summary = asdict(summary)

    summary = {k: fmt(v) for k, v in summary.items()}
    return summary


def _redact_column(column: Dict[str, Any]) -> Dict[str, Any]:
    def redact_key(data: Dict[str, Any]) -> Dict[str, Any]:
        return {f"REDACTED_{i}": v for i, (_, v) in enumerate(data.items())}

    def redact_value(data: Dict[str, Any]) -> Dict[str, Any]:
        return {k: f"REDACTED_{i}" for i, (k, _) in enumerate(data.items())}

    keys_to_redact = [
        "block_alias_char_counts",
        "block_alias_values",
        "category_alias_char_counts",
        "category_alias_values",
        "character_counts",
        "script_char_counts",
        "value_counts_index_sorted",
        "value_counts_without_nan",
        "word_counts",
    ]

    values_to_redact = ["first_rows"]

    for field in keys_to_redact:
        if field not in column:
            continue
        is_dict = (isinstance(v, dict) for v in column[field].values())
        if any(is_dict):
            column[field] = {k: redact_key(v) for k, v in column[field].items()}
        else:
            column[field] = redact_key(column[field])

    for field in values_to_redact:
        if field not in column:
            continue
        is_dict = (isinstance(v, dict) for v in column[field].values())
        if any(is_dict):
            column[field] = {k: redact_value(v) for k, v in column[field].items()}
        else:
            column[field] = redact_value(column[field])

    return column


def redact_summary(summary: dict, config: Settings) -> dict:
    """Redact summary to export to json file.

    Args:
        summary (dict): summary to redact

    Returns:
        dict: redacted summary
    """
    for _, col in summary["variables"].items():
        if (config.vars.cat.redact and col["type"] == "Categorical") or (
            config.vars.text.redact and col["type"] == "Text"
        ):
            col = _redact_column(col)
    return summary


def _encode_it(o: Any) -> Any:
    if is_dataclass(o):
        o = asdict(o)  # type: ignore[arg-type]
    if isinstance(o, dict):
        return {_encode_it(k): _encode_it(v) for k, v in o.items()}
    else:
        if isinstance(o, (bool, int, float, str)):
            return o
        elif isinstance(o, list):
            return [_encode_it(v) for v in o]
        elif isinstance(o, set):
            return {_encode_it(v) for v in o}
        elif isinstance(o, pd.Series):
            return _encode_it(o.to_list())
        elif isinstance(o, pd.DataFrame):
            return _encode_it(o.to_dict(orient="records"))
        elif isinstance(o, np.ndarray):
            return _encode_it(o.tolist())
        elif isinstance(o, Sample):
            return _encode_it(o.dict())
        elif isinstance(o, np.generic):
            return o.item()
        else:
            return str(o)


def render_legacy_json(description: BaseDescription, config: Settings) -> str:
    """Serialise a description exactly as ``to_json()`` did before agent/1."""
    description_dict = format_summary(description)
    description_dict = _encode_it(description_dict)
    description_dict = redact_summary(description_dict, config)
    return json.dumps(description_dict, indent=4)
