"""Agent-oriented profile output (``data-profiling.agent/1``).

The default ``ProfileReport.to_json()`` output: compact, bounded in size,
findings first, with closed vocabularies for types, roles, tags and severities.
"""

from data_profiling.model.agent.budget import estimate_tokens, fit_to_budget
from data_profiling.model.agent.builder import (
    AgentOptions,
    build_column,
    build_findings,
    build_profile,
)
from data_profiling.model.agent.comparison import build_comparison
from data_profiling.model.agent.render import render_json, render_markdown
from data_profiling.model.agent.schema import json_schema
from data_profiling.model.agent.vocab import (
    DETAIL_LEVELS,
    ROLES,
    SCHEMA_ID,
    SEVERITIES,
    TAGS,
    TYPES,
)

__all__ = [
    "AgentOptions",
    "DETAIL_LEVELS",
    "ROLES",
    "SCHEMA_ID",
    "SEVERITIES",
    "TAGS",
    "TYPES",
    "build_column",
    "build_comparison",
    "build_findings",
    "build_profile",
    "estimate_tokens",
    "fit_to_budget",
    "json_schema",
    "render_json",
    "render_markdown",
]
