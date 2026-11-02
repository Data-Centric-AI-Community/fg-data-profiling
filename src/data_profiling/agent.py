"""Public entry points for the agent-oriented profile output.

```python
from data_profiling import agent

agent.json_schema()   # JSON Schema of ProfileReport.to_json()
agent.ROLES           # closed role vocabulary
```
"""

from data_profiling.model.agent import (  # noqa: F401
    DETAIL_LEVELS,
    ROLES,
    SCHEMA_ID,
    SEVERITIES,
    TAGS,
    TYPES,
    estimate_tokens,
    json_schema,
)

__all__ = [
    "DETAIL_LEVELS",
    "ROLES",
    "SCHEMA_ID",
    "SEVERITIES",
    "TAGS",
    "TYPES",
    "estimate_tokens",
    "json_schema",
]
