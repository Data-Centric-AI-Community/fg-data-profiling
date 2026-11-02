# JSON output for AI agents

`ProfileReport.to_json()` returns a compact profile designed to be read by AI
agents and LLM-based systems. It follows the schema `data-profiling.agent/1`:
a dataset header, the findings sorted by severity, and one bounded entry per
column with its type, semantic role and key statistics.

The output size grows with the number of columns, not rows, and it contains no
charts and no full value-count tables. On a 300-row, 8-column table the
`standard` output is about 600 tokens; the previous JSON dump of the same
profile was about 76,000.

``` python linenums="1" title="Export the profile for an agent"
from data_profiling import ProfileReport

profile = ProfileReport(df, title="Customers")

profile.to_json()          # compact JSON string
profile.to_dict()          # the same payload as a Python dict
profile.to_markdown()      # the same content as Markdown, for prompts
profile.to_file("customers.json")
profile.to_file("customers.md")
```

## Example

Abridged output at the default `standard` detail:

``` json
{
  "schema": "data-profiling.agent/1",
  "detail": "standard",
  "dataset": {"title": "Customers", "rows": 300, "columns": 8, "missing_cells_pct": 1.25,
              "duplicate_rows_pct": 0,
              "types": {"numeric": 3, "categorical": 2, "datetime": 1, "boolean": 1, "text": 1},
              "roles": {"measure": 2, "dimension": 2, "identifier": 1, "timestamp": 1,
                        "free_text": 1, "constant": 1}},
  "findings": [
    {"sev": "high", "tag": "constant", "cols": ["const"], "msg": "has a constant value"},
    {"sev": "medium", "tag": "missing", "cols": ["income"], "msg": "30 (10.0%) missing values", "value": 10},
    {"sev": "low", "tag": "skewed", "cols": ["income"], "msg": "is highly skewed(γ1 = 3.68)", "value": 3.68}
  ],
  "columns": {
    "income": {"type": "numeric", "role": "measure", "role_basis": ["name=income"],
               "missing_pct": 10, "distinct": 270, "tags": ["missing", "skewed"],
               "stats": {"min": 1690, "p5": 4480, "p25": 11600, "p50": 23200, "p75": 45400,
                         "p95": 119000, "max": 402000, "mean": 36800, "std": 44500, "skew": 3.68}},
    "city":   {"type": "categorical", "role": "dimension", "role_basis": ["type=categorical"],
               "distinct": 3, "top": [["Porto", 38.3], ["Faro", 33.3], ["Lisbon", 28.3]]},
    "id":     {"type": "numeric", "role": "identifier", "role_basis": ["unique", "integer", "monotonic", "name=id"],
               "distinct": 300, "tags": ["uniform", "unique"], "range": [0, 299], "monotonic": "increasing_strict"},
    "review": {"type": "text", "role": "free_text", "role_basis": ["type=text"], "distinct": 188,
               "len": {"min": 7, "p50": 20, "max": 24}, "top_terms": ["bad", "fast", "good", "slow"]}
  },
  "correlations": {"methods": ["auto"], "threshold": 0.5, "pairs": []},
  "omitted": {"sample_rows": "include_samples=True", "histograms": "detail=full",
              "value_counts": "top 5 per column"}
}
```

## Structure

| Block | Content |
|---|---|
| `schema` | Always `data-profiling.agent/1`. Check it when you depend on the shape. |
| `dataset` | Rows, columns, missing cells and duplicate rows (%), counts per type and per role, dataset metadata (`description`, `creator`, `author`, `url`) and, in time-series mode, `time_index`. |
| `findings` | The data quality alerts, highest severity first. `cols` lists the columns involved (empty for dataset-level findings); `value` holds the main metric where there is one (percentages are 0–100). |
| `columns` | One entry per column. See below. |
| `correlations` | Labelled pairs `[a, b, r]` with \|r\| ≥ `threshold`, strongest first. A fourth element names the method when several are computed. |
| `sample` | Up to five head rows, only with `include_samples=True`. |
| `omitted` | What exists but was left out, and how to get it. |

Column entries always carry `type` and `role`, plus `missing_pct` (when above
zero), `distinct`, the column's `tags` and, when set, the user description as
`desc`. The `standard` and `full` levels add role-appropriate statistics:

| Role | Added fields |
|---|---|
| `measure` | `stats` (min, quantiles, max, mean, std, skew), `zeros_pct`, `negative_pct`, `infinite_pct` |
| `identifier` | `range`, `monotonic` (no distribution statistics) |
| `code`, `dimension` | `top` as `[value, pct]` pairs, `other_pct` |
| `timestamp` | `range` |
| `free_text` | `len` (min / median / max length), `top_terms` |
| `constant` | `value` |

`full` adds coarse histograms (at most 10 bins), kurtosis, MAD, coefficient of
variation, and text and categorical diagnostics.

## Vocabularies

All tags come from closed lists, published in the JSON Schema so an agent can
branch on them without parsing text:

``` python
from data_profiling import agent

agent.json_schema()   # JSON Schema (draft 2020-12) of to_json()
agent.TYPES           # numeric, categorical, boolean, datetime, text, url, path, file, image, timeseries, unsupported
agent.ROLES           # identifier, code, measure, dimension, timestamp, free_text, constant, empty, unknown
agent.TAGS            # alert types, e.g. missing, constant, skewed, high_correlation, imbalance, unique
agent.SEVERITIES      # high, medium, low, info
```

Severities default to a fixed table (for example `constant` → high,
`missing` → medium, `skewed` → low, `unique` → info) and can be overridden with
`json_output.severity`. The alert thresholds themselves are shared with the HTML
report, so both outputs always flag the same findings.

The roles are explained in [Column roles](column_roles.md).

## Detail levels and token budgets

| `detail` | Per column | Typical size |
|---|---|---|
| `brief` | type, role, missing, distinct, tags | ~25–50 tokens |
| `standard` (default) | + statistics for the column's role | ~60–120 tokens |
| `full` | + histograms and diagnostics | ~200–400 tokens |

`max_tokens` caps the output. The profile degrades in priority order until it
fits: sample rows go first, then columns without findings are reduced to
`brief`, correlations are trimmed, columns with findings are reduced, columns
without findings are dropped (listed by name in `omitted.columns`), low-priority
findings are dropped, and finally the remaining columns. Tokens are estimated at
~4 characters each; pass `tokenizer=` for exact counts.

``` python linenums="1" title="Fit the profile into a prompt"
import tiktoken

enc = tiktoken.get_encoding("cl100k_base")
profile.to_json(max_tokens=2000, tokenizer=lambda s: len(enc.encode(s)))
```

An agent can start from `brief` and ask for more where it matters:

``` python
profile.to_json(detail="brief")
profile.to_json(columns=["income", "city"], detail="full")
profile.describe_column("income")   # one column, detail="full", with its findings
```

## Privacy

The output contains aggregate statistics plus, by default, a few observed values:
the top values of categorical columns, top terms of text columns and date
ranges. Sample rows are only included with `include_samples=True`.

`include_values=False` drops the values that identify records: top values,
top terms, date and time-index ranges, identifier ranges, constant values and
sample rows. Numeric summary statistics stay (min, quantiles, max, mean, std),
because they describe the distribution an agent needs to reason about; note that
min and max are observed values. This is the recommended setting before sending
a profile of sensitive data to a hosted model. `sensitive=True` (or
`vars.cat.redact` / `vars.text.redact`) removes the values of categorical and
text columns, including from sample rows.

``` python
profile.to_json(include_values=False)
```

## Comparing datasets

For a comparison report, `to_json()` returns one profile per dataset plus the
differences between the first dataset and each of the others:

``` python linenums="1" title="Compare two datasets"
comparison = train_report.compare(test_report)
comparison.to_json()
```

``` json
{"schema": "data-profiling.agent/1", "kind": "comparison", "labels": ["Train", "Test"],
 "delta": [{"from": "Train", "to": "Test",
            "rows": [5000, 1200],
            "columns": {"age": {"missing_pct": [0, 40]}},
            "findings": {"new": [{"sev": "medium", "tag": "missing", "cols": ["age"], "msg": "...", "value": 40}],
                         "resolved": []}}],
 "reports": [{"dataset": {...}, "findings": [...], "columns": {...}}, {...}]}
```

With `max_tokens`, the budget is shared equally between the reports; the deltas
are always kept whole. Findings on columns that exist in only one dataset are
reported through `columns_added` / `columns_removed`, not as new or resolved
findings.

A column change is reported when its type or role changes, missing values move
by at least 1 percentage point, the distinct count changes by at least 10%, or
the mean shifts by at least 0.1 standard deviations.

## Settings

All arguments default to the `json_output` settings, so the output can also be
configured once, in a config file or the constructor:

``` python
ProfileReport(df, json_output={"detail": "brief", "include_values": False})
```

See [Available settings](../advanced_settings/available_settings.md#json-output-settings).
The minimal configuration (`minimal=True`) defaults to `brief`.

## Previous JSON output

The previous JSON dump of the full description is available until the next
major release:

``` python
profile.to_json(legacy=True)
profile.to_file("report.json", legacy=True)
```

It emits a `FutureWarning`. See the [migration guide](../migration/json-agent.md).
For programmatic access to every computed statistic, use
`profile.get_description()`.
