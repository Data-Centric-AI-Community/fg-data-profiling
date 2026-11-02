# Migrating to the agent JSON output

`ProfileReport.to_json()` now returns the agent-oriented profile
(`data-profiling.agent/1`) instead of a dump of the full description. This page
lists what changed and how to move existing code.

## Timeline

| Release | `to_json()` | `to_json(legacy=True)` |
|---|---|---|
| This major release | agent output | previous dump, unchanged, with a `FutureWarning` |
| Following minor releases | agent output, additive changes only | same |
| Next major release | agent output | removed |

The schema version (`data-profiling.agent/1`) is independent of the package
version. New fields are added without changing it; renamed or removed fields
change it.

## Keeping the previous output for now

``` python
profile.to_json(legacy=True)
profile.to_file("report.json", legacy=True)
```

Or in configuration, for code you cannot change yet:

``` yaml
json_output:
  legacy: true
```

From the command line: `data_profiling --legacy-json data.csv report.json`.

The legacy output is frozen: it is the same as before, including its known
limitations (chart SVGs embedded as strings, unbounded value counts, alerts as
plain strings, correlation matrices without row labels). The only difference is
that the configuration embedded under `package.data_profiling_config` lists the
new settings.

## Where the old fields went

| Previous path | New path |
|---|---|
| `table.n` | `dataset.rows` |
| `table.n_var` | `dataset.columns` |
| `table.p_cells_missing` | `dataset.missing_cells_pct` (×100) |
| `table.p_duplicates` | `dataset.duplicate_rows_pct` (×100) |
| `table.types` | `dataset.types` (lower-case keys) |
| `analysis.title` | `dataset.title` (HTML tags removed) |
| `alerts[i]` (string) | `findings[i]` with `sev`, `tag`, `cols`, `msg`, `value` |
| `variables.<c>.type` | `columns.<c>.type` (lower-case) |
| `variables.<c>.p_missing` | `columns.<c>.missing_pct` (×100, omitted when 0) |
| `variables.<c>.n_distinct` | `columns.<c>.distinct` |
| `variables.<c>.mean`, `std`, `min`, `max`, `5%`…`95%`, `skewness` | `columns.<c>.stats.mean`, `std`, `min`, `max`, `p5`…`p95`, `skew` |
| `variables.<c>.value_counts_without_nan` | `columns.<c>.top` (top 5, as percentages) |
| `variables.<c>.word_counts` | `columns.<c>.top_terms` |
| `variables.<c>.histogram` | `columns.<c>.histogram` with `detail="full"` (at most 10 bins) |
| `correlations.<method>` (matrix) | `correlations.pairs` (`[a, b, r]` above `corr_threshold`) |
| `sample` | `sample` with `include_samples=True` (5 head rows) |

Not in the JSON any more: scatter and missing-value charts, full value counts,
character and Unicode statistics, the duplicate rows table and the embedded
configuration. All of them remain available from
`profile.get_description()`, which returns the full computed description.

## Other changes in this release

- **Column roles.** Every column has a semantic role, shown in the HTML report
  and the JSON output. See [Column roles](../features/column_roles.md).
- **Skewness alert threshold.** `vars.num.skewness_threshold` changed from 20 to
  1, the common rule of thumb for highly skewed data. Set it back with
  `vars={"num": {"skewness_threshold": 20}}` to keep the previous alerts.
- **Column definitions.** Values in `variables.descriptions` can now be a dict
  (`{"description": ..., "role": ...}`); plain strings work as before.
- **Markdown export.** `to_markdown()` and `to_file("report.md")`.
- **Dumps.** `dumps()` / `dump()` now also store the source column dtypes and,
  for comparison reports, the per-report descriptions, so roles and the agent
  JSON are the same after `load()`. Dumps from earlier versions still load; a
  comparison report loaded from an older dump can only produce
  `to_json(legacy=True)`.
