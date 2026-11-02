# Missingness associations (experimental)

Enable aggregate explanations in the existing Missing values section:

```python
from data_profiling import ProfileReport

profile = ProfileReport(df, missing_diagrams={"associations": True})
```

This opt-in analysis reports columns that are missing together and groups of
inferred categorical/boolean values with elevated missing rates. It honors custom
`type_schema` choices. For example, income and spend missing on the same rows can
produce one missingness-pair finding; income missing mainly for segment C can
produce a categorical finding. These are observed associations, not causes.
No findings does not establish that missingness is random.

Details are computed once under `description.missing["associations"]`; alerts link
to the details. JSON contains bounded aggregate records, never row indices.
Comparison reports retain each dataset's findings separately. Sensitive category
labels are redacted when categorical/text redaction is enabled.

The provisional safeguards are positive phi >= 0.3 for missingness pairs,
bias-corrected Cramer's V >= 0.3 for each category-versus-rest table, missing-rate
uplift >= 0.20, and group/complement support >= max(20 rows, 1% of all rows).
At least that many missing rows must occur in the group. Continuity-corrected
chi-square tests are used when every expected cell count is at least five;
sparse tables use two-sided Fisher exact tests. These tests are corrected
together using Benjamini-Hochberg at 0.05 before retaining findings.
Predictor-null rows are excluded from category tests and covered by
missingness-indicator tests instead. Negative indicator associations are excluded.

Empty/all-missing indicators and constant predictors are skipped. Predictors
with more than 20 distinct values are excluded. The analysis skips explicitly,
with a warning, above 100 eligible columns, 5,000 candidate hypotheses, or an
estimated 64 MiB of indicator/workspace arrays. It retains at most 50 findings,
ranked deterministically, and warns on truncation. These bounds apply to this
analysis's working arrays, not the entire report's memory footprint.
Eligible column names longer than 200 characters also cause an explicit skip;
category labels are shortened and given group numbers to bound output size.

Minimal reports always bypass the analysis, even if explicitly requested.
`missing_diagrams=None` disables it. Spark returns an explicit unsupported status
and warning when requested; no dataframe collection is performed. Configuration
defaults, statistical thresholds and limits remain provisional pending #1881.
