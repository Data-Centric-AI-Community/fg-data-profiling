# Column roles

A column's *type* says what its values look like: numeric, text, datetime.
Its *role* says what the column means for analysis. A zip code stored as an
integer is numeric, but averaging it makes no sense; `customer_id` is numeric,
but its histogram says nothing. `data-profiling` infers a role for every column
and shows it in the HTML report (a badge next to the type) and in the JSON
output.

| Role | Meaning |
|---|---|
| `identifier` | Identifies a row or an entity: IDs, UUIDs, unique emails |
| `code` | A code drawn from a fixed scheme: zip codes, SKUs, ISO codes |
| `measure` | A quantity you can aggregate: prices, amounts, counts |
| `dimension` | A category to group by: country, segment, 0/1 flags |
| `timestamp` | A point in time |
| `free_text` | Unstructured text |
| `constant` | A single value |
| `empty` | No values |
| `unknown` | No rule was decisive; see `role_candidates` |

## How roles are inferred

Roles come from explicit rules applied to the statistics that were already
computed, so they add no profiling cost. Every role carries the conditions that
produced it, in `role_basis`:

``` python
profile.get_roles()
```

``` python
{
    "customer_id": {"role": "identifier", "role_basis": ["unique", "integer", "unordered", "name=id"]},
    "order_uuid":  {"role": "identifier", "role_basis": ["unique", "fixed_length"]},
    "zip":         {"role": "code",       "role_basis": ["digits_only", "fixed_width"]},
    "country":     {"role": "dimension",  "role_basis": ["low_cardinality(n=5)"]},
    "price":       {"role": "measure",    "role_basis": ["name=price"]},
    "amount":      {"role": "measure",    "role_basis": ["default"]},
    "year":        {"role": "unknown",    "role_basis": ["low_cardinality_integer", "year_range", "name=year"],
                    "role_candidates": ["dimension", "measure"]},
}
```

Column names are supporting evidence. They can confirm a role or settle a case
the statistics leave open (an integer column named `zip` becomes a `code`; a
low-cardinality integer named `quantity` becomes a `measure`), and whenever a
name was used it appears in `role_basis` as `name=...`, so a role that relies on
a name is visible as such. A `["default"]` basis means no rule matched and the
role is a fallback.

Some roles cannot be inferred from data at all. `year` is a dimension when you
group by it and a measure when you fit a trend; a 1–5 rating can be either. For
these the role is `unknown` with the candidates listed. The right answer depends
on the question being asked, so it is left to the user, or to the agent reading
the JSON, which knows the task.

## Reviewing roles

`get_roles(unresolved=True)` returns only the columns whose role is `unknown`
or a `default` fallback. The HTML report lists the same columns in the *Column
roles* card of the overview, with the configuration snippet that sets them.

## Setting roles

Roles set by the user always win, and their basis becomes `["user"]`.

When creating the report, next to the column descriptions:

``` python linenums="1" title="Set roles in the configuration"
profile = ProfileReport(
    df,
    variables={"roles": {"zip_int": "code", "year": "dimension"}},
)
```

After profiling, without recomputing any statistics:

``` python linenums="1" title="Set roles on an existing report"
profile.get_roles(unresolved=True)
profile.set_roles({"category_id": "dimension", "quantity": "measure"})
profile.to_file("report.html")   # re-rendered with the new roles
```

Roles set with `set_roles` are stored in `profile.config.variables.roles`, so
they are kept by `dump()` / `load()` and apply to `compare()`.

### Column-definition files

The column descriptions file can carry the role as well. A plain string is
still read as a description, so existing files keep working:

``` json title="columns.json"
{
    "zip_int": {"description": "Customer postal code", "role": "code"},
    "year":    {"role": "dimension"},
    "price":   "Unit price in EUR"
}
```

``` python
import json

definitions = json.load(open("columns.json"))
profile = ProfileReport(df, variables={"descriptions": definitions})
```

When the same column appears in both, `variables.roles` wins over a role in
`variables.descriptions`.

### Validation

- An unknown role name raises a `ValueError` listing the valid roles. Roles are a
  closed set; use the description for anything more specific.
- A column that is not in the profile gives a warning, not an error, so one
  definition file can serve several versions of a table.
- An unusual combination, such as `timestamp` on a numeric column (legitimate
  for epoch seconds), gives a warning and is kept.
