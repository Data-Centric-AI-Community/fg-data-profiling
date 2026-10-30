# Dataset description & Metadata

## Dataset description

When sharing reports with coworkers or publishing online, it might be
important to include metadata of the dataset, such as author, copyright
holder or descriptions. `data-profiling` allows complementing a report
with that information. Inspired by [schema.org\'s
Dataset](https://schema.org/Dataset), the currently supported properties
are *description*, *creator*, *author*, *url*, *copyright_year* and
*copyright_holder*.

The following example shows how to generate a report with a
*description*, *copyright_holder* *copyright_year*, *creator* and *url*.
In the generated report, these properties are found in the *Overview*,
under *About*.

``` python linenums="1" title="Add profile report description"
report = df.profile_report(
    title="Masked data",
    dataset={
        "description": "This profiling report was generated using a sample of 5% of the original dataset.",
        "copyright_holder": "StataCorp LLC",
        "copyright_year": 2020,
        "url": "http://www.stata-press.com/data/r15/auto2.dta",
    },
)

report.to_file(Path("stata_auto_report.html"))
```

## Column descriptions

In addition to providing dataset details, often users want to include
column-specific descriptions when sharing reports with team members and
stakeholders. `data-profiling` supports creating these descriptions, so
that the report includes a built-in data dictionary. By default, the
descriptions are presented in the *Overview* section of the report, next
to each variable.

``` python linenums="1" title="Generate a report with per-variable descriptions"
profile = df.profile_report(
    variables={
        "descriptions": {
            "files": "Files in the filesystem, # variable name: variable description",
            "datec": "Creation date",
            "datem": "Modification date",
        }
    }
)

profile.to_file("report.html")
```

Alternatively, column descriptions can be loaded from a JSON file:

``` python linenums="1" title="dataset_column_definition.json"
{
    "column name 1": "column 1 definition",
    "column name 2": "column 2 definition"
}
```

``` python linenums="1" title="Generate a report with descriptions per variable from a JSON definitions file"
import json
import pandas as pd
import data_profiling

definition_file = "dataset_column_definition.json"

# Read the variable descriptions
with open(definition_file, "r") as f:
    definitions = json.load(f)

# By default, the descriptions are presented in the Overview section, next to each variable
report = df.profile_report(variables={"descriptions": definitions})

# We can disable showing the descriptions next to each variable
report = df.profile_report(
    variables={"descriptions": definitions}, show_variable_description=False
)

report.to_file("report.html")
```

## Dataset schema

In addition to providing dataset details, users often want to include
set type schemas. This is particularly important when integrating
`data-profiling` generation with the information already in a data
catalog. When using `data-profiling` ProfileReport, users can set the
type_schema property to control the generated profiling data types. By
default, the `type_schema` is automatically inferred with [visions](https://github.com/dylan-profiler/visions).

``` python linenums="1" title="Set the variable type schema to Generate the profile report"
import json
import pandas as pd

from data_profiling import ProfileReport
from data_profiling.utils.cache import cache_file

file_name = cache_file(
    "titanic.csv",
    "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv",
)
df = pd.read_csv(file_name)

type_schema = {"Survived": "categorical", "Embarked": "categorical"}

# We can set the type_schema only for the variables that we are certain of their types. All the other will be automatically inferred.
report = ProfileReport(df, title="Titanic EDA", type_schema=type_schema)

report.to_file("report.html")
```

<img referrerpolicy="no-referrer-when-downgrade" src="https://static.scarf.sh/a.png?x-pxid=baa0e45f-0c03-4190-9646-9d8ea2640ba2" />
