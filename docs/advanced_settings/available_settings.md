# Available Settings

A set of options is available in order to customize the behaviour of ``data-profiling`` and the appearance of the generated report. The depth of customization allows the creation of behaviours highly targeted at the specific dataset being analysed. The available settings are listed below. To learn how to change them, check :doc:`changing_settings`.

## General settings

Global report settings: 

{{ read_csv('./tables/config_general.csv') }}

## JSON output settings

Settings of the agent-oriented output returned by `to_json()` and `to_markdown()`. See [JSON output for AI agents](../features/agent_output.md).

{{ read_csv('./tables/config_json_output.csv') }}

## Variable summary settings

Settings related with the information displayed for each variable. 

{{ read_csv('./tables/config_variables.csv') }}

```python linenums="1" title="Configuration example"
  profile = df.profile_report(
      sort="ascending",
      vars={
          "num": {"low_categorical_threshold": 0},
          "cat": {
              "length": True,
              "characters": False,
              "words": False,
              "n_obs": 5,
          },
      },
  )

  profile.config.variables.descriptions = {
      "files": "Files in the filesystem",
      "datec": "Creation date",
      "datem": "Modification date",
  }

  profile.to_file("report.html")
```

## Setting dataset schema type

Configure the schema type for a given dataset.

```python linenums="1" title='Set the variable type schema to Generate the profile report'
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

  # We can set the type_schema only for the variables that we are certain of their types.
  # All the other will be automatically inferred.
  report = ProfileReport(df, title="Titanic EDA", type_schema=type_schema)

  report.to_file("report.html")
```
## Missing data overview plots

Settings related with the missing data section and the visualizations it can include. 

{{ read_csv('./tables/config_missing.csv') }}

```python title="Configuration example: disable heatmap for large datasets" linenums="1"
  profile = df.profile_report(
      missing_diagrams={
          "heatmap": False,
      }
  )
  profile.to_file("report.html")
```

## Correlations

Settings regarding correlation metrics and thresholds.    
The default value is `auto`, but the following correlation matrices are available:

{{ read_csv('./tables/corr_matrices.csv') }}

For each correlation matrix you can use the following configurations:

{{ read_csv('./tables/config_correlations.csv') }}

For instance, to disable all correlation computations (might be relevant for large datasets):

````python linenums="1" title="Disabling all correlation matrices"
    profile = df.profile_report(
        title="Report without correlations",
        correlations={
            "auto": {"calculate": False},
            "pearson": {"calculate": False},
            "spearman": {"calculate": False},
            "kendall": {"calculate": False},
            "phi_k": {"calculate": False},
            "cramers": {"calculate": False},
        },
    )

    # or using a shorthand that is available for correlations
    profile = df.profile_report(
        title="Report without correlations",
        correlations=None,
    )
````

## Interactions

Settings related with the interactions section.  

{{ read_csv('./tables/config_interactions.csv') }}

## Report's appearance

Settings related with the appearance and style of the report.

{{ read_csv('./tables/config_html.csv') }}

<img referrerpolicy="no-referrer-when-downgrade" src="https://static.scarf.sh/a.png?x-pxid=baa0e45f-0c03-4190-9646-9d8ea2640ba2" />