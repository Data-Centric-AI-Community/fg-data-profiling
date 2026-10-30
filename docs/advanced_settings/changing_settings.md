# Changing settings

There are three ways to change the settings listed in
`available_settings`{.interpreted-text role="doc"}:

-   through code
-   through a custom configuration file
-   through environment variables

## Through code

```python linenums="1" title="Configuration example"
# Create the ProfileReport object without specifying a DataFrame
profile = df.profile_report(title="Profiling Report", pool_size=1)

# Change the configuration as desired
profile.config.html.minify_html = False

# Specify a DataFrame and trigger the report's computation
profile.to_file("output.html")
```

Some related settings are grouped together in configuration shorthands,
making it easy to selectively enable or disable certain report sections
or functionality:

-   `samples`: control whether the dataset preview is shown.
-   `correlations`: control whether correlation computations are
    executed.
-   `missing_diagrams`: control whether missing value analysis is
    executed.
-   `duplicates`: control whether duplicate rows are previewed.
-   `interactions`: control whether interactions are computed.

```python linenums="1" title="Disable samples, correlations, missing diagrams and duplicates at once"
r = ProfileReport(
    samples=None,
    correlations=None,
    missing_diagrams=None,
    duplicates=None,
    interactions=None,
)
```

## Through a custom configuration file

To control `data-profiling` through a custom file, you can start with
one of the sample configuration files below:

-   [default configuration
    file](https://github.com/Data-Centric-AI-Community/data-profiling/blob/master/src/data_profiling/config_default.yaml)
    (default)
-   [minimal configuration
    file](https://github.com/Data-Centric-AI-Community/data-profiling/blob/master/src/data_profiling/config_minimal.yaml)
    (minimal computation, optimized for performance)

Change the configuration to your liking and point towards that
configuration file when computing the report:

``` python linenums="1" title="Custom configuration file"
from data_profiling import ProfileReport

profile = ProfileReport(df, config_file="your_config.yml")
profile.to_file("report.html")
```

## Through environment variables

Any configuration setting can also be read from environment variables.
For example:

```python linenums="1" title="Setting title for the report with parameters"
from data_profiling import ProfileReport

profile = ProfileReport(df, title="My Custom Profiling Report")
```

is equivalent to setting the title as an environment variable

```python linenums="1" title="Set title through environment variables"
import os
from data_profiling import ProfileReport

os.environ["PROFILE_TITLE"] = "My Custom Profiling Report"

profile = ProfileReport(df)
```

<img referrerpolicy="no-referrer-when-downgrade" src="https://static.scarf.sh/a.png?x-pxid=baa0e45f-0c03-4190-9646-9d8ea2640ba2" />
