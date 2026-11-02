# Quickstart

Start by loading your pandas `DataFrame` as you normally would, e.g. by
using:

``` python linenums="1"
import numpy as np
import pandas as pd
from data_profiling import ProfileReport

df = pd.DataFrame(np.random.rand(100, 5), columns=["a", "b", "c", "d", "e"])
```

To generate the standard profiling report, merely run:

``` python
profile = ProfileReport(df, title="YData Profiling Report")
```

## Using inside Jupyter Notebooks

There are two interfaces to consume the report inside a Jupyter notebook
(see animations below): through widgets and through an embedded HTML
report.

![Running data-proling inside a Jupyter Notebook](../_static/img/widgets.gif)

This is achieved by simply displaying the report as a set of widgets. In
a Jupyter Notebook, run:

``` python
profile.to_widgets()
```

The HTML report can be directly embedded in a cell in a similar fashion:

``` python
profile.to_notebook_iframe()
```

![data-profiling widgets](../_static/img/iframe.gif)

## Exporting the report to a file

To generate a HTML report file, save the `ProfileReport` to an object
and use the `to_file()` function:

``` python
profile.to_file("your_report.html")
```

The profile can also be exported as compact JSON or Markdown, designed to be
read by AI agents and LLM-based systems (schema `data-profiling.agent/1`):

``` python linenums="1" title="Export the profile for an AI agent"
# As a JSON string: dataset header, findings by severity, one entry per column
json_data = profile.to_json()

# Same content as Markdown, to paste into a prompt
markdown = profile.to_markdown()

# As files
profile.to_file("your_report.json")
profile.to_file("your_report.md")
```

See [JSON output for AI agents](../features/agent_output.md) for the schema,
detail levels and token budgets. The previous JSON dump is still available with
`profile.to_json(legacy=True)` until the next major release; see the
[migration guide](../migration/json-agent.md).

## Command line usage

For standard formatted CSV files (which can be read directly by pandas
without additional settings), the `data_profiling` executable can be
used in the command line. The example below generates a report named
*Example Profiling Report*, using a configuration file called
`default.yaml`, in the file `report.html` by processing a `data.csv`
dataset.

``` bash
data_profiling --title "Example Profiling Report" --config_file default.yaml data.csv report.html
```

Information about all available options and arguments can be viewed
through the command below. The CLI allows defining input and output
filenames, setting a custom report title, specifying
`a configuration file for custom behaviour <../advanced_usage/changing_settings>`{.interpreted-text
role="doc"} and control other advanced aspects of the experience.

``` bash
data_profiling -h
```
<figure markdown>
  ![Image title](../_static/img/cli.png){width="500"}
  <figcaption>Options available in the CLI</figcaption>
</figure>

## Deeper profiling

The contents, behaviour and appearance of the report are easily
customizable. The example below used the explorative mode, a lightweight
data profiling option.

``` python
profile = ProfileReport(df, title="Profiling Report", explorative=True)
```

On the CLI utility `data_profiling`, this mode can be activated with
the `-e` flag. Learn more about configuring `data_profiling` on the
`../advanced_usage/available_settings`{.interpreted-text role="doc"}.

<img referrerpolicy="no-referrer-when-downgrade" src="https://static.scarf.sh/a.png?x-pxid=baa0e45f-0c03-4190-9646-9d8ea2640ba2" />