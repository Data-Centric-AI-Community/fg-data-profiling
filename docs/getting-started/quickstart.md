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

The HTML file is self-contained, so it can be opened locally or sent as an
attachment. To share it as a link instead, any static host works. For
example, [deed.page](https://deed.page) takes the file over HTTP without an
account:

``` bash
curl -T your_report.html "https://deed.page/v1/sites?slug=your-report-<suffix>"
```

Replace `<suffix>` with your own random string. The JSON response contains
the public URL (`https://your-report-<suffix>.deed.page/`) and an owner token
that is shown only once: repeating the request with
`-H "Authorization: Bearer <token>"` replaces the report, and a `DELETE`
request to `https://deed.page/v1/sites/your-report-<suffix>` with the same
header takes it down. Anyone with the link can see the report, including the
sample rows it shows by default, so see
[Sensitive data](../features/sensitive_data.md) before publishing reports
built from private datasets.

Alternatively, the report's data can be obtained as a JSON file:

``` python linenums="1" title="Save your profile report as a JSON file"
# As a JSON string
json_data = profile.to_json()

# As a file
profile.to_file("your_report.json")
```

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