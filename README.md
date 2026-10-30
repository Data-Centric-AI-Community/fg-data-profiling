# fg-data-profiling

> **`ydata-profiling` is now `fg-data-profiling`.** This package has been renamed to `fg-data-profiling`. Please follow the [Migration Guide](#migration-guide) as soon as possible — the old package will no longer receive updates or bug fixes.

[![Build Status](https://github.com/ydataai/pandas-profiling/actions/workflows/tests.yml/badge.svg?branch=master)](https://github.com/ydataai/pandas-profiling/actions/workflows/tests.yml)
[![PyPI download month](https://img.shields.io/pypi/dm/fg-data-profiling.svg)](https://pypi.python.org/pypi/fg-data-profiling/)
[![](https://pepy.tech/badge/pandas-profiling)](https://pypi.org/project/fg-data-profiling/)
[![Code Coverage](https://codecov.io/gh/ydataai/pandas-profiling/branch/master/graph/badge.svg?token=gMptB4YUnF)](https://codecov.io/gh/ydataai/pandas-profiling)
[![Release Version](https://img.shields.io/github/release/ydataai/pandas-profiling.svg)](https://github.com/ydataai/pandas-profiling/releases)
[![Python Version](https://img.shields.io/pypi/pyversions/fg-data-profiling)](https://pypi.org/project/fg-data-profiling/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/python/black)
<img referrerpolicy="no-referrer-when-downgrade" src="https://static.scarf.sh/a.png?x-pxid=cb7e69df-af81-4352-809a-d4251756affc" />

<p align="center"><img width="300" src="https://assets.ydata.ai/oss/ydata-profiling_black.png" alt="Data Profiling Logo"></p>

<p align="center">
  <a href="https://ydata-profiling.ydata.ai/docs/master/">Documentation</a>
  |
  <a href="https://tiny.ydata.ai/dcai-data-profiling">Discord</a>
  | 
  <a href="https://stackoverflow.com/questions/tagged/pandas-profiling+or+data-profiling">Stack Overflow</a>
  |
  <a href="https://ydata-profiling.ydata.ai/docs/master/pages/reference/changelog.html#changelog">Latest changelog</a>

</p>

<p align="center">
  Do you like this project? Show us your love and <a href="https://engage.ydata.ai">give feedback!</a>
</p>

`fg-data-profiling` primary goal is to provide a one-line Exploratory Data Analysis (EDA) experience in a consistent and fast solution. Like pandas `df.describe()` function, that is so handy, fg-data-profiling delivers an extended analysis of a DataFrame while allowing the data analysis to be exported in different formats such as **html** and **json**.

The package outputs a simple and digested analysis of a dataset, including **time-series** and **text**.

> **Looking for a scalable solution that can fully integrate with your database systems?**<br>
> Leverage YData Fabric Data Catalog to connect to different databases and storages (Oracle, snowflake, PostGreSQL, GCS, S3, etc.) and leverage an interactive and guided profiling experience in Fabric. Check out the [Community Version](http://ydata.ai/register?utm_source=data-profiling&utm_medium=documentation&utm_campaign=YData%20Fabric%20Community).

## Migration Guide
 
### 1. Uninstall the old package
 
```bash
pip uninstall ydata-profiling
```
 
### 2. Install the new package
 
```bash
pip install fg-data-profiling
```
 
### 3. Update your imports
 
Find and replace all occurrences of the old import in your codebase:
 
```python
# Before
import ydata_profiling
from ydata_profiling import ProfileReport

# After
import data_profiling
from data_profiling import ProfileReport
```
 
You can use this one-liner to find all affected files:
 
```bash
grep -r "ydata_profiling" . --include="*.py"
```

## ▶️ Quickstart

### Install
```cmd
pip install fg-data-profiling
```
or
```cmd
conda install -c conda-forge fg-data-profiling
```
### Start profiling

Start by loading your pandas `DataFrame` as you normally would, e.g. by using:

```python
import numpy as np
import pandas as pd
from data_profiling import ProfileReport

df = pd.DataFrame(np.random.rand(100, 5), columns=["a", "b", "c", "d", "e"])
```

To generate the standard profiling report, merely run:

```python
profile = ProfileReport(df, title="Profiling Report")
```

## 📊 Key features

- **Type inference**: automatic detection of columns' data types (*Categorical*, *Numerical*, *Date*, etc.)
- **Warnings**: A summary of the problems/challenges in the data that you might need to work on (*missing data*, *inaccuracies*, *skewness*, etc.)
- **Univariate analysis**: including descriptive statistics (mean, median, mode, etc) and informative visualizations such as distribution histograms
- **Multivariate analysis**: including correlations, a detailed analysis of missing data, duplicate rows, and visual support for variables pairwise interaction
- **Time-Series**: including different statistical information relative to time dependent data such as auto-correlation and seasonality, along ACF and PACF plots.
- **Text analysis**: most common categories (uppercase, lowercase, separator), scripts (Latin, Cyrillic) and blocks (ASCII, Cyrilic)
- **File and Image analysis**: file sizes, creation dates, dimensions, indication of truncated images and existence of EXIF metadata
- **Compare datasets**: one-line solution to enable a fast and complete report on the comparison of datasets
- **Flexible output formats**: all analysis can be exported to an HTML report that can be easily shared with different parties, as JSON for an easy integration in automated systems and as a widget in a Jupyter Notebook.

The report contains three additional sections:

- **Overview**: mostly global details about the dataset (number of records, number of variables, overall missigness and duplicates, memory footprint)
- **Alerts**: a comprehensive and automatic list of potential data quality issues (high correlation, skewness, uniformity, zeros, missing values, constant values, between others)
- **Reproduction**: technical details about the analysis (time, version and configuration)

### 🎁 Latest features

- Want to scale? Check the latest release with ⭐⚡[Spark support](https://ydata-profiling.ydata.ai/docs/master/pages/integrations/pypspark.html)! 
- Looking for how you can do an EDA for Time-Series 🕛 ? Check [this blogpost](https://towardsdatascience.com/how-to-do-an-eda-for-time-series-cbb92b3b1913).
- You want to compare 2 datasets and get a report? Check [this blogpost](https://medium.com/towards-artificial-intelligence/how-to-compare-2-dataset-with-pandas-profiling-2ae3a9d7695e)

### ✨ Spark

Spark support has been released, but we are always looking for an extra pair of hands 👐.
[Check current work in progress!](https://github.com/Data-Centric-AI-Community/fg-data-profiling/projects/3).

## 📝 Use cases
fg-data-profiling can be used to deliver a variety of different use-case. The documentation includes guides, tips and tricks for tackling them:

| Use case | Description                                                                                 |
|----------|---------------------------------------------------------------------------------------------|
| [Comparing datasets](https://docs.profiling.ydata.ai/latest/features/comparing_datasets)                        | Comparing multiple version of the same dataset                                              |
| [Profiling a Time-Series dataset](https://docs.profiling.ydata.ai/latest/features/time_series_datasets)               | Generating a report for a time-series dataset with a single line of code                    |
|[Profiling large datasets](https://docs.profiling.ydata.ai/latest/features/big_data)                            | Tips on how to prepare data and configure `fg-data-profiling` for working with large datasets |
| [Handling sensitive data](https://docs.profiling.ydata.ai/latest/features/sensitive_data)                       | Generating reports which are mindful about sensitive data in the input dataset              |
| [Dataset metadata and data dictionaries](https://docs.profiling.ydata.ai/latest/features/metadata)               | Complementing the report with dataset details and column-specific data dictionaries         |
| [Customizing the report's appearance](https://docs.profiling.ydata.ai/latest/features/custom_reports) | Changing the appearance of the report's page and of the contained visualizations            |
| [Profiling Databases](https://docs.profiling.ydata.ai/latest/features/collaborative_data_profiling) | For a seamless profiling experience in your organization's databases, check [Fabric Data Catalog](https://ydata.ai/products/data_catalog), which allows to consume data from different types of storages such as RDBMs (Azure SQL, PostGreSQL, Oracle, etc.) and object storages (Google Cloud Storage, AWS S3, Snowflake, etc.), among others. |
### Using inside Jupyter Notebooks

There are two interfaces to consume the report inside a Jupyter notebook: through widgets and through an embedded HTML report.

<img alt="Notebook Widgets" src="https://data-profiling.ydata.ai/docs/master/assets/widgets.gif" width="800" />

The above is achieved by simply displaying the report as a set of widgets. In a Jupyter Notebook, run:

```python
profile.to_widgets()
```

The HTML report can be directly embedded in a cell in a similar fashion:

```python
profile.to_notebook_iframe()
```

<img alt="HTML" src="https://ydata-profiling.ydata.ai/docs/master/assets/iframe.gif" width="800" />

### Exporting the report to a file

To generate a HTML report file, save the `ProfileReport` to an object and use the `to_file()` function:

```python
profile.to_file("your_report.html")
```

Alternatively, the report's data can be obtained as a JSON file:

```python
# As a JSON string
json_data = profile.to_json()

# As a file
profile.to_file("your_report.json")
```

### Using in the command line

For standard formatted CSV files (which can be read directly by pandas without additional settings), the `data_profiling` executable can be used in the command line. The example below generates a report named *Example Profiling Report*, using a configuration file called `default.yaml`, in the file `report.html` by processing a `data.csv` dataset.

```sh
data_profiling --title "Example Profiling Report" --config_file default.yaml data.csv report.html
```

Additional details on the CLI are available [on the documentation](https://ydata-profiling.ydata.ai/docs/master/pages/getting_started/quickstart.html#command-line-usage).

## 👀 Examples

The following example reports showcase the potentialities of the package across a wide range of dataset and data types:

* [Census Income](https://ydata-profiling.ydata.ai/examples/master/census/census_report.html) (US Adult Census data relating income with other demographic properties)
* [NASA Meteorites](https://ydata-profiling.ydata.ai/examples/master/meteorites/meteorites_report.html) (comprehensive set of meteorite landing - object properties and locations) [![Open In Colab](https://camo.githubusercontent.com/52feade06f2fecbf006889a904d221e6a730c194/68747470733a2f2f636f6c61622e72657365617263682e676f6f676c652e636f6d2f6173736574732f636f6c61622d62616467652e737667)](https://colab.research.google.com/github/ydataai/pandas-profiling/blob/master/examples/meteorites/meteorites_cloud.ipynb) [![Binder](https://camo.githubusercontent.com/483bae47a175c24dfbfc57390edd8b6982ac5fb3/68747470733a2f2f6d7962696e6465722e6f72672f62616467655f6c6f676f2e737667)](https://mybinder.org/v2/gh/ydataai/pandas-profiling/master?filepath=examples%2Fmeteorites%2Fmeteorites%5Fcloud.ipynb)
* [Titanic](https://ydata-profiling.ydata.ai/examples/master/titanic/titanic_report.html) (the "Wonderwall" of datasets) [![Open In Colab](https://camo.githubusercontent.com/52feade06f2fecbf006889a904d221e6a730c194/68747470733a2f2f636f6c61622e72657365617263682e676f6f676c652e636f6d2f6173736574732f636f6c61622d62616467652e737667)](https://colab.research.google.com/github/ydataai/pandas-profiling/blob/master/examples/titanic/titanic_cloud.ipynb) [![Binder](https://camo.githubusercontent.com/483bae47a175c24dfbfc57390edd8b6982ac5fb3/68747470733a2f2f6d7962696e6465722e6f72672f62616467655f6c6f676f2e737667)](https://mybinder.org/v2/gh/ydataai/pandas-profiling/master?filepath=examples%2Ftitanic%2Ftitanic%5Fcloud.ipynb)
* [NZA](https://ydata-profiling.ydata.ai/examples/master/nza/nza_report.html) (open data from the Dutch Healthcare Authority)
* [Stata Auto](https://ydata-profiling.ydata.ai/examples/master/stata_auto/stata_auto_report.html) (1978 Automobile data)
* [Colors](https://ydata-profiling.ydata.ai/examples/master/colors/colors_report.html) (a simple colors dataset)
* [Vektis](https://ydata-profiling.ydata.ai/examples/master/vektis/vektis_report.html) (Vektis Dutch Healthcare data)
* [UCI Bank Dataset](https://ydata-profiling.ydata.ai/examples/master/bank_marketing_data/uci_bank_marketing_report.html) (marketing dataset from a bank)
* [Russian Vocabulary](https://ydata-profiling.ydata.ai/examples/master/features/russian_vocabulary.html) (100 most common Russian words, showcasing unicode text analysis)
* [Website Inaccessibility](https://ydata-profiling.ydata.ai/examples/master/features/website_inaccessibility_report.html) (website accessibility analysis, showcasing support for URL data)
* [Orange prices](https://ydata-profiling.ydata.ai/examples/master/features/united_report.html) and 
* [Coal prices](https://ydata-profiling.ydata.ai/examples/master/features/flatly_report.html) (simple pricing evolution datasets, showcasing the theming options)
* [USA Air Quality](https://github.com/Data-Centric-AI-Community/fg-data-profiling/tree/master/examples/usaairquality) (Time-series air quality dataset EDA example)
* [HCC](https://github.com/Data-Centric-AI-Community/fg-data-profiling/tree/master/examples/hcc) (Open dataset from healthcare, showcasing compare between two sets of data, before and after preprocessing)

## 🛠️ Installation
Additional details, including information about widget support, are available [on the documentation](https://ydata-profiling.ydata.ai/docs/master/pages/getting_started/installation.html).

### Using pip
[![PyPi Downloads](https://pepy.tech/badge/fg-data-profiling)](https://pepy.tech/project/fg-data-profiling)
[![PyPi Monthly Downloads](https://pepy.tech/badge/pandas-profiling/month)](https://pepy.tech/project/fg-data-profiling/month)
[![PyPi Version](https://badge.fury.io/py/fg-data-profiling.svg)](https://pypi.org/project/fg-data-profiling/)

You can install using the `pip` package manager by running:

```sh
pip install -U fg-data-profiling
```

#### Extras

The package declares "extras", sets of additional dependencies.

* `[notebook]`: support for rendering the report in Jupyter notebook widgets.
* `[unicode]`: support for more detailed Unicode analysis, at the expense of additional disk space.
* `[spark]`: support for pyspark for big dataset analysis

Install these with e.g.

```sh
pip install -U "fg-data-profiling[notebook,unicode,spark]"
```


### Using conda
[![Conda Downloads](https://img.shields.io/conda/dn/conda-forge/pandas-profiling.svg)](https://anaconda.org/conda-forge/pandas-profiling)
[![Conda Version](https://img.shields.io/conda/vn/conda-forge/pandas-profiling.svg)](https://anaconda.org/conda-forge/pandas-profiling) 


You can install using the `conda` package manager by running:

```sh
conda install -c conda-forge fg-data-profiling
```

### From source (development)

Download the source code by cloning the repository or click on [Download ZIP](https://github.com/ydataai/pandas-profiling/archive/master.zip) to download the latest stable version.

Install it by navigating to the proper directory and running:

```sh
pip install -e .
```

The profiling report is written in HTML and CSS, which means a modern browser is required. 

You need [Python 3](https://python3statement.github.io/) to run the package. Other dependencies can be found in the requirements files:

| Filename | Requirements|
|----------|-------------|
| [requirements.txt](https://github.com/ydataai/pandas-profiling/blob/master/requirements.txt) | Package requirements|
| [requirements-dev.txt](https://github.com/ydataai/pandas-profiling/blob/master/requirements-dev.txt)  |  Requirements for development|
| [requirements-test.txt](https://github.com/ydataai/pandas-profiling/blob/master/requirements-test.txt) | Requirements for testing|
| [setup.py](https://github.com/ydataai/pandas-profiling/blob/master/setup.py) | Requirements for widgets etc. |

## 🔗 Integrations

To maximize its usefulness in real world contexts, `fg-data-profiling` has a set of implicit and explicit integrations with a variety of other actors in the Data Science ecosystem: 

| Integration type | Description |
|---|---|
| [Other DataFrame libraries](https://docs.profiling.ydata.ai/latest/integrations/other_dataframe_libraries) | How to compute the profiling of data stored in libraries other than pandas |
| [Great Expectations](https://ydata-profiling.ydata.ai/docs/master/pages/integrations/great_expectations.html) | Generating [Great Expectations](https://greatexpectations.io) expectations suites directly from a profiling report |
| [Interactive applications](https://docs.profiling.ydata.ai/latest/integrations/interactive_applications) | Embedding profiling reports in [Streamlit](http://streamlit.io), [Dash](http://dash.plotly.com) or [Panel](https://panel.holoviz.org) applications |
| [Pipelines](https://ydata-profiling.ydata.ai/docs/master/pages/integrations/pipelines.html) | Integration with DAG workflow execution tools like [Airflow](https://airflow.apache.org) or [Kedro](https://kedro.org) |
| [Cloud services](https://ydata-profiling.ydata.ai/docs/master/pages/integrations/cloud_services.html) | Using `fg-data-profiling` in hosted computation services like [Lambda](https://lambdalabs.com), [Google Cloud](https://github.com/GoogleCloudPlatform/analytics-componentized-patterns/blob/master/retail/propensity-model/bqml/bqml_kfp_retail_propensity_to_purchase.ipynb) or [Kaggle](https://www.kaggle.com/code) |
| [IDEs](https://ydata-profiling.ydata.ai/docs/master/pages/integrations/ides.html) | Using `fg-data-profiling` directly from integrated development environments such as [PyCharm](https://www.jetbrains.com/pycharm/) |

## 🙋 Support
Need help? Want to share a perspective? Report a bug? Ideas for collaborations? Reach out via the following channels:

- [Stack Overflow](https://stackoverflow.com/questions/tagged/pandas-profiling+or+data-profiling): ideal for asking questions on how to use the package
- [GitHub Issues](https://github.com/Data-Centric-AI-Community/fg-data-profiling/issues): bugs, proposals for changes, feature requests
- [Discord](https://tiny.ydata.ai/dcai-data-profiling): ideal for projects discussions, ask questions, collaborations, general chat

> **Need Help?**<br>
> Get your questions answered with a product owner by [booking a Pawsome chat](https://meetings.hubspot.com/fabiana-clemente)! 🐼

> ❗ Before reporting an issue on GitHub, check out [Common Issues](https://docs.profiling.ydata.ai/latest/support-contribution/common_issues).

## 🤝🏽 Contributing
Learn how to get involved in the [Contribution Guide](https://ydata-profiling.ydata.ai/docs/master/pages/support_contrib/contribution_guidelines.html).

A low-threshold place to ask questions or start contributing is the [Data Centric AI Community's Discord](https://tiny.ydata.ai/dcai-data-profiling).


A big thank you to all our amazing contributors! 

<a href="https://github.com/Data-Centric-AI-Community/fg-data-profiling/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=Data-Centric-AI-Community/fg-data-profiling" />
</a>

Contributors wall made with [contrib.rocks](https://contrib.rocks).
