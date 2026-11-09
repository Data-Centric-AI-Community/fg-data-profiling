# Profiling large datasets

<img referrerpolicy="no-referrer-when-downgrade" src="https://static.scarf.sh/a.png?x-pxid=baa0e45f-0c03-4190-9646-9d8ea2640ba2" />

By default, `data-profiling` comprehensively summarizes the input
dataset in a way that gives the most insights for data analysis. For
small datasets, these computations can be performed in *quasi*
real-time. For larger datasets, deciding upfront which calculations to
make might be required. Whether a computation scales to a large datasets
not only depends on the exact size of the dataset, but also on its
complexity and on whether fast computations are available. If the
computation time of the profiling becomes a bottleneck,
`data-profiling` offers several alternatives to overcome it.

## Pyspark

[Spark](https://spark.apache.org/)

!!! Note "Minimal mode"

    This mode was introduced in version v4.0.0


`data-profiling` now supports Spark Dataframes profiling. You can find
an example of the integration
[here](https://github.com/Data-Centric-AI-Community/data-profiling/blob/master/examples/features/spark_example.py).

**Features supported:** - Univariate variables' analysis - Head and Tail
dataset sample - Correlation matrices: Pearson and Spearman

*Coming soon* - Missing values analysis - Interactions - Improved
histogram computation

Keep an eye on the
[GitHub](https://github.com/Data-Centric-AI-Community/data-profiling/issues) page to
follow the updates on the implementation of [Pyspark Dataframes
support](https://github.com/orgs/ydataai/projects/16/views/2).

## Minimal mode

!!! Note "**Minimal mode**"

    This mode was introduced in version v2.4.0

`data-profiling` includes a minimal configuration file where the most
expensive computations are turned off by default. This is the
recommended starting point for larger datasets.

``` python
profile = ProfileReport(large_dataset, minimal=True)
profile.to_file("output.html")
```

This configuration file can be found here:
[config_minimal.yaml](https://github.com/Data-Centric-AI-Community/data-profiling/blob/master/src/data_profiling/config_minimal.yaml).
More details on settings and configuration are available in
`../advanced_usage/available_settings`{.interpreted-text role="doc"}.

## Sample the dataset

An alternative way to handle really large datasets is to use a portion
of it to generate the profiling report. Several users report this is a
good way to scale back the computation time while maintaining
representativity.

<blockquote class="twitter-tweet" align="center">
    <p lang="en" dir="ltr">pandas-profiling is a nifty tool to compute descriptive statistics on a dataframe and issue warnings on columns with many missing values, high skew, high cardinality categorical values, high correlation, etc: <a href="https://t.co/57IezPW1nI">https://t.co/57IezPW1nI</a> demo notebook: <a href="https://t.co/JpqTO9FK1p">https://t.co/JpqTO9FK1p</a></p>&mdash; Olivier Grisel (@ogrisel) <a href="https://twitter.com/ogrisel/status/951425284963733505?ref_src=twsrc%5Etfw">January 11, 2018</a></blockquote> <script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>

```python linenums="1" title="Sampling a large dataset"
# Sample 10.000 rows
sample = large_dataset.sample(10000)

profile = ProfileReport(sample, minimal=True)
profile.to_file("output.html")
```

The reader of the report might want to know that the profile is
generated using a sample from the data. This can be done by adding a
description to the report (see `metadata`{.interpreted-text role="doc"}
for details).

``` python linenums="1" title="Sample 5% of your dataset"
description = "Disclaimer: this profiling report was generated using a sample of 5% of the original dataset."
sample = large_dataset.sample(frac=0.05)

profile = sample.profile_report(dataset={"description": description}, minimal=True)
profile.to_file("output.html")
```

## Disable expensive computations

To decrease the computational burden in particularly large datasets but
still maintain some information of interest that may stem from them,
some computations can be filtered only for certain columns.
Particularly, a list of targets can be provided to **Interactions**, so
that only the interactions with these variables in specific are
computed.

``` python linenums="1" title="Disable expensive computations"
from data_profiling import ProfileReport
import pandas as pd

# Reading the data
data = pd.read_csv(
    "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
)

# Creating the profile without specifying the data source, to allow editing the configuration
profile = ProfileReport()
profile.config.interactions.targets = ["Name", "Sex", "Age"]

# Assigning a DataFrame and exporting to a file, triggering computation
profile.df = data
profile.to_file("report.html")
```

The setting controlling this, `ìnteractions.targets`, can be changed via
multiple interfaces (configuration files or environment variables). For
details, see `../advanced_usage/changing_settings`{.interpreted-text
role="doc"}.

# Concurrency

`data-profiling` is a project under active development. One of the
highly desired features is the addition of a scalable backend such as
[Modin](https://github.com/modin-project/modin) or
[Dask](https://dask.org/).

Keep an eye on the
[GitHub](https://github.com/Data-Centric-AI-Community/data-profiling/issues) page to
follow the updates on the implementation of a concurrent and highly
scalable backend. Specifically, development of a Spark backend is
[currently
underway](https://github.com/Data-Centric-AI-Community/data-profiling/projects/3).
