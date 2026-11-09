# ⚡ Pyspark

!!! note ""Spark support"
    **Spark dataframes support** - Spark Dataframes profiling is available
    from data-profiling version 4.0.0 onwards

Data Profiling is a core step in the process of developing AI solutions.
For small datasets, the data can be loaded into memory and easily
accessed with Python and pandas dataframes. However, for larger datasets
what can be done?

Big data engines, that distribute the workload through different
machines, are the answer. Particularly, Spark rose as one of the most
used and adopted engines by the data community. `data-profiling`
provides an ease-to-use interface to generate complete and comprehensive
data profiling out of your Spark dataframes with a single line of code.

Getting started 

## Installing Pyspark for Linux and Windows 

!!! tip
    
    Ensure that you first install the system requirements (spark and java).

    :   -   Go to [Download Java
            JDK](https://www.oracle.com/java/technologies/javase-jdk13-downloads.html)
            and download the Java Development Kit (JDK).
        -   Download and install a [Spark version bigger than
            3.3](https://spark.apache.org/downloads.html)
        -   Set your environment variables

``` console
export SPARK_VERSION=3.3.0
export SPARK_DIRECTORY=/opt/spark
export HADOOP_VERSION=2.7
mkdir -p ${SPARK_DIRECTORY}
sudo apt-get update
sudo apt-get -y install openjdk-8-jdk
curl https://archive.apache.org/dist/spark/spark-${SPARK_VERSION}/spark-${SPARK_VERSION}-bin-hadoop${HADOOP_VERSION}.tgz \
--output ${SPARK_DIRECTORY}/spark.tgz
cd ${SPARK_DIRECTORY} && tar -xvzf spark.tgz && mv spark-${SPARK_VERSION}-bin-hadoop${HADOOP_VERSION} sparkenv
```

A more detailed tutorial for the installation can be found
[here](https://www.datacamp.com/tutorial/installation-of-pyspark).

## Installing Pyspark for MacOS

Use `Homebrew` to ensure that the system requirements are
installed (java and scala (optional)) 

```console
console brew
install <openjdk@11>
```

``` console
#Install scala is optional
brew install scala
```

### Install pyspark 

```console 
brew install apache-spark
```

After successful installation of Apache Spark run pyspark from the
command line to launch PySpark shell and confirm both python and pyspark
versions. A more detailed tutorial for the installation can be found
[here](https://sparkbyexamples.com/pyspark/how-to-install-pyspark-on-mac/)

### Install data-profiling

Create a pip virtual environment or a conda environment and install `fg-data-profiling` with pyspark as a dependency

```console
pip install "fg-data-profiling[spark]"
```

## Profiling with Spark - Supported Features

!!! note "Minimal mode"
    This mode was introduced in version v4.0.0

`data-profiling` now supports Spark Dataframes profiling. You can find
an example of the integration
[here](https://github.com/Data-Centric-AI-Community/data-profiling/blob/master/examples/features/spark_example.py).

**Features supported:**

:   -   Univariate variables' analysis
    -   Head and Tail dataset sample
    -   Correlation matrices: Pearson and Spearman

*Coming soon*

:   -   Missing values analysis
    -   Interactions
    -   Improved histogram computation

## Profiling with Spark DataFrames

A quickstart example to profile data from a CSV leveraging Pyspark
engine and `data-profiling`.

``` python linenums="1" title="Profiling with Spark Dataframes"
from pyspark.sql import SparkSession
from data_profiling import ProfileReport

spark = (
    SparkSession.builder.master("local[1]")
    .appName("SparkByExamples.com")
    .getOrCreate()
)

df = spark.read.csv("{insert-file-path}")

df.printSchema()

a = ProfileReport(df)
a.to_file("spark_profile.html")
```

### data-profiling in Databricks

Yes! We have fantastic new coming with a full tutorial on how you can
use data-profiling in Databricks Notebooks.

The notebook example can be found
[here](https://github.com/Data-Centric-AI-Community/data-profiling/tree/master/examples/integrations/databricks).

Stay tuned - we are going to update the documentation soon!

<img referrerpolicy="no-referrer-when-downgrade" src="https://static.scarf.sh/a.png?x-pxid=c3c3ddfc-4c47-4192-9105-5dac1ffa3a49" />