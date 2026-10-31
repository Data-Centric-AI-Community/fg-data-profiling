# Pipelines

With Python, command-line and Jupyter interfaces, `data-profiling`
integrates seamlessly with DAG execution tools like Airflow, Dagster,
Kedro and Prefect, allowing it to easily becomes a building block of
data ingestion and analysis pipelines. Integration with
[Dagster](https://github.com/dagster-io/dagster) or
[Prefect](https://github.com/prefecthq/prefect) can be achieved in a
similar way as with Airflow.

## Airflow

Integration with Airflow can be easily achieved through the
[BashOperator](https://airflow.apache.org/docs/stable/_api/airflow/operators/bash_operator/index.html)
or the
[PythonOperator](https://airflow.apache.org/docs/stable/_api/airflow/operators/python_operator/index.html#airflow.operators.python_operator.PythonOperator).

``` python linenums="1" title="data-profiling with Airflow"
# Using the command line interface
profiling_task = BashOperator(
    task_id="Profile Data",
    bash_command="pandas_profiling dataset.csv report.html",
    dag=dag,
)
```

``` python linenums="1" title="data-profiling with Airflow"
# Using the Python interface
import data_profiling

def profile_data(file_name, report_file):
    df = pd.read_csv(file_name)
    report = data_profiling.ProfileReport(df, title="Profiling Report in Airflow")
    report.to_file(report_file)

    return "Report generated at {}".format(report_file)


profiling_task2 = PythonOperator(
    task_id="Profile Data",
    op_kwargs={"file_name": "dataset.csv", "report_file": "report.html"},
    python_callable=profile_data,
    dag=dag,
)
```

## Kedro

There is a community created [Kedro
plugin](https://github.com/BrickFrog/kedro-pandas-profiling) available.

<img referrerpolicy="no-referrer-when-downgrade" src="https://static.scarf.sh/a.png?x-pxid=2d9fcb48-0fa6-43d8-a0ba-fdad344050fd" />