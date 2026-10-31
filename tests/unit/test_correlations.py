import numpy as np
import pandas as pd
import pytest

from data_profiling import ProfileReport
from data_profiling.model.alerts import check_correlation_alerts
from data_profiling.report.presentation.core import CorrelationTable, Image
from data_profiling.report.structure.correlations import get_correlation_items


@pytest.mark.skip
def generate_df(n: int):
    return pd.DataFrame(
        {
            "num_int": np.random.randint(0, 20, n),
            "num_float": 100 * np.random.normal(0, 0.1, n),
            "cat_str": np.random.choice(
                a=["CAT_A", "CAT_B", "CAT_C"], size=n, p=[0.5, 0.3, 0.2]
            ),
            "cat_int": np.random.choice(
                a=[1, 2, 5, 10], size=n, p=[0.4, 0.3, 0.2, 0.1]
            ),
        }
    )


@pytest.mark.skip
def generate_report(correlation_table: bool):
    df = generate_df(300)

    correlations = {
        "auto": {"calculate": True},
        "pearson": {"calculate": True},
        "spearman": {"calculate": True},
        "kendall": {"calculate": True},
        "phi_k": {"calculate": True},
        "cramers": {"calculate": True},
    }

    return ProfileReport(
        df,
        title="Profiling Report",
        correlations=correlations,
        correlation_table=correlation_table,
    )


def test_standard_report_with_correlation_table():
    report = generate_report(correlation_table=True)
    renderable = get_correlation_items(report.config, report.description_set)
    for cor_item in renderable.content["items"]:
        diagram, table = cor_item.content["items"]
        assert isinstance(table, CorrelationTable)
        assert isinstance(diagram, Image)


def test_standard_report_without_correlation_table():
    report = generate_report(correlation_table=False)
    renderable = get_correlation_items(report.config, report.description_set)
    for diagram in renderable.content["items"]:
        assert isinstance(diagram, Image)


def test_compare_report_with_correlation_table():
    report1 = generate_report(correlation_table=True)
    report2 = generate_report(correlation_table=True)
    comp_report = report1.compare(report2)
    renderable = get_correlation_items(comp_report.config, comp_report.description_set)
    for cor_items in renderable.content["items"]:
        diagrams, tables = cor_items.content["items"]
        for table in tables.content["items"]:
            assert isinstance(table, CorrelationTable)
        for diagram in diagrams.content["items"]:
            assert isinstance(diagram, Image)


def test_compare_report_without_correlation_table():
    report1 = generate_report(correlation_table=False)
    report2 = generate_report(correlation_table=False)
    comp_report = report1.compare(report2)
    renderable = get_correlation_items(comp_report.config, comp_report.description_set)
    for cor_items in renderable.content["items"]:
        for diagram in cor_items.content["items"]:
            assert isinstance(diagram, Image)


def test_correlation_alerts_are_merged_across_methods(config):
    columns = ["x", "y", "z"]
    correlations = {
        "pearson": pd.DataFrame(
            [[1.0, 0.95, 0.95], [0.95, 1.0, 0.95], [0.95, 0.95, 1.0]],
            index=columns,
            columns=columns,
        ),
        "auto": pd.DataFrame(
            [[1.0, 0.1, 0.95], [0.1, 1.0, 0.1], [0.95, 0.1, 1.0]],
            index=columns,
            columns=columns,
        ),
    }

    alerts = check_correlation_alerts(config, correlations)

    fields = {alert.column_name: alert.values["fields"] for alert in alerts}
    assert fields == {"x": ["y", "z"], "y": ["x", "z"], "z": ["x", "y"]}
