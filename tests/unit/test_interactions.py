import re

import numpy as np
import pandas as pd
import pytest

from data_profiling import ProfileReport
from data_profiling.config import Settings
from data_profiling.report.presentation.core import Container
from data_profiling.report.presentation.flavours import HTMLReport
from data_profiling.report.structure.report import get_interactions


def test_interactions_target():
    n_rows = 10
    n_columns = 50
    n_targets = 2

    df = pd.DataFrame(
        np.random.randint(0, 1000, size=(n_rows, n_columns)),
        columns=[f"column_{c}" for c in range(n_columns)],
    )
    targets = [f"column_{target}" for target in range(0, n_targets)]

    profile = ProfileReport(
        df, minimal=True, interactions={"continuous": True, "targets": targets}
    )

    total = sum(len(v.keys()) for k, v in profile.get_description().scatter.items())
    assert total == n_targets * n_columns


@pytest.mark.parametrize("comparison", [False, True])
def test_interaction_targets_are_unique_for_colliding_column_names(comparison):
    config = Settings()
    config.html.style._labels = ["first", "second"]
    columns = ["x", "a b", "a-b"]
    plot = ["<svg></svg>"] * 2 if comparison else "<svg></svg>"
    interactions = {x: {y: plot for y in columns} for x in columns}
    items = get_interactions(config, interactions)
    assert [item.name for item in items] == columns
    assert all(
        [child.name for child in item.content["items"]] == columns for item in items
    )
    html = HTMLReport(
        Container(items, sequence_type="tabs", anchor_id="interactions")
    ).render()
    ids = re.findall(r'\bid="([^"]+)"', html)
    targets = re.findall(r'data-bs-target="#([^"]+)"', html)
    assert len(ids) == len(set(ids))
    assert len(targets) == len(set(targets))
    assert all(ids.count(target) == 1 for target in targets)
