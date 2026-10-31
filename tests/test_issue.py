import numpy as np
import pandas as pd
import pytest
from data_profiling import ProfileReport


def test_extreme_memory_usage_issue_1749():
    """
    Regression test for Issue #1749.
    Ensures that extreme outliers (e.g., 1e16) do not cause a MemoryError.
    """
    data = np.random.uniform(size=6)
    data[0] = 1e16
    df = pd.DataFrame(dict(a=data))

    try:
        # Пробуем сгенерировать структуру отчета
        report = ProfileReport(df, tsmode=False, lazy=False)
        report.get_description()
    except MemoryError:
        # Если вдруг ошибка вернется, тест об этом сообщит
        pytest.fail("ProfileReport raised MemoryError on extreme data values (Issue #1749)")