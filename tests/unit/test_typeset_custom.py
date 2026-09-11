import datetime

import numpy as np
import pandas as pd
import pytest
from visions.test.utils import (
    contains,
    convert,
    get_contains_cases,
    get_convert_cases,
    get_inference_cases,
    infers,
)

from data_profiling.config import Settings
from data_profiling.model.typeset import ProfilingTypeSet
from tests.unit.test_utils import patch_arg


def get_profiling_series():
    data = {
        "empty": pd.Series([], dtype=object),
        # Boolean
        "booleans_type": [False, True, True],
        "booleans_type_nan": [False, True, np.nan],
        "str_yes_no": ["Y", "N", "Y"],
        "str_yes_no_mixed": ["Y", "n", "y"],
        "str_yes_no_nan": ["Y", "N", np.nan],
        "str_true_false": ["True", "False", "False"],
        "str_true_false_nan": ["True", "False", np.nan],
        "str_true_false_none": ["True", "False", None],
        # Numeric
        "num_with_inf": [1, 2, 3, 6, np.inf],
        "integers": [1, 0, 0, 0],
        "inf_only": [np.inf],
        "integers_nan": [1, 0, 1, 0, np.nan],
        # test Describe
        "id": [chr(97 + c) for c in range(1, 9)] + ["d"],
        "catnum": [str(c) for c in range(1, 100)],
        "x": [50, 50, -10, 0, 0, 5, 15, -3, np.nan],
        "y": [
            0.000001,
            654.152,
            np.nan,
            15.984512,
            3122,
            -3.1415926535,
            111,
            15.9,
            13.5,
        ],
        "str": [
            "a",
            "long text value",
            "Élysée",
            "",
            None,
            "some <b> B.s </div> </div> HTML stuff",
            "c",
            "c",
            "c",
        ],
        "str_cat": pd.Series(
            ["male", "male", None, "female", "female", "male", "male"]
        ),
        "str_num": ["1", "10", "3.14", "566"],
        "str_date": ["2000/01/01", "2001/07/24", "2011/12/24", "1980/03/10"],
        "str_date2": ["2000-01-01", "2001-07-24", "2011-12-24", "1980-03-10"],
        "s1": np.ones(9),
        "s2": ["some constant text $ % value {obj} " for _ in range(1, 10)],
        "cat": pd.Series(
            ["male", "male", None, "female", "female", "male", "male"],
            dtype="category",
        ),
        "somedate": [
            datetime.date(2011, 7, 4),
            datetime.datetime(2022, 1, 1, 13, 57),
            datetime.datetime(1990, 12, 9),
            np.nan,
            datetime.datetime(1990, 12, 9),
            datetime.datetime(1950, 12, 9),
            datetime.datetime(1898, 1, 2),
            datetime.datetime(1950, 12, 9),
            datetime.datetime(1950, 12, 9),
        ],
        "bool_tf": [True, True, False, True, False, True, True, False, True],
        "bool_tf_with_nan": [
            True,
            False,
            False,
            False,
            False,
            True,
            True,
            False,
            np.nan,
        ],
        "bool_01": [1, 1, 0, 1, 1, 0, 0, 0, 1],
        "bool_01_with_nan": [1, 0, 1, 0, 0, 1, 1, 0, np.nan],
        "list": [
            [1, 2],
            [1, 2],
            [1, 2],
            [1, 2],
            [1, 2],
            [1, 2],
            [1, 2],
            [1, 2],
            [1, 2],
        ],
        "mixed": [1, 2, "a", 4, 5, 6, 7, 8, 9],
        "dict": [
            {"a": "a"},
            {"b": "b"},
            {"c": "c"},
            {"d": "d"},
            {"e": "e"},
            {"f": "f"},
            {"g": "g"},
            {"h": "h"},
            {"i": "i"},
        ],
        "tuple": [
            (1, 2),
            (3, 4),
            (5, 6),
            (7, 8),
            (9, 10),
            (11, 12),
            (13, 14),
            (15, 16),
            (17, 18),
        ],
        "date_str": ["2018-01-01", "2017-02-01", "2018-04-07"],
        "nullable_int": pd.Series([1, None], dtype="Int64"),
    }

    return {key: pd.Series(values, name=key) for key, values in data.items()}


series = get_profiling_series()

config = Settings()
config.vars.num.low_categorical_threshold = 0
my_typeset = ProfilingTypeSet(config)

type_map = {str(k): k for k in my_typeset.types}
Numeric = type_map["Numeric"]
Text = type_map["Text"]
Categorical = type_map["Categorical"]
Boolean = type_map["Boolean"]
DateTime = type_map["DateTime"]
Unsupported = type_map["Unsupported"]

config2 = Settings()
config2.vars.num.low_categorical_threshold = 2
typeset2 = ProfilingTypeSet(config2)
type_map2 = {str(k): k for k in typeset2.types}
Numeric2 = type_map2["Numeric"]
Text2 = type_map2["Text"]
Categorical2 = type_map2["Categorical"]
DateTime2 = type_map2["DateTime"]
Boolean2 = type_map2["Boolean"]


contains_map = {
    Numeric: {
        "x",
        "y",
        "s1",
        "num_with_inf",
        "integers",
        "integers_nan",
        "bool_01",
        "bool_01_with_nan",
        "inf_only",
        "nullable_int",
    },
    Text: {
        "str",
        "str_cat",
        "str_num",
        "str_date",
        "str_date2",
        "str_yes_no",
        "str_yes_no_mixed",
        "str_yes_no_nan",
        "str_true_false",
        "str_true_false_none",
        "str_true_false_nan",
        "id",
        "catnum",
        "date_str",
        "s2",
    },
    Categorical: {"cat"},
    Boolean: {
        "bool_tf",
        "bool_tf_with_nan",
        "booleans_type",
        "booleans_type_nan",
    },
    DateTime: {
        "somedate",
    },
    Unsupported: {"empty", "list", "mixed", "dict", "tuple"},
}


@pytest.mark.parametrize(
    **patch_arg(get_contains_cases(series, contains_map, my_typeset), "contains_type")
)
def test_contains(name, series, contains_type, member):
    """Test the generated combinations for "series in type"

    Args:
        series: the series to test
        contains_type: the type to test against
        member: the result
    """
    result, message = contains(name, series, contains_type, member)
    assert result, message


inference_map = {
    "x": Numeric,
    "y": Numeric,
    "s1": Numeric,
    "num_with_inf": Numeric,
    "integers": Numeric,
    "integers_nan": Numeric,
    "bool_01": Numeric,
    "bool_01_with_nan": Numeric,
    "id": Text,
    "str_cat": Categorical,
    "str_num": Numeric,
    "str_date": DateTime,
    "str_date2": DateTime,
    "s2": Categorical,
    "date_str": DateTime,
    "str": Text,
    "cat": Categorical,
    "bool_tf": Boolean,
    "bool_tf_with_nan": Boolean,
    "booleans_type": Boolean,
    "booleans_type_nan": Boolean,
    "str_yes_no": Boolean,
    "str_yes_no_mixed": Boolean,
    "str_yes_no_nan": Boolean,
    "str_true_false": Boolean,
    "str_true_false_none": Boolean,
    "str_true_false_nan": Boolean,
    "somedate": DateTime,
    "empty": Unsupported,
    "list": Unsupported,
    "mixed": Unsupported,
    "dict": Unsupported,
    "tuple": Unsupported,
    "inf_only": Numeric,
    "nullable_int": Numeric,
    "catnum": Numeric,
}


@pytest.mark.parametrize(
    **patch_arg(
        get_inference_cases(series, inference_map, my_typeset), "inference_type"
    )
)
def test_inference(name, series, inference_type, typeset, difference):
    """Test the generated combinations for "inference(series) == type_"

    Args:
        series: the series to test
        type_: the type to test against
    """
    result, message = infers(name, series, inference_type, typeset, difference)
    assert result, message


# Conversions in one single step
convert_map = [
    # Model type, Relation type
    (
        Categorical2,
        Numeric2,
        {
            "integers",
            "inf_only",
            "integers_nan",
            "s1",
            "bool_01",
            "bool_01_with_nan",
            "nullable_int",
        },
    ),
    (
        Boolean2,
        Text2,
        {
            "str_yes_no",
            "str_yes_no_mixed",
            "str_yes_no_nan",
            "str_true_false",
            "str_true_false_nan",
            "str_true_false_none",
        },
    ),
    (
        Categorical2,
        Text2,
        {
            "str_cat",
            "s2",
        },
    ),
    (Numeric2, Text2, {"str_num", "catnum"}),
    (DateTime2, Text2, {"str_date", "str_date2", "date_str"}),
]


@pytest.mark.parametrize(**get_convert_cases(series, convert_map, typeset2))
def test_conversion(name, source_type, relation_type, series, member):
    """Test the generated combinations for "convert(series) == type" and "infer(series) = source_type"

    Args:
        series: the series to test
        source_type: the type to test against
    """
    result, message = convert(name, source_type, relation_type, series, member)
    assert result, message
