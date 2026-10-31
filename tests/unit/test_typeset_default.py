import os
from typing import Dict, Sequence
from urllib.parse import urlparse

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
from data_profiling.profile_report import ProfileReport
from tests.unit.test_utils import patch_arg


def get_sequences() -> Dict[str, Sequence]:
    sequences = {
        "complex_series_float": [
            complex(0, 0),
            complex(1, 0),
            complex(3, 0),
            complex(-1, 0),
        ],
        "url_nan_series": [
            urlparse("http://www.cwi.nl:80/%7Eguido/Python.html"),
            urlparse("https://github.com/dylan-profiling/hurricane"),
            np.nan,
        ],
        "mixed": [True, False, np.nan],
        "float_nan_series": [1.0, 2.5, np.nan],
        "float_series5": [np.nan, 1.2],
        "float_with_inf": [np.inf, -np.inf, 1000000.0, 5.5],
        "inf_series": [np.inf, -np.inf],
        "int_nan_series": [1, 2, np.nan],
        "nan_series": [np.nan],
        "nan_series_2": [np.nan, np.nan, np.nan, np.nan],
        "string_num_nan": ["1.0", "2.0", np.nan],
        "string_with_sep_num_nan": ["1,000.0", "2.1", np.nan],
        "string_flt_nan": ["1.0", "45.67", np.nan],
        "string_str_nan": [
            "I was only robbing the register,",
            "I hope you understand",
            "One of us had better call up the cops",
            "In the hot New Jersey night",
            np.nan,
        ],
        "float_series3": np.array([1.2, 2, 3, 4], dtype=np.float64),
        "np_uint32": np.array([1, 2, 3, 4], dtype=np.uint32),
        "string_np_unicode_series": np.array(["upper", "hall"], dtype=np.str_),
        "complex_series": [
            complex(0, 0),
            complex(1, 2),
            complex(3, -1),
        ],
        "bool_series3": np.array([1, 0, 0, 1], dtype=np.bool_),
        "complex_series_nan": [complex(0, 0), complex(1, 2), complex(3, -1), None],
        "complex_series_nan_2": [
            complex(0, 0),
            complex(1, 2),
            complex(3, -1),
            np.nan,
        ],
        "complex_series_py_nan": [
            complex(0, 0),
            complex(1, 2),
            complex(3, -1),
            np.nan,
        ],
    }
    return sequences


def get_series() -> Dict[str, pd.Series]:
    """
    Taken from Vision to remove the `complex_series_nan` that causes an exception due to a bug
    in pandas 2 and numpy with the value `np.nan * 0j` and  `complex(np.nan, np.nan)`.
    See: https://github.com/numpy/numpy/issues/12919
    """
    from visions.backends.pandas.sequences import get_sequences as get_pandas_sequences
    from visions.backends.python.sequences import get_sequences as get_builtin_sequences

    sequences = get_builtin_sequences()
    sequences.update(get_sequences())

    del sequences["complex_series_nan"]

    test_series = {name: pd.Series(sequence) for name, sequence in sequences.items()}
    test_series.update(get_pandas_sequences())
    assert all(isinstance(v, pd.Series) for v in test_series.values())

    return test_series


base_path = os.path.abspath(os.path.dirname(__file__))

series = get_series()
del series["categorical_complex_series"]

my_config = Settings()
my_config.vars.num.low_categorical_threshold = 0
my_typeset_default = ProfilingTypeSet(my_config)


type_map = {str(k): k for k in my_typeset_default.types}
Numeric = type_map["Numeric"]
Text = type_map["Text"]
Categorical = type_map["Categorical"]
Boolean = type_map["Boolean"]
DateTime = type_map["DateTime"]
Unsupported = type_map["Unsupported"]

contains_map = {
    Numeric: {
        "int_series",
        "Int64_int_series",
        "int_range",
        "Int64_int_nan_series",
        "int_series_boolean",
        "np_uint32",
        "pd_uint32",
        "float_series",
        "float_series2",
        "float_series3",
        "float_series4",
        "inf_series",
        "float_nan_series",
        "float_series5",
        "int_nan_series",
        "float_with_inf",
        "float_series6",
        "complex_series",
        "complex_series_py",
        # "complex_series_nan",
        "complex_series_py_nan",
        "complex_series_nan_2",
        "complex_series_float",
        "complex_series_py_float",
    },
    Text: {
        "timestamp_string_series",
        "string_with_sep_num_nan",
        "string_series",
        "string_unicode_series",
        "string_np_unicode_series",
        "path_series_linux_str",
        "path_series_windows_str",
        "int_str_range",
        "string_date",
        "textual_float",
        "textual_float_nan",
        "ip_str",
        "string_flt",
        "string_num",
        "str_url",
        "string_str_nan",
        "string_num_nan",
        "string_bool_nan",
        "string_flt_nan",
        "str_complex",
        "uuid_series_str",
        "str_int_leading_zeros",
        "email_address_str",
        "str_float_non_leading_zeros",
        "str_int_zeros",
        "str_complex_nan",
        "all_null_empty_str",
        "py_datetime_str",
        "string_dtype_series",
    },
    Categorical: {
        "categorical_float_series",
        "categorical_int_series",
        "categorical_string_series",
        "categorical_char",
        "ordinal",
    },
    Boolean: {
        "bool_series",
        "bool_series2",
        "bool_series3",
        "nullable_bool_series",
        "mixed",
        "bool_nan_series",
    },
    DateTime: {
        "timestamp_series",
        "timestamp_aware_series",
        "datetime",
        "timestamp_series_nat",
        "date_series_nat",
        "date",
    },
    Unsupported: {
        "module",
        "nan_series",
        "nan_series_2",
        "timedelta_series",
        "timedelta_series_nat",
        "timedelta_negative",
        "path_series_linux",
        "path_series_linux_missing",
        "path_series_windows",
        "url_series",
        "url_nan_series",
        "url_none_series",
        "file_test_py",
        "file_mixed_ext",
        "file_test_py_missing",
        "image_png",
        "image_png_missing",
        "image_png",
        "image_png_missing",
        "uuid_series",
        "uuid_series_missing",
        "mixed_list[str,int]",
        "mixed_dict",
        "callable",
        "mixed_integer",
        "mixed_list",
        "time",
        "empty",
        "empty_bool",
        "empty_float",
        "empty_object",
        "empty_int64",
        "ip",
        "ip_missing",
        "ip_mixed_v4andv6",
        "email_address_missing",
        "email_address",
        "all_null_none",
        "all_null_nan",
        "all_null_nat",
    },
}


@pytest.mark.parametrize(
    **patch_arg(
        get_contains_cases(series, contains_map, my_typeset_default), "contains_type"
    )
)
def test_contains(name, series, contains_type, member):
    """Test the generated combinations for "series in type".

    Args:
        series: the series to test
        contains_type: the type to test against
        member: the result
    """
    result, message = contains(name, series, contains_type, member)
    assert result, message


inference_map = {
    "int_series": Numeric,
    "categorical_int_series": Categorical,
    "int_nan_series": Numeric,
    "Int64_int_series": Numeric,
    "Int64_int_nan_series": Numeric,
    "np_uint32": Numeric,
    "pd_uint32": Numeric,
    "int_range": Numeric,
    "float_series": Numeric,
    "float_nan_series": Numeric,
    "int_series_boolean": Numeric,
    "float_series2": Numeric,
    "float_series3": Numeric,
    "float_series4": Numeric,
    "float_series5": Numeric,
    "float_series6": Numeric,
    "complex_series_float": Numeric,
    "categorical_float_series": Categorical,
    "float_with_inf": Numeric,
    "inf_series": Numeric,
    "nan_series": Unsupported,
    "nan_series_2": Unsupported,
    "string_series": Text,
    "categorical_string_series": Categorical,
    "timestamp_string_series": DateTime,
    "string_with_sep_num_nan": Text,  # TODO: Introduce thousands separator
    "string_unicode_series": Text,
    "string_np_unicode_series": Text,
    "string_num_nan": Numeric,
    "string_num": Numeric,
    "string_flt_nan": Numeric,
    "string_flt": Numeric,
    "string_str_nan": Text,
    "string_bool_nan": Boolean,
    "int_str_range": Numeric,
    "string_date": DateTime,
    "str_url": Text,
    "bool_series": Boolean,
    "bool_nan_series": Boolean,
    "nullable_bool_series": Boolean,
    "bool_series2": Boolean,
    "bool_series3": Boolean,
    "complex_series": Numeric,
    "complex_series_nan": Numeric,
    "complex_series_nan_2": Numeric,
    "complex_series_py_nan": Numeric,
    "complex_series_py": Numeric,
    "timestamp_series": DateTime,
    "timestamp_series_nat": DateTime,
    "timestamp_aware_series": DateTime,
    "datetime": DateTime,
    "timedelta_series": Unsupported,
    "timedelta_series_nat": Unsupported,
    "timedelta_negative": Unsupported,
    "geometry_series_missing": Unsupported,
    "geometry_series": Unsupported,
    "path_series_linux": Unsupported,
    "path_series_linux_missing": Unsupported,
    "path_series_linux_str": Text,
    "path_series_windows": Unsupported,
    "path_series_windows_str": Text,
    "url_series": Unsupported,
    "url_nan_series": Unsupported,
    "url_none_series": Unsupported,
    "mixed_list[str,int]": Unsupported,
    "mixed_dict": Unsupported,
    "mixed_integer": Unsupported,
    "mixed_list": Unsupported,
    "mixed": Boolean,
    "callable": Unsupported,
    "module": Unsupported,
    "textual_float": Numeric,
    "textual_float_nan": Numeric,
    "empty": Unsupported,
    "empty_float": Unsupported,
    "empty_bool": Unsupported,
    "empty_int64": Unsupported,
    "empty_object": Unsupported,
    "ip": Unsupported,
    "ip_str": Text,
    "ip_missing": Unsupported,
    "date_series_nat": DateTime,
    "date": DateTime,
    "time": Unsupported,
    "categorical_char": Categorical,
    "ordinal": Categorical,
    "str_complex": Text,
    "uuid_series": Unsupported,
    "uuid_series_str": Text,
    "uuid_series_missing": Unsupported,
    "ip_mixed_v4andv6": Unsupported,
    "file_test_py": Unsupported,
    "file_test_py_missing": Unsupported,
    "file_mixed_ext": Unsupported,
    "image_png": Unsupported,
    "image_png_missing": Unsupported,
    "str_int_leading_zeros": Numeric,
    "str_float_non_leading_zeros": Numeric,
    "str_int_zeros": Numeric,
    "email_address_str": Text,
    "str_complex_nan": Text,
    "email_address": Unsupported,
    "email_address_missing": Unsupported,
    "all_null_nat": Unsupported,
    "all_null_empty_str": Text,
    "py_datetime_str": DateTime,
    "all_null_none": Unsupported,
    "complex_series_py_float": Numeric,
    "all_null_nan": Unsupported,
    "string_dtype_series": Text,
}


@pytest.mark.parametrize(
    **patch_arg(
        get_inference_cases(series, inference_map, my_typeset_default), "inference_type"
    )
)
def test_inference(name, series, inference_type, typeset, difference):
    """Test the generated combinations for "inference(series) == type"

    Args:
        series: the series to test
        inference_type: the type to test against
    """
    result, message = infers(name, series, inference_type, typeset, difference)
    assert result, message


# Conversions in one single step
convert_map = [
    # Model type, Relation type
    (Categorical, Numeric, {"mixed"}),
    (
        Numeric,
        Text,
        {
            "string_flt",
            "string_num_nan",
            "string_num",
            "string_flt_nan",
            "textual_float",
            "textual_float_nan",
            "int_str_range",
            "str_float_non_leading_zeros",
            "str_int_zeros",
            "str_int_leading_zeros",
            "mixed",
            "int_series",
            "categorical_int_series",
            "categorical_float_series",
        },
    ),
    (
        Boolean,
        Text,
        {
            "string_bool_nan",
            "nullable_bool_series",
        },
    ),
    (
        DateTime,
        Text,
        {"py_datetime_str", "timestamp_string_series", "string_date"},
    ),
    (Categorical, Text, {"categorical_string_series"}),
    (Categorical, Numeric, {"categorical_float_series"}),
]


@pytest.mark.parametrize(**get_convert_cases(series, convert_map, my_typeset_default))
def test_conversion(name, source_type, relation_type, series, member):
    """Test the generated combinations for "convert(series) == type" and "infer(series) = source_type"

    Args:
        name: the test name
        source_type: the type to test against
        relation_type: the type to test against
        series: the series to test
    """
    result, message = convert(name, source_type, relation_type, series, member)
    assert result, message


@pytest.fixture
def dataframe(size: int = 1000) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "boolean": np.random.choice([True, False], size=size),
            "numeric": np.random.rand(size),
            "categorical": np.random.choice(np.arange(5), size=size),
            "timeseries": np.arange(size),
        }
    )


def convertion_map() -> list:
    types = {
        "boolean": ["Categorical", "Unsupported"],
        "numeric": ["Categorical", "Boolean", "Unsupported"],
        "categorical": ["Numeric", "Boolean", "TimeSeries", "Unsupported"],
        "timeseries": ["Numeric", "Boolean", "Categorical", "Unsupported"],
    }
    return [(k, {k: i}) for k, v in types.items() for i in v]


@pytest.mark.parametrize("column,type_schema", convertion_map())
def test_type_schema(dataframe: pd.DataFrame, column: str, type_schema: dict):
    prof = ProfileReport(dataframe[[column]], tsmode=True, type_schema=type_schema)
    prof.get_description()
    assert isinstance(prof.typeset, ProfilingTypeSet)
    assert prof.typeset.type_schema[column] == prof.typeset._get_type(
        type_schema[column]
    )


def test_type_schema_with_null_column():
    df = pd.DataFrame({"null_col": [None] * 100})
    prof = ProfileReport(df, type_schema={"null_col": "datetime"})
    description = prof.description_set
    assert description.variables["null_col"]["type"] == "Unsupported"

    prof = ProfileReport(df, type_schema={"null_col": "numeric"})
    description = prof.description_set
    assert description.variables["null_col"]["type"] == "Unsupported"
