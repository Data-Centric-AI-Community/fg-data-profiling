import datetime
import os
import warnings
from functools import partial, wraps
from typing import Callable, List, Sequence, Type
from urllib.parse import urlparse

import filetype
import pandas as pd
from multimethod import multimethod
from pandas.api import types as pdt

from data_profiling.config import Settings
from data_profiling.model.typeset_base import (
    BaseTypeSet,
    Generic,
    IdentityRelation,
    InferenceRelation,
    ProfilingBaseType,
    TypeRelation,
    series_not_empty,
)
from data_profiling.model.typeset_relations import (
    numeric_is_category,
    series_is_string,
    string_is_bool,
    string_is_category,
    string_is_datetime,
    string_is_numeric,
    string_to_bool,
    string_to_datetime,
    string_to_numeric,
    to_bool,
    to_category,
)

pandas_has_string_dtype_flag = hasattr(pdt, "is_string_dtype")


def series_handle_nulls(fn: Callable[..., bool]) -> Callable[..., bool]:
    """Decorator for nullable series"""

    @wraps(fn)
    def inner(series: pd.Series, state: dict, *args, **kwargs) -> bool:
        if "hasnans" not in state:
            state["hasnans"] = series.hasnans

        if state["hasnans"]:
            series = series.dropna()
            if series.empty:
                return False

        return fn(series, state, *args, **kwargs)

    return inner


def typeset_types(config: Settings) -> List[Type[ProfilingBaseType]]:
    """Define the active types, in precedence order, based on the config.

    When more than one relation leaving a type matches a series, the target
    listed first wins:

    - Boolean before Numeric: a string listed in ``vars.bool.mappings`` is a
      flag, even when it also parses as a number (e.g. a user maps "1"/"0").
    - DateTime, URL and Path before Categorical: a date, URL or path column
      keeps its type even when it has few distinct values.
    - Categorical before TimeSeries: a numeric column with at most
      ``vars.num.low_categorical_threshold`` distinct values is categorical in
      every mode, as it is when the same values arrive as strings.
    - Categorical before Text: the low-cardinality reading wins over free text.
    """

    class Unsupported(Generic):
        """Base type. All other types have relationship with this type."""

        pass

    class Numeric(ProfilingBaseType):
        """Type for all numeric (float, int) columns.

        Can be transformed from
        - Unsupported
        - String

        Examples
        --------
        >>> s = pd.Series([1, 2, 5, 3, 8, 9])
        >>> s in Numeric
        True

        >>> s = pd.Series([.34, 2.9, 55, 3.14, 89, 91])
        >>> s in Numeric
        True
        """

        @staticmethod
        def get_relations() -> Sequence[TypeRelation]:
            return [
                IdentityRelation(Unsupported),
                InferenceRelation(
                    Text,
                    relationship=lambda x, y: partial(string_is_numeric, k=config)(
                        x, y
                    ),
                    transformer=string_to_numeric,
                ),
            ]

        @staticmethod
        @multimethod
        @series_not_empty
        @series_handle_nulls
        def contains_op(series: pd.Series, state: dict) -> bool:
            return pdt.is_numeric_dtype(series) and not pdt.is_bool_dtype(series)

    class Text(ProfilingBaseType):
        """Type for plaintext columns.
        Like name, note, string identifier, residence etc.

        Examples
        --------
        >>> s = pd.Series(["AX01", "BC32", "AC00"])
        >>> s in Categorical
        True

        >>> s = pd.Series([1, 2, 3, 4])
        >>> s in Categorical
        False
        """

        @staticmethod
        def get_relations() -> Sequence[TypeRelation]:
            return [
                IdentityRelation(Unsupported),
            ]

        @staticmethod
        @multimethod
        @series_not_empty
        @series_handle_nulls
        def contains_op(series: pd.Series, state: dict) -> bool:
            return (
                not isinstance(series.dtype, pd.CategoricalDtype)
                and pdt.is_string_dtype(series)
                and series_is_string(series, state)
            )

    class DateTime(ProfilingBaseType):
        @staticmethod
        def get_relations() -> Sequence[TypeRelation]:
            return [
                IdentityRelation(Unsupported),
                InferenceRelation(
                    Text,
                    relationship=lambda x, y: partial(string_is_datetime)(x, y),
                    transformer=string_to_datetime,
                ),
            ]

        @staticmethod
        @multimethod
        @series_not_empty
        @series_handle_nulls
        def contains_op(series: pd.Series, state: dict) -> bool:
            is_datetime = pdt.is_datetime64_any_dtype(series)
            if is_datetime:
                return True
            has_builtin_datetime = (
                series.dropna()
                .apply(type)
                .isin([datetime.date, datetime.datetime])
                .all()
            )
            return has_builtin_datetime

    class Categorical(ProfilingBaseType):
        """Type for categorical columns.
        Categorical columns in pandas categorical format
        and columns in string format with small count of unique values.

        Can be transformed from:
            - Unsupported
            - Numeric
            - String

        Examples
        --------
        >>> s = pd.Series(["male", "female", "female", "male"], dtype="category")
        >>> s in Categorical
        True

        >>> s = pd.Series(["male", "female"])
        >>> s in Categorical
        False

        >>> s = pd.Series(["male", "female", "female", "male"])
        >>> s in Categorical
        True
        """

        @staticmethod
        def get_relations() -> Sequence[TypeRelation]:
            return [
                IdentityRelation(Unsupported),
                InferenceRelation(
                    Numeric,
                    relationship=lambda x, y: partial(numeric_is_category, k=config)(
                        x, y
                    ),
                    transformer=to_category,
                ),
                InferenceRelation(
                    Text,
                    relationship=lambda x, y: partial(string_is_category, k=config)(
                        x, y
                    ),
                    transformer=to_category,
                ),
            ]

        @staticmethod
        @multimethod
        @series_not_empty
        @series_handle_nulls
        def contains_op(series: pd.Series, state: dict) -> bool:
            is_valid_dtype = isinstance(
                series.dtype, pd.CategoricalDtype
            ) and not pdt.is_bool_dtype(series)
            if is_valid_dtype:
                return True
            return False

    class Boolean(ProfilingBaseType):
        """Type for boolean columns."""

        @staticmethod
        def get_relations() -> Sequence[TypeRelation]:
            # Numeric [0, 1] goes via Categorical with distinct_count_without_nan <= 2
            mapping = config.vars.bool.mappings

            return [
                IdentityRelation(Unsupported),
                InferenceRelation(
                    Text,
                    relationship=lambda x, y: partial(string_is_bool, k=mapping)(x, y),
                    transformer=lambda s, st: to_bool(
                        partial(string_to_bool, k=mapping)(s, st)
                    ),
                ),
            ]

        @staticmethod
        @multimethod
        @series_not_empty
        @series_handle_nulls
        def contains_op(series: pd.Series, state: dict) -> bool:
            if pdt.is_object_dtype(series):
                try:
                    return series.isin({True, False}).all()
                except:  # noqa: E722
                    return False

            return pdt.is_bool_dtype(series)

    class URL(ProfilingBaseType):
        @staticmethod
        def get_relations() -> Sequence[TypeRelation]:
            return [IdentityRelation(Text)]

        @staticmethod
        @multimethod
        @series_handle_nulls
        def contains_op(series: pd.Series, state: dict) -> bool:
            # TODO: use coercion utils
            try:
                url_gen = (urlparse(x) for x in series)
                return all(x.netloc and x.scheme for x in url_gen)  # noqa: TC300
            except AttributeError:
                return False

    class Path(ProfilingBaseType):
        @staticmethod
        def get_relations() -> Sequence[TypeRelation]:
            return [IdentityRelation(Text)]

        @staticmethod
        @multimethod
        @series_handle_nulls
        def contains_op(series: pd.Series, state: dict) -> bool:
            # TODO: use coercion utils
            try:
                return all(os.path.isabs(p) for p in series)
            except TypeError:
                return False

    class File(ProfilingBaseType):
        @staticmethod
        def get_relations() -> Sequence[TypeRelation]:
            return [IdentityRelation(Path)]

        @staticmethod
        @multimethod
        @series_handle_nulls
        def contains_op(series: pd.Series, state: dict) -> bool:
            return all(os.path.exists(p) for p in series)

    class Image(ProfilingBaseType):
        @staticmethod
        def get_relations() -> Sequence[TypeRelation]:
            return [IdentityRelation(File)]

        @staticmethod
        @multimethod
        @series_handle_nulls
        def contains_op(series: pd.Series, state: dict) -> bool:
            return all(
                filetype.guess(str(p))
                and filetype.guess(str(p)).mime.startswith("image/")
                for p in series
            )

    class TimeSeries(ProfilingBaseType):
        @staticmethod
        def get_relations() -> Sequence[TypeRelation]:
            return [IdentityRelation(Numeric)]

        @staticmethod
        @multimethod
        @series_not_empty
        @series_handle_nulls
        def contains_op(series: pd.Series, state: dict) -> bool:
            def is_timedependent(series: pd.Series) -> bool:
                autocorrelation_threshold = config.vars.timeseries.autocorrelation
                lags = config.vars.timeseries.lags
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", RuntimeWarning)
                    for lag in lags:
                        autcorr = series.autocorr(lag=lag)
                        if autcorr >= autocorrelation_threshold:
                            return True

                return False

            is_numeric = pdt.is_numeric_dtype(series) and not pdt.is_bool_dtype(series)
            return is_numeric and is_timedependent(series)

    active = {Unsupported, Boolean, Numeric, Text, Categorical, DateTime}
    if config.vars.path.active:
        active.add(Path)
        if config.vars.file.active:
            active.add(File)
            if config.vars.image.active:
                active.add(Image)

    if config.vars.url.active:
        active.add(URL)

    if config.vars.timeseries.active:
        active.add(TimeSeries)

    precedence = [
        Unsupported,
        Boolean,
        Numeric,
        DateTime,
        URL,
        Path,
        File,
        Image,
        Categorical,
        TimeSeries,
        Text,
    ]
    return [t for t in precedence if t in active]


class ProfilingTypeSet(BaseTypeSet):
    def __init__(self, config: Settings, type_schema: dict = None):
        self.config = config
        super().__init__(typeset_types(config))
        self.type_schema = self._init_type_schema(type_schema or {})

    def _init_type_schema(self, type_schema: dict) -> dict:
        return {k: self._get_type(v) for k, v in type_schema.items()}
