"""Type inference for Spark columns, with the same rules as the pandas typeset.

The candidate types and their order come from the typeset
(``BaseTypeSet.relations_from``), and the thresholds from
``data_profiling.model.type_rules``. What differs is how the facts about a
column are computed: one Spark aggregation per string column instead of
pandas Series methods.

Value parsing follows Spark, so a few strings are read differently than on
pandas:

- numbers: Spark's cast, adjusted to Python's float(): Java-style suffixes
  ("1.0d") are rejected, underscores between digits ("1_000") and signed NaN
  ("-nan") are accepted;
- dates: ISO formats ("2024-01-31", "2024-01-31 10:00:00", with "T" and time
  zone) and "yyyy/M/d". Formats that pandas reads with dateutil, such as
  "31/01/2024" or "Jan 31 2024", stay text on Spark.

URL, Path, File, Image and TimeSeries are not inferred on Spark.
"""
from typing import Dict, Mapping, Optional, Tuple

from pyspark.sql import Column, DataFrame
from pyspark.sql import functions as F
from pyspark.sql.types import CharType, StringType, VarcharType

from data_profiling.config import Settings
from data_profiling.model.type_rules import is_categorical_cardinality, is_few_numbers
from data_profiling.model.typeset_base import BaseTypeSet

STRING_TYPES = (StringType, CharType, VarcharType)
# Strings that pandas parses as a missing datetime.
MISSING_DATE_MARKERS = ("", "nat")
DATE_FORMATS = ("yyyy/M/d",)


def _quote(name: str) -> str:
    return "`" + name.replace("`", "``") + "`"


def _cleaned_number(value: Column) -> Column:
    """The string spelled the way Spark's cast reads it when float() does."""
    without_underscores = F.regexp_replace(value, r"(?<=[0-9])_(?=[0-9])", "")
    return F.regexp_replace(without_underscores, r"(?i)^(\s*)[+-](nan\s*)$", "$1$2")


def _number(value: Column, cleaned_name: str) -> Column:
    """The value as a double, or null when Python's float() would reject it.

    ``cleaned_name`` is a column holding ``_cleaned_number(value)``: the cast
    is written in SQL (``try_cast`` has no Column API before Spark 4).
    """
    # Java-style suffixes ("1.0d", "2f") are numbers to Spark, not to Python.
    return F.when(value.rlike(r"[0-9.][dDfF]\s*$"), F.lit(None)).otherwise(
        F.expr(f"try_cast({_quote(cleaned_name)} AS DOUBLE)")
    )


def _date(value: Column) -> Column:
    parsed = [F.try_to_timestamp(value)]
    parsed += [F.try_to_timestamp(value, F.lit(fmt)) for fmt in DATE_FORMATS]
    return F.coalesce(*parsed)


def _boolean(value: Column, config: Settings) -> Column:
    mappings = config.vars.bool.mappings
    true_values = [k for k, b in mappings.items() if b]
    false_values = [k for k, b in mappings.items() if not b]
    lowered = F.lower(value)
    return F.when(lowered.isin(*true_values), F.lit(True)).when(
        lowered.isin(*false_values), F.lit(False)
    )


def cast_frame(config: Settings, df: DataFrame, types: Mapping[str, str]) -> DataFrame:
    """Cast the columns of ``df`` to their inferred types.

    String columns inferred as Numeric, DateTime or Boolean are converted, and
    numeric columns inferred as Categorical become strings, as on pandas.
    Other columns are kept as they are.
    """
    work = df
    columns = []
    for i, field in enumerate(df.schema.fields):
        name = field.name
        value = F.col(_quote(name))
        target = types.get(name)
        is_string = isinstance(field.dataType, STRING_TYPES)
        if is_string and target == "Numeric":
            cleaned = f"__dp_number_{i}"
            work = work.withColumn(cleaned, _cleaned_number(value))
            column = _number(value, cleaned)
        elif is_string and target == "DateTime":
            column = _date(value)
        elif is_string and target == "Boolean":
            column = _boolean(value, config)
        elif not is_string and target == "Categorical":
            column = value.cast("string")
        else:
            column = value
        columns.append(column.alias(name))
    return work.select(*columns)


def string_facts(config: Settings, series: DataFrame) -> Dict[str, int]:
    """Counts that decide the type of a string column, in one aggregation."""
    v = F.col("v")
    frame = series.select(F.col(_quote(series.columns[0])).alias("v")).withColumn(
        "__dp_number", _cleaned_number(v)
    )
    number = _number(v, "__dp_number")
    date = _date(v)
    is_number = number.isNotNull()
    is_date = date.isNotNull()
    is_marker = F.lower(F.trim(v)).isin(*MISSING_DATE_MARKERS)
    bool_values = list(config.vars.bool.mappings)

    def count_where(condition: Column) -> Column:
        return F.sum(F.when(condition, 1).otherwise(0))

    row = frame.agg(
        F.count(v).alias("n"),
        F.countDistinct(v).alias("n_distinct"),
        count_where(v.isNotNull() & ~F.lower(v).isin(*bool_values)).alias("n_not_bool"),
        count_where(v.isNotNull() & ~is_number).alias("n_not_number"),
        count_where(is_number & F.isnan(number)).alias("n_number_nan"),
        count_where(is_date).alias("n_date"),
        count_where(v.isNotNull() & ~is_date & ~is_marker).alias("n_not_date"),
        count_where(is_date & ~v.rlike(r"[0-9]")).alias("n_date_no_digit"),
        count_where(is_date & is_number).alias("n_date_number"),
    ).first()
    return {key: int(row[key] or 0) for key in row.asDict()}


def _string_relation(target: str, facts: Dict[str, int], config: Settings) -> bool:
    """Whether a string column becomes ``target``, as the pandas relations decide."""
    n = facts["n"]
    all_bool = facts["n_not_bool"] == 0
    if target == "Boolean":
        return all_bool
    if target == "Numeric":
        # string_is_numeric: every value is a number, not all of them NaN, and
        # more distinct strings than vars.num.low_categorical_threshold.
        return (
            facts["n_not_number"] == 0
            and facts["n_number_nan"] < n
            and not is_few_numbers(facts["n_distinct"], config)
        )
    if target == "DateTime":
        # string_is_datetime: every value is a date or a missing marker, at
        # least one date, every date has a digit, and not all are plain numbers.
        return (
            facts["n_not_date"] == 0
            and facts["n_date"] > 0
            and facts["n_date_no_digit"] == 0
            and facts["n_date_number"] < facts["n_date"]
        )
    if target == "Categorical":
        return (
            is_categorical_cardinality(facts["n_distinct"], n, config) and not all_bool
        )
    return False  # URL, Path, ...: not inferred on Spark


def _few_numbers(series: DataFrame, config: Settings) -> bool:
    if config.vars.num.low_categorical_threshold < 1:
        return False
    column = F.col(_quote(series.columns[0]))
    non_nan = F.when(~F.isnan(column), column)
    n_unique = series.agg(F.countDistinct(non_nan).alias("n")).first()["n"]
    return is_few_numbers(int(n_unique), config)


def infer_spark_type(
    config: Settings,
    series: DataFrame,
    typeset: BaseTypeSet,
    start: str,
) -> Tuple[str, DataFrame]:
    """Walk the typeset from ``start`` (the type of the Spark schema).

    Returns the inferred type and the column cast to it. Candidates are tried
    in the typeset's precedence order; relations without a Spark rule above
    never match.
    """
    if start not in {str(t) for t in typeset.types}:
        return start, series
    facts: Optional[Dict[str, int]] = None
    node = start
    visited = {node}
    while True:
        chosen = None
        for relation in typeset.relations_from(typeset._get_type(node)):
            target = str(relation.type)
            if target in visited:
                continue
            if node == "Text":
                if facts is None:
                    facts = string_facts(config, series)
                    if facts["n"] == 0:
                        # As on pandas: a column without values is Unsupported.
                        return "Unsupported", series
                matches = _string_relation(target, facts, config)
            elif node == "Numeric" and target == "Categorical":
                matches = _few_numbers(series, config)
            else:
                matches = False
            if matches:
                chosen = target
                break
        if chosen is None:
            return node, series
        series = cast_frame(config, series, {series.columns[0]: chosen})
        node = chosen
        visited.add(node)
