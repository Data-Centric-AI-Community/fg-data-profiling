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
  ("-nan") are accepted. A column whose numbers are all integers in the
  64-bit range is cast to BIGINT, so large integers keep their exact value;
  any other number column is cast to DOUBLE;
- dates: ISO formats ("2024-01-31", "2024-01-31 10:00:00", with "T" and time
  zone) and "yyyy/M/d". Formats that pandas reads with dateutil, such as
  "31/01/2024" or "Jan 31 2024", stay text on Spark.

URL, Path, File, Image and TimeSeries are not inferred on Spark.
"""
from typing import Dict, Iterable, Mapping, Optional, Sequence, Tuple

from pyspark.sql import Column, DataFrame
from pyspark.sql import functions as F
from pyspark.sql.types import CharType, NumericType, StringType, VarcharType

from data_profiling.config import Settings
from data_profiling.model.type_rules import is_categorical_cardinality, is_few_numbers
from data_profiling.model.typeset_base import BaseTypeSet

STRING_TYPES = (StringType, CharType, VarcharType)
# Strings that pandas parses as a missing datetime (its NaT strings). The
# match is exact: pandas neither lowercases nor strips them, so "Nat" or
# " NaT" are not markers.
MISSING_DATE_MARKERS = ("", "NaT", "nat", "NAT", "nan", "NaN", "NAN")
DATE_FORMATS = ("yyyy/M/d",)
INTEGER_LITERAL = r"^\s*[+-]?[0-9]+\s*$"


def _quote(name: str) -> str:
    return "`" + name.replace("`", "``") + "`"


def _unused_name(base: str, taken: set) -> str:
    """``base``, with ``_`` appended until it differs from every name in
    ``taken`` (lowercased names), and record it there.

    Spark resolves column names case-insensitively by default, so names that
    differ only in case also collide.
    """
    name = base
    while name.lower() in taken:
        name += "_"
    taken.add(name.lower())
    return name


def _taken_names(columns: Iterable[str]) -> set:
    return {column.lower() for column in columns}


def _cleaned_number(value: Column) -> Column:
    """The string spelled the way Spark's cast reads it when float() does."""
    without_underscores = F.regexp_replace(value, r"(?<=[0-9])_(?=[0-9])", "")
    return F.regexp_replace(without_underscores, r"(?i)^(\s*)[+-](nan\s*)$", "$1$2")


def _number(value: Column, cleaned_name: str, sql_type: str = "DOUBLE") -> Column:
    """The value as ``sql_type`` (DOUBLE or BIGINT), or null when Python's
    float() would reject it.

    ``cleaned_name`` is a column holding ``_cleaned_number(value)``: the cast
    is written in SQL (``try_cast`` has no Column API before Spark 4).
    """
    # Java-style suffixes ("1.0d", "2f") are numbers to Spark, not to Python.
    return F.when(value.rlike(r"[0-9.][dDfF]\s*$"), F.lit(None)).otherwise(
        F.expr(f"try_cast({_quote(cleaned_name)} AS {sql_type})")
    )


def _is_bigint(cleaned_name: str) -> Column:
    """Whether the cleaned string is an integer in the 64-bit range."""
    cleaned = F.col(_quote(cleaned_name))
    exact = F.expr(f"try_cast({_quote(cleaned_name)} AS BIGINT)")
    return cleaned.rlike(INTEGER_LITERAL) & exact.isNotNull()


def _numeric_sql_types(
    frame: DataFrame, columns: Sequence[Tuple[str, str]]
) -> Dict[str, str]:
    """BIGINT for the number columns whose numbers are all 64-bit integers,
    DOUBLE for the others, in one aggregation.

    ``columns`` holds pairs of a column name and the name of the column with
    its cleaned strings (see ``_number``).
    """
    if not columns:
        return {}
    counts = [
        F.sum(
            F.when(
                _number(F.col(_quote(name)), cleaned).isNotNull()
                & ~_is_bigint(cleaned),
                1,
            ).otherwise(0)
        ).alias(f"n_not_bigint_{k}")
        for k, (name, cleaned) in enumerate(columns)
    ]
    row = frame.agg(*counts).first()
    return {
        name: "DOUBLE" if row[f"n_not_bigint_{k}"] else "BIGINT"
        for k, (name, _) in enumerate(columns)
    }


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


def cast_frame(
    config: Settings,
    df: DataFrame,
    types: Mapping[str, str],
    numeric_types: Optional[Mapping[str, str]] = None,
) -> DataFrame:
    """Cast the columns of ``df`` to their inferred types.

    String columns inferred as Numeric, DateTime or Boolean are converted, and
    numeric columns inferred as Categorical become strings, as on pandas.
    Other columns are kept as they are. Returns ``df`` itself when no column
    needs a cast.

    Numeric string columns become BIGINT when all their numbers are 64-bit
    integers and DOUBLE otherwise. ``numeric_types`` gives that choice per
    column ("BIGINT" or "DOUBLE"); columns missing from it are decided with
    one aggregation over ``df``.
    """
    work = df
    # Temporary columns get names no input column has, so they never replace
    # one (withColumn overwrites a column of the same name).
    taken = _taken_names(df.columns)
    cleaned_names: Dict[str, str] = {}
    for i, field in enumerate(df.schema.fields):
        if (
            isinstance(field.dataType, STRING_TYPES)
            and types.get(field.name) == "Numeric"
        ):
            cleaned_names[field.name] = _unused_name(f"__dp_number_{i}", taken)
            work = work.withColumn(
                cleaned_names[field.name], _cleaned_number(F.col(_quote(field.name)))
            )
    sql_types = dict(numeric_types or {})
    sql_types.update(
        _numeric_sql_types(
            work,
            [(n, c) for n, c in cleaned_names.items() if n not in sql_types],
        )
    )

    columns = []
    changed = False
    for field in df.schema.fields:
        name = field.name
        value = F.col(_quote(name))
        target = types.get(name)
        is_string = isinstance(field.dataType, STRING_TYPES)
        if is_string and target == "Numeric":
            column = _number(value, cleaned_names[name], sql_types[name])
        elif is_string and target == "DateTime":
            column = _date(value)
        elif is_string and target == "Boolean":
            column = _boolean(value, config)
        elif isinstance(field.dataType, NumericType) and target == "Categorical":
            # Only numbers become strings: arrays are Categorical from the
            # schema already, and stringifying them would merge distinct
            # values (["a, b"] and ["a", "b"] both print as "[a, b]").
            column = value.cast("string")
        else:
            columns.append(value)
            continue
        changed = True
        columns.append(column.alias(name))
    if not changed:
        return df
    return work.select(*columns)


def string_facts(config: Settings, series: DataFrame) -> Dict[str, int]:
    """Counts that decide the type of a string column, in one aggregation."""
    # The column is selected alone as "v", so the temporary "__dp_number"
    # cannot collide with an input column.
    v = F.col("v")
    frame = series.select(F.col(_quote(series.columns[0])).alias("v")).withColumn(
        "__dp_number", _cleaned_number(v)
    )
    number = _number(v, "__dp_number")
    date = _date(v)
    is_number = number.isNotNull()
    is_date = date.isNotNull()
    is_marker = v.isin(*MISSING_DATE_MARKERS)
    bool_values = list(config.vars.bool.mappings)

    def count_where(condition: Column) -> Column:
        return F.sum(F.when(condition, 1).otherwise(0))

    row = frame.agg(
        F.count(v).alias("n"),
        F.countDistinct(v).alias("n_distinct"),
        count_where(v.isNotNull() & ~F.lower(v).isin(*bool_values)).alias("n_not_bool"),
        count_where(v.isNotNull() & ~is_number).alias("n_not_number"),
        count_where(is_number & F.isnan(number)).alias("n_number_nan"),
        count_where(is_number & ~_is_bigint("__dp_number")).alias("n_not_bigint"),
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
        numeric_types = None
        if facts is not None and chosen == "Numeric":
            # Integers are cast exactly; the facts already tell which.
            exact = facts["n_not_bigint"] == 0
            numeric_types = {series.columns[0]: "BIGINT" if exact else "DOUBLE"}
        series = cast_frame(config, series, {series.columns[0]: chosen}, numeric_types)
        node = chosen
        visited.add(node)
