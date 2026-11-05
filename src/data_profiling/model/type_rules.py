"""Type inference rules shared by the pandas and Spark backends.

Each backend computes the facts about a column in its own way (a pandas
Series method, a Spark aggregation); the decisions on those facts live here,
so both backends apply the same thresholds. The order in which candidate types
are tried comes from the typeset (see ``BaseTypeSet.relations_from``).
"""
from data_profiling.config import Settings


def is_few_numbers(n_unique: int, config: Settings) -> bool:
    """A numeric column with at most ``vars.num.low_categorical_threshold``
    distinct values is categorical."""
    return 1 <= n_unique <= config.vars.num.low_categorical_threshold


def is_categorical_cardinality(n_unique: int, n: int, config: Settings) -> bool:
    """A string column is categorical when it has between 1 and
    ``vars.cat.cardinality_threshold`` distinct values, and the share of
    distinct values among the non-null values is below
    ``vars.cat.percentage_cat_threshold``.

    Args:
        n_unique: distinct non-null values.
        n: non-null values.
    """
    if n == 0:
        return False
    ratio = n_unique / n
    unique_threshold = config.vars.cat.percentage_cat_threshold
    below_ratio = (
        ratio < unique_threshold if unique_threshold <= 1 else ratio <= unique_threshold
    )
    return 1 <= n_unique <= config.vars.cat.cardinality_threshold and below_ratio
