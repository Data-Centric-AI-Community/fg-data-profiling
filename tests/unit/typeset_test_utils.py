"""Parametrization helpers for the typeset tests.

Three kinds of cases:

- contains: ``series in Type`` for every series and every type. A series that
  belongs to a type also belongs to all its ancestors.
- inference: ``typeset.infer_type(series)`` equals the expected type.
- conversion: for each inference relation ``relation_type -> source_type``,
  the relation holds exactly for the listed series, and the transformer runs
  on them. Every inference relation in the typeset must be covered.
"""
from typing import Dict, Iterable, List, Mapping, Set, Tuple, Type

import pandas as pd
import pytest

from data_profiling.model.typeset_base import (
    BaseTypeSet,
    ProfilingBaseType,
    TypeRelation,
)

T = Type[ProfilingBaseType]
ConvertMap = Iterable[Tuple[T, T, Set[str]]]


def _with_ancestors(
    typeset: BaseTypeSet, contains_map: Mapping[T, Set[str]]
) -> Dict[T, Set[str]]:
    expanded = {t: set(names) for t, names in contains_map.items()}
    for type_ in reversed(typeset.topological_order()):
        parent = typeset.parent_of(type_)
        if parent is not None and type_ in expanded:
            expanded.setdefault(parent, set()).update(expanded[type_])
    return expanded


def contains_cases(
    sequences: Mapping[str, pd.Series],
    contains_map: Mapping[T, Set[str]],
    typeset: BaseTypeSet,
) -> dict:
    expanded = _with_ancestors(typeset, contains_map)
    used = set().union(*expanded.values())
    unused = set(sequences) - used
    unknown = used - set(sequences)
    if unused:
        raise ValueError(f"Series missing from the contains map: {sorted(unused)}")
    if unknown:
        raise ValueError(f"Contains map lists unknown series: {sorted(unknown)}")

    params = [
        pytest.param(series, type_, name in names, id=f"{name} x {type_}")
        for name, series in sequences.items()
        for type_, names in expanded.items()
    ]
    return {"argnames": "series,contains_type,member", "argvalues": params}


def inference_cases(
    sequences: Mapping[str, pd.Series], inference_map: Mapping[str, T]
) -> dict:
    missing = set(sequences) - set(inference_map)
    if missing:
        raise ValueError(f"Series missing from the inference map: {sorted(missing)}")
    params = [
        pytest.param(series, inference_map[name], id=name)
        for name, series in sequences.items()
    ]
    return {"argnames": "series,expected_type", "argvalues": params}


def _relation(source_type: T, relation_type: T) -> TypeRelation:
    for relation in source_type.relations:
        if relation.inferential and relation.related_type is relation_type:
            return relation
    raise ValueError(f"No inference relation {relation_type}->{source_type}")


def conversion_cases(
    sequences: Mapping[str, pd.Series],
    convert_map: ConvertMap,
    typeset: BaseTypeSet,
) -> dict:
    convert_map = list(convert_map)
    covered = {(source, related) for source, related, names in convert_map if names}
    untested: List[str] = [
        str(relation)
        for type_ in typeset.types
        for relation in typeset.relations_from(type_)
        if relation.inferential and (relation.type, type_) not in covered
    ]
    if untested:
        raise ValueError(f"Inference relations without test series: {untested}")

    params = []
    for source_type, relation_type, names in convert_map:
        unknown = set(names) - set(sequences)
        if unknown:
            raise ValueError(f"Convert map lists unknown series: {sorted(unknown)}")
        for name, series in sequences.items():
            if series in relation_type:
                params.append(
                    pytest.param(
                        series,
                        source_type,
                        relation_type,
                        name in names,
                        id=f"{name}: {relation_type} -> {source_type}",
                    )
                )
    return {
        "argnames": "series,source_type,relation_type,member",
        "argvalues": params,
    }


def assert_contains(series: pd.Series, contains_type: T, member: bool) -> None:
    result = series in contains_type
    assert result == member, f"{contains_type}: expected {member}, got {result}"


def assert_infers(typeset: BaseTypeSet, series: pd.Series, expected_type: T) -> None:
    inferred = typeset.infer_type(series)
    assert inferred == expected_type, (
        f"expected {expected_type}, got {inferred}; "
        f"steps: {typeset.explain_type(series)}"
    )


def assert_converts(
    series: pd.Series, source_type: T, relation_type: T, member: bool
) -> None:
    relation = _relation(source_type, relation_type)
    result = relation.is_relation(series, {})
    assert result == member, f"{relation}: expected {member}, got {result}"
    if member:
        relation.transform(series, {})
