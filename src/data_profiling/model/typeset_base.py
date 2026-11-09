"""Type system used to infer the variable type of each column.

The design (a tree of identity relations plus inference relations, walked from
a generic root type) follows visions (https://github.com/dylan-profiler/visions)
by Ian Eaves and Simon Brugman, which this package used for type inference
until version 4. This implementation adds an explicit precedence order, so that
inference is deterministic, and ``explain_type``.

A typeset is an ordered collection of types. Each type declares relations to
other types:

- an ``IdentityRelation`` to its parent: the type is a refinement of the
  parent, decided by the type's own ``contains_op`` (e.g. ``URL`` is a ``Text``
  whose values parse as URLs). Identity relations form a tree rooted at a
  subclass of ``Generic``.
- ``InferenceRelation``s from other types: a series of the source type can be
  converted into this type (e.g. a ``Text`` series of numeric strings becomes
  ``Numeric``), with a predicate and a transformer.

``detect_type`` walks the identity tree only and never changes the data.
``infer_type`` walks identity and inference relations and returns the cast
series alongside the type.

Precedence is explicit: when more than one relation leaving a type matches,
the target that comes first in the typeset's type order wins. The order of the
sequence passed to ``BaseTypeSet`` is the precedence order, so inference is
deterministic. ``explain_type`` reports every step and every candidate that
matched, to make ambiguous columns visible.
"""

import copy
import functools
from dataclasses import dataclass, field, replace
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Type

import pandas as pd

Relationship = Callable[[Any, dict], bool]
Transformer = Callable[[Any, dict], Any]


def identity_transform(series: Any, state: dict) -> Any:
    return series


STEP_CACHE = "step"


def step_cache(state: dict) -> dict:
    """Cache for results computed on the series at the current traversal step.

    Predicates and the transformer of the chosen relation see the same series,
    so a transformer can reuse what a predicate computed (a parsed column, a
    distinct count). The traversal clears this cache after every transform.
    """
    return state.setdefault(STEP_CACHE, {})


def series_not_empty(fn: Callable[..., bool]) -> Callable[..., bool]:
    """Decorator: an empty series belongs to no type."""

    @functools.wraps(fn)
    def inner(series: pd.Series, *args, **kwargs) -> bool:
        if series.empty:
            return False
        return fn(series, *args, **kwargs)

    return inner


def _non_null(series: pd.Series, state: Optional[dict]) -> pd.Series:
    """The non-null values of the series, computed once per series and step."""
    if state is None:
        return series.dropna() if series.hasnans else series
    cache = step_cache(state).setdefault("non_null", {})
    entry = cache.get(id(series))
    # The entry keeps a reference to the series, so the id cannot be reused.
    if entry is None or entry[0] is not series:
        entry = (series, series.dropna() if series.hasnans else series)
        cache[id(series)] = entry
    return entry[1]


def series_handle_nulls(fn: Callable[..., bool]) -> Callable[..., bool]:
    """Decorator: evaluate the predicate on the non-null values only.

    A series with only null values belongs to no type.
    """

    @functools.wraps(fn)
    def inner(series: pd.Series, *args, **kwargs) -> bool:
        state = args[0] if args and isinstance(args[0], dict) else None
        non_null = _non_null(series, state)
        if non_null is not series and non_null.empty:
            return False
        return fn(non_null, *args, **kwargs)

    return inner


@dataclass(frozen=True)
class TypeRelation:
    """A directed relation ``related_type -> type``."""

    related_type: Type["ProfilingBaseType"]
    relationship: Optional[Relationship] = None
    transformer: Transformer = identity_transform
    type: Optional[Type["ProfilingBaseType"]] = None
    inferential: bool = False

    def is_relation(self, series: Any, state: Optional[dict] = None) -> bool:
        if self.relationship is None:
            raise NotImplementedError(f"Relation {self} has no relationship")
        return bool(self.relationship(series, {} if state is None else state))

    def transform(self, series: Any, state: Optional[dict] = None) -> Any:
        return self.transformer(series, {} if state is None else state)

    def __str__(self) -> str:
        return f"{self.related_type}->{self.type}"


@dataclass(frozen=True)
class IdentityRelation(TypeRelation):
    """The owning type is a refinement of ``related_type``.

    The relationship defaults to the owning type's ``contains_op``.
    """

    inferential: bool = field(default=False, init=False)


@dataclass(frozen=True)
class InferenceRelation(TypeRelation):
    """A series of ``related_type`` can be converted into the owning type."""

    inferential: bool = field(default=True, init=False)

    def __post_init__(self) -> None:
        if self.relationship is None:
            raise ValueError("InferenceRelation requires a relationship")


class ProfilingTypeMeta(type):
    def __contains__(cls, series: Any) -> bool:
        return bool(cls.contains_op(series, {}))  # type: ignore[attr-defined]

    def __str__(cls) -> str:
        return cls.__name__

    def __repr__(cls) -> str:
        return cls.__name__

    @property
    def relations(cls) -> Tuple[TypeRelation, ...]:
        """The type's relations, bound to the type (cached per class)."""
        cached = cls.__dict__.get("_bound_relations")
        if cached is None:
            bound = []
            for relation in cls.get_relations():  # type: ignore[attr-defined]
                if relation.relationship is None:
                    relation = replace(
                        relation, type=cls, relationship=cls.contains_op  # type: ignore[attr-defined]
                    )
                else:
                    relation = replace(relation, type=cls)
                bound.append(relation)
            cached = tuple(bound)
            type.__setattr__(cls, "_bound_relations", cached)
        return cached


class ProfilingBaseType(metaclass=ProfilingTypeMeta):
    """Base class for variable types."""

    @staticmethod
    def get_relations() -> Sequence[TypeRelation]:
        return []

    @staticmethod
    def contains_op(series: Any, state: dict) -> bool:
        raise NotImplementedError


class Generic(ProfilingBaseType):
    """Root type: every series belongs to it."""

    @staticmethod
    def contains_op(series: Any, state: dict) -> bool:
        return True


T = Type[ProfilingBaseType]


def _isolated(state: dict) -> dict:
    """A copy of the traversal state that predicates can modify freely."""
    try:
        return copy.deepcopy(state)
    except Exception:  # uncopyable values: fall back to a shallow copy
        return dict(state)


class BaseTypeSet:
    """An ordered collection of types and the relations between them.

    Args:
        types: the types, in precedence order. When several relations leaving
            the same type match a series, the one whose target comes first in
            this sequence wins. Exactly one type must subclass ``Generic``;
            it is the root. Every other type needs exactly one
            ``IdentityRelation`` to a type in the set.
    """

    def __init__(self, types: Sequence[T]):
        if isinstance(types, (set, frozenset)):
            raise TypeError(
                "types must be an ordered sequence: its order is the "
                "precedence used when more than one type matches"
            )
        self.types: Tuple[T, ...] = tuple(dict.fromkeys(types))
        # User-declared types per column name; they skip inference.
        self.type_schema: Dict[Any, T] = {}
        precedence = {t: i for i, t in enumerate(self.types)}

        roots = [t for t in self.types if issubclass(t, Generic)]
        if len(roots) != 1:
            raise ValueError(
                f"A typeset needs exactly one root type (a subclass of Generic), "
                f"got {roots}"
            )
        self.root_node: T = roots[0]

        self._parent: Dict[T, Optional[T]] = {self.root_node: None}
        edges: Dict[T, List[TypeRelation]] = {t: [] for t in self.types}
        for t in self.types:
            identity_parents = []
            for relation in t.relations:
                if relation.related_type not in precedence:
                    # Relation to a type that is not active in this typeset
                    # (e.g. Text -> URL when URL detection is turned off).
                    continue
                edges[relation.related_type].append(relation)
                if not relation.inferential:
                    identity_parents.append(relation.related_type)

            if t is self.root_node:
                if identity_parents:
                    raise ValueError(f"The root type {t} cannot have a parent")
                continue
            if len(identity_parents) != 1:
                raise ValueError(
                    f"Type {t} needs exactly one IdentityRelation to a type in "
                    f"the typeset, got {identity_parents}"
                )
            self._parent[t] = identity_parents[0]

        for t in self.types:
            self._ancestors(t)  # raises on cycles

        def order(relation: TypeRelation) -> int:
            return precedence[relation.type]  # type: ignore[index]

        # sorted() is stable: declaration order breaks ties.
        self._edges: Dict[T, Tuple[TypeRelation, ...]] = {
            t: tuple(sorted(rels, key=order)) for t, rels in edges.items()
        }
        self._identity_edges: Dict[T, Tuple[TypeRelation, ...]] = {
            t: tuple(r for r in rels if not r.inferential)
            for t, rels in self._edges.items()
        }

    # Structure

    def _get_type(self, type_name: str) -> T:
        """The type with this name (case-insensitive)."""
        for t in self.types:
            if t.__name__.lower() == type_name.lower():
                return t
        raise ValueError(f"Type [{type_name}] not found.")

    def parent_of(self, type_: T) -> Optional[T]:
        """The identity parent of a type, ``None`` for the root."""
        return self._parent[type_]

    def _ancestors(self, type_: T) -> List[T]:
        chain: List[T] = []
        node = self._parent[type_]
        while node is not None:
            if node is type_ or node in chain:
                raise ValueError(f"Cyclical identity relations involving {type_}")
            chain.append(node)
            node = self._parent[node]
        return chain

    def topological_order(self) -> List[T]:
        """Types ordered so that every parent comes before its children."""
        return sorted(self.types, key=lambda t: len(self._ancestors(t)))

    def relations_from(
        self, type_: T, inferential: bool = True
    ) -> Tuple[TypeRelation, ...]:
        """Relations leaving ``type_``, in precedence order."""
        return self._edges[type_] if inferential else self._identity_edges[type_]

    # Traversal

    def _traverse(
        self,
        series: Any,
        inferential: bool,
        trace: Optional[List[dict]] = None,
    ) -> Tuple[Any, List[T], dict]:
        state: dict = {}
        node = self.root_node
        path = [node]
        while True:
            chosen = None
            matched = []
            for relation in self.relations_from(node, inferential):
                if relation.type in path:
                    continue
                if chosen is None:
                    if relation.is_relation(series, state):
                        chosen = relation
                        matched.append(str(relation.type))
                        if trace is None:
                            break
                # Only explain_type gets here: the remaining candidates are
                # evaluated for the trace on a copy of the state, so they
                # cannot change what the chosen transformer and the next steps
                # see. The traversal stays identical to infer/detect.
                elif relation.is_relation(series, _isolated(state)):
                    matched.append(str(relation.type))
            if trace is not None and matched:
                trace.append(
                    {
                        "from": str(node),
                        "matched": matched,
                        "chosen": str(chosen.type),  # type: ignore[union-attr]
                    }
                )
            if chosen is None:
                return series, path, state
            series = chosen.transform(series, state)
            state.pop(STEP_CACHE, None)
            node = chosen.type  # type: ignore[assignment]
            path.append(node)

    def detect(self, series: Any) -> Tuple[Any, List[T], dict]:
        """Walk identity relations only. Returns (series, path, state)."""
        return self._traverse(series, inferential=False)

    def infer(self, series: Any) -> Tuple[Any, List[T], dict]:
        """Walk identity and inference relations. Returns (cast series, path, state)."""
        return self._traverse(series, inferential=True)

    def detect_type(self, series: Any) -> T:
        return self.detect(series)[1][-1]

    def infer_type(self, series: Any) -> T:
        return self.infer(series)[1][-1]

    def cast_to_detected(self, series: Any) -> Any:
        return self.detect(series)[0]

    def cast_to_inferred(self, series: Any) -> Any:
        return self.infer(series)[0]

    def infer_and_cast(self, series: Any) -> Tuple[T, Any]:
        """Inferred type and cast series from a single traversal."""
        cast, path, _ = self.infer(series)
        return path[-1], cast

    def explain_type(self, series: Any, inferential: bool = True) -> List[dict]:
        """Every step of the traversal with all the candidates that matched.

        A step with more than one entry in ``matched`` is a column where the
        precedence order decided the type. Example::

            [{"from": "Unsupported", "matched": ["Text"], "chosen": "Text"},
             {"from": "Text", "matched": ["Numeric", "Categorical"],
              "chosen": "Numeric"}]
        """
        trace: List[dict] = []
        self._traverse(series, inferential=inferential, trace=trace)
        return trace

    def __repr__(self) -> str:
        return f"{type(self).__name__}({', '.join(map(str, self.types))})"
