"""Semantic column roles.

A column's *type* (Numeric, Text, ...) says what its values look like. Its
*role* says what the column means for analysis: an identifier, a code, a
measure, a dimension, ... Roles are inferred from the computed statistics with
explicit rules. Each inferred role carries the rule conditions that produced it
(``basis``), so humans and agents can judge how much to trust it. When no rule
is decisive the role is ``unknown`` and ``candidates`` lists the plausible roles.

Roles never change what is computed, only how a profile is presented and
serialised, so they can be overridden after profiling without recomputation.

User overrides come from ``config.variables.roles`` and from dict-valued entries
in ``config.variables.descriptions`` (``{"col": {"description": ..., "role": ...}}``).
"""

import re
import warnings
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Mapping, Optional

ROLES = (
    "identifier",
    "code",
    "measure",
    "dimension",
    "timestamp",
    "free_text",
    "constant",
    "empty",
    "unknown",
)

#: Integers with at most this many distinct values are ambiguous between a
#: dimension (category codes, ratings) and a measure (counts).
LOW_CARDINALITY_INTEGER = 20
#: Non-unique strings with at most this many distinct values are dimensions.
LOW_CARDINALITY_STRING = 50
#: Number of observed values inspected for value-pattern rules.
_PATTERN_SAMPLE = 200

_ID_TOKENS = {"id", "uuid", "guid", "pk", "key"}
_CODE_TOKENS = {
    "zip",
    "zipcode",
    "postal",
    "postcode",
    "code",
    "sku",
    "iso",
    "ean",
    "isbn",
}
_MEASURE_TOKENS = {
    "price",
    "amount",
    "qty",
    "quantity",
    "count",
    "total",
    "sum",
    "cost",
    "revenue",
    "weight",
    "height",
    "age",
    "score",
    "duration",
    "balance",
    "salary",
    "income",
}
_YEAR_TOKENS = {"year", "yr"}
_EMAIL = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")
_STRING_DTYPES = ("object", "string", "str", "category")


@dataclass
class ColumnRole:
    role: str
    basis: List[str]
    candidates: List[str] = field(default_factory=list)

    @property
    def needs_review(self) -> bool:
        """Unknown roles and weak fallbacks should be reviewed by a person or agent."""
        return self.role == "unknown" or self.basis == ["default"]

    def to_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {"role": self.role, "role_basis": list(self.basis)}
        if self.candidates:
            d["role_candidates"] = list(self.candidates)
        return d


def _name_tokens(name: Any) -> List[str]:
    s = str(name)
    s = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", s)
    return [t for t in re.split(r"[^0-9a-zA-Z]+", s.lower()) if t]


def _name_hint(name: Any) -> Optional[str]:
    """Return 'identifier' / 'code' / 'measure' / 'year' from the column name, if any."""
    tokens = _name_tokens(name)
    if not tokens:
        return None
    last = tokens[-1]
    if last in _ID_TOKENS or (len(tokens) == 1 and tokens[0] in _ID_TOKENS):
        return "identifier"
    if any(t in _CODE_TOKENS for t in tokens):
        return "code"
    if any(t in _YEAR_TOKENS for t in tokens):
        return "year"
    if any(t in _MEASURE_TOKENS for t in tokens):
        return "measure"
    return None


def _name_basis(name: Any, hint: str) -> str:
    tokens = _name_tokens(name)
    vocab = {
        "identifier": _ID_TOKENS,
        "code": _CODE_TOKENS,
        "measure": _MEASURE_TOKENS,
        "year": _YEAR_TOKENS,
    }[hint]
    token = next((t for t in reversed(tokens) if t in vocab), hint)
    return f"name={token}"


def _is_integer_valued(summary: Mapping[str, Any]) -> bool:
    try:
        mn, mx = float(summary["min"]), float(summary["max"])
    except (KeyError, TypeError, ValueError):
        return False
    if not (mn.is_integer() and mx.is_integer()):
        return False
    vc = summary.get("value_counts_without_nan")
    if vc is not None and len(vc) > 0:
        try:
            return all(float(v).is_integer() for v in list(vc.index[:_PATTERN_SAMPLE]))
        except (TypeError, ValueError):
            return False
    return True


def _observed_values(summary: Mapping[str, Any]) -> List[str]:
    vc = summary.get("value_counts_without_nan")
    if vc is None or len(vc) == 0:
        return []
    return [str(v) for v in list(vc.index[:_PATTERN_SAMPLE])]


def _scalar(value: Any) -> Any:
    return value[0] if isinstance(value, list) else value


def infer_role(
    name: Any, summary: Mapping[str, Any], source_dtype: Optional[str] = None
) -> ColumnRole:
    """Infer the role of one column from its summary statistics.

    Args:
        name: column name, used only as supporting evidence.
        summary: the column's entry in ``BaseDescription.variables``.
        source_dtype: dtype of the column before type inference (e.g. ``object``),
            when known. Lets numeric-looking strings be recognised as codes.
    """
    vtype = str(summary.get("type", "Unsupported"))
    count = summary.get("count", summary.get("n", 0)) or 0
    n_distinct = summary.get("n_distinct", 0) or 0
    is_unique = bool(summary.get("is_unique", False))
    hint = _name_hint(name)

    if count == 0:
        return ColumnRole("empty", ["all_missing"])
    if n_distinct == 1:
        return ColumnRole("constant", ["n_distinct=1"])
    if vtype == "DateTime":
        return ColumnRole("timestamp", ["type=datetime"])
    if vtype == "Boolean":
        return ColumnRole("dimension", ["type=boolean"])
    if vtype == "TimeSeries":
        return ColumnRole("measure", ["type=timeseries"])

    if vtype == "Numeric":
        return _infer_numeric(name, summary, source_dtype, hint, n_distinct, is_unique)

    if vtype in ("Categorical", "Text"):
        return _infer_string(
            name, summary, vtype, hint, n_distinct, is_unique, source_dtype
        )

    return ColumnRole("unknown", [f"type={vtype.lower()}"])


def _infer_numeric(
    name: Any,
    summary: Mapping[str, Any],
    source_dtype: Optional[str],
    hint: Optional[str],
    n_distinct: int,
    is_unique: bool,
) -> ColumnRole:
    integer = _is_integer_valued(summary)
    name_basis = [_name_basis(name, hint)] if hint else []
    stored_as_string = bool(
        source_dtype and str(source_dtype).lower().startswith(_STRING_DTYPES)
    )

    if integer and stored_as_string:
        mn, mx = int(float(summary["min"])), int(float(summary["max"]))
        if len(str(abs(mn))) == len(str(abs(mx))):
            return ColumnRole("code", ["stored_as_string", "fixed_width"])
        if hint in ("code", "identifier"):
            return ColumnRole("code", ["stored_as_string"] + name_basis)

    if integer and is_unique:
        order = (
            "monotonic" if summary.get("monotonic", 0) not in (0, None) else "unordered"
        )
        return ColumnRole("identifier", ["unique", "integer", order] + name_basis)

    if integer and n_distinct == 2:
        try:
            if {float(summary["min"]), float(summary["max"])} == {0.0, 1.0}:
                return ColumnRole("dimension", ["binary_0_1"])
        except (KeyError, TypeError, ValueError):
            pass

    if integer and hint == "code":
        return ColumnRole("code", ["integer"] + name_basis)

    if integer and n_distinct <= LOW_CARDINALITY_INTEGER:
        return _low_cardinality_integer(
            float(summary["min"]), float(summary["max"]), hint, name_basis
        )

    if hint == "measure":
        return ColumnRole("measure", name_basis)
    return ColumnRole("measure", ["default"])


def _low_cardinality_integer(
    mn: float, mx: float, hint: Optional[str], name_basis: List[str]
) -> ColumnRole:
    basis = ["low_cardinality_integer"]
    if hint == "measure":
        return ColumnRole("measure", basis + name_basis)
    if hint == "identifier":
        # A low-cardinality *_id is a reference to a category, not a row key.
        return ColumnRole("dimension", basis + name_basis)
    if 1900 <= mn and mx <= 2100:
        basis.append("year_range")
    return ColumnRole("unknown", basis + name_basis, ["dimension", "measure"])


def _is_numeric_dtype(source_dtype: Optional[str]) -> bool:
    s = str(source_dtype or "").lower()
    return s.startswith(
        (
            "int",
            "uint",
            "float",
            "bool",
            "long",
            "short",
            "double",
            "byte",
            # Spark SQL type names
            "bigint",
            "smallint",
            "tinyint",
            "decimal",
        )
    )


def _infer_string(
    name: Any,
    summary: Mapping[str, Any],
    vtype: str,
    hint: Optional[str],
    n_distinct: int,
    is_unique: bool,
    source_dtype: Optional[str] = None,
) -> ColumnRole:
    values = _observed_values(summary)
    lengths = {len(v) for v in values}
    name_basis = [_name_basis(name, hint)] if hint else []

    if _is_numeric_dtype(source_dtype) and values:
        # Numbers treated as categories (few distinct values): keep integer semantics.
        try:
            numbers = [float(v) for v in values]
        except ValueError:
            numbers = []
        if numbers and all(x.is_integer() for x in numbers):
            if set(numbers) <= {0.0, 1.0}:
                return ColumnRole("dimension", ["binary_0_1"])
            if hint == "code":
                return ColumnRole("code", ["integer"] + name_basis)
            return _low_cardinality_integer(
                min(numbers), max(numbers), hint, name_basis
            )
        return ColumnRole("dimension", ["type=categorical"])

    if (
        values
        and all(v.isdigit() for v in values)
        and len(lengths) == 1
        and (lengths.pop() >= 3 or any(v.startswith("0") for v in values))
    ):
        return ColumnRole("code", ["digits_only", "fixed_width"])
    lengths = {len(v) for v in values}
    if is_unique and values and len(lengths) == 1:
        return ColumnRole("identifier", ["unique", "fixed_length"])
    if is_unique and values and all(_EMAIL.match(v) for v in values):
        return ColumnRole("identifier", ["unique", "pattern=email"])
    if is_unique and hint == "identifier":
        return ColumnRole("identifier", ["unique"] + name_basis)
    if is_unique:
        return ColumnRole(
            "unknown", ["unique", "variable_length"], ["identifier", "free_text"]
        )
    if hint == "code" and n_distinct > LOW_CARDINALITY_STRING:
        return ColumnRole("code", name_basis)
    if vtype == "Categorical":
        return ColumnRole("dimension", ["type=categorical"])
    if n_distinct <= LOW_CARDINALITY_STRING:
        return ColumnRole("dimension", [f"low_cardinality(n={n_distinct})"])
    return ColumnRole("free_text", ["type=text"])


def column_description(config: Any, column: Any) -> str:
    """Description text for a column from ``variables.descriptions``.

    Entries may be a plain string (the description) or a dict with a
    ``description`` key (the column-definition form, which may also set ``role``).
    """
    value = config.variables.descriptions.get(column, "")
    if isinstance(value, Mapping):
        return str(value.get("description", "") or "")
    return "" if value is None else str(value)


def user_roles(config: Any) -> Dict[str, str]:
    """Roles set by the user: dict-valued descriptions first, ``variables.roles`` wins."""
    roles: Dict[str, str] = {}
    for col, value in config.variables.descriptions.items():
        if isinstance(value, Mapping) and value.get("role"):
            roles[col] = value["role"]
    roles.update(getattr(config.variables, "roles", {}) or {})
    return roles


def validate_roles(
    roles: Mapping[str, str],
    columns: Optional[Iterable[str]] = None,
    types: Optional[Mapping[str, str]] = None,
) -> None:
    """Validate user roles.

    Raises:
        ValueError: on a role outside :data:`ROLES`.

    Warns on columns not present in the profile (a definition file may cover
    several versions of a table) and on unusual role/type combinations.
    """
    for col, role in roles.items():
        if role not in ROLES:
            raise ValueError(
                f"Invalid role {role!r} for column {col!r}. Valid roles: {', '.join(ROLES)}."
            )
    if columns is not None:
        known = set(columns)
        missing = [c for c in roles if c not in known]
        if missing:
            warnings.warn(
                f"Roles set for columns that are not in the profile: {missing}",
                stacklevel=3,
            )
    if types:
        unusual = {
            "timestamp": {"Numeric", "Categorical", "Text"},
            "free_text": {"Numeric", "Boolean", "DateTime"},
            "measure": {"Text", "Boolean"},
        }
        for col, role in roles.items():
            vtype = _scalar(types.get(col))
            if vtype in unusual.get(role, set()):
                warnings.warn(
                    f"Role {role!r} on {vtype} column {col!r} is unusual; keeping it as set.",
                    stacklevel=3,
                )


def infer_roles(
    variables: Mapping[str, Mapping[str, Any]],
    overrides: Optional[Mapping[str, str]] = None,
    source_dtypes: Optional[Mapping[str, str]] = None,
) -> Dict[str, ColumnRole]:
    """Resolve the role of every column: user overrides first, then rules."""
    overrides = overrides or {}
    source_dtypes = source_dtypes or {}
    roles: Dict[str, ColumnRole] = {}
    for col, summary in variables.items():
        if col in overrides:
            roles[col] = ColumnRole(overrides[col], ["user"])
            continue
        if isinstance(summary.get("type"), list):
            # Merged comparison summaries hold one value per report.
            roles[col] = ColumnRole("unknown", ["comparison"])
            continue
        roles[col] = infer_role(col, summary, source_dtypes.get(col))
    return roles


def source_dtypes_of(df: Any) -> Dict[str, str]:
    """Column dtypes before type inference, for pandas or Spark DataFrames."""
    if df is None:
        return {}
    try:
        import pandas as pd

        if isinstance(df, pd.DataFrame):
            return {col: str(dtype) for col, dtype in df.dtypes.items()}
        return {col: str(dtype) for col, dtype in df.dtypes}  # Spark: list of tuples
    except Exception:  # pragma: no cover - never let dtype lookup break profiling
        return {}
