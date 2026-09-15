"""Logic for alerting the user on possibly problematic patterns in the data (e.g. high number of zeros , constant
values, high correlations)."""

from enum import Enum, auto, unique
from typing import Dict, List, Optional, Set

import numpy as np
import pandas as pd

import html

from data_profiling.config import Settings
from data_profiling.model.correlations import perform_check_correlation
from data_profiling.utils.styles import get_alert_styles


def fmt_percent(value: float, edge_cases: bool = True) -> str:
    """Format a ratio as a percentage.

    Args:
        edge_cases: Check for edge cases?
        value: The ratio.

    Returns:
        The percentage with 1 point precision.
    """
    if edge_cases and round(value, 3) == 0 and value > 0:
        return "< 0.1%"
    if edge_cases and round(value, 3) == 1 and value < 1:
        return "> 99.9%"

    return f"{value*100:2.1f}%"


@unique
class AlertType(Enum):
    """Alert types"""

    CONSTANT = auto()
    """This variable has a constant value."""

    ZEROS = auto()
    """This variable contains zeros."""

    HIGH_CORRELATION = auto()
    """This variable is highly correlated."""

    HIGH_CARDINALITY = auto()
    """This variable has a high cardinality."""

    UNSUPPORTED = auto()
    """This variable is unsupported."""

    DUPLICATES = auto()
    """This variable contains duplicates."""

    NEAR_DUPLICATES = auto()
    """This variable contains duplicates."""

    SKEWED = auto()
    """This variable is highly skewed."""

    IMBALANCE = auto()
    """This variable is imbalanced."""

    MISSING = auto()
    """This variable contains missing values."""

    INFINITE = auto()
    """This variable contains infinite values."""

    TYPE_DATE = auto()
    """This variable is likely a datetime, but treated as categorical."""

    UNIQUE = auto()
    """This variable has unique values."""

    DIRTY_CATEGORY = auto()
    """This variable is a categories with potential fuzzy values, and for that reason might incur in consistency issues."""

    CONSTANT_LENGTH = auto()
    """This variable has a constant length."""

    REJECTED = auto()
    """Variables are rejected if we do not want to consider them for further analysis."""

    UNIFORM = auto()
    """The variable is uniformly distributed."""

    NON_STATIONARY = auto()
    """The variable is a non-stationary series."""

    SEASONAL = auto()
    """The variable is a seasonal time series."""

    EMPTY = auto()
    """The DataFrame is empty."""


class Alert:
    """An alert object (type, values, column)."""

    _anchor_id: Optional[str] = None

    def __init__(
        self,
        alert_type: AlertType,
        values: Optional[Dict] = None,
        column_name: Optional[str] = None,
        fields: Optional[Set] = None,
        is_empty: bool = False,
    ):
        self.fields = fields or set()
        self.alert_type = alert_type
        self.values = values or {}
        self.column_name = column_name
        self._is_empty = is_empty
        self._styles = get_alert_styles()

    @property
    def alert_type_name(self) -> str:
        return self.alert_type.name.replace("_", " ").capitalize()

    @property
    def anchor_id(self) -> Optional[str]:
        if self._anchor_id is None:
            self._anchor_id = str(hash(self.column_name))
        return self._anchor_id

    def fmt(self) -> str:
        # TODO: render in template
        style = self._styles.get(self.alert_type.name.lower(), "secondary")
        hint = ""

        if self.alert_type == AlertType.HIGH_CORRELATION and self.values is not None:
            num = len(self.values["fields"])
            title = ", ".join(self.values["fields"])
            corr = self.values["corr"]
            safe_title = html.escape(title)
            hint = f'data-bs-toggle="tooltip" data-bs-placement="right" data-bs-title="This variable has a high {corr} correlation with {num} fields: {safe_title}"'

        return (
            f'<span class="badge text-bg-{style}" {hint}>{self.alert_type_name}</span>'
        )

    def _get_description(self) -> str:
        """Return a human level description of the alert.

        Returns:
            str: alert description
        """
        alert_type = self.alert_type.name
        column = self.column_name
        return f"[{alert_type}] alert on column {column}"

    def __repr__(self):
        return self._get_description()


class ConstantLengthAlert(Alert):
    def __init__(
        self,
        values: Optional[Dict] = None,
        column_name: Optional[str] = None,
        is_empty: bool = False,
    ):
        super().__init__(
            alert_type=AlertType.CONSTANT_LENGTH,
            values=values,
            column_name=column_name,
            fields={"composition_min_length", "composition_max_length"},
            is_empty=is_empty,
        )

    def _get_description(self) -> str:
        return f"[{self.column_name}] has a constant length"


class ConstantAlert(Alert):
    def __init__(
        self,
        values: Optional[Dict] = None,
        column_name: Optional[str] = None,
        is_empty: bool = False,
    ):
        super().__init__(
            alert_type=AlertType.CONSTANT,
            values=values,
            column_name=column_name,
            fields={"n_distinct"},
            is_empty=is_empty,
        )

    def _get_description(self) -> str:
        return f"[{self.column_name}] has a constant value"


class DuplicatesAlert(Alert):
    def __init__(
        self,
        values: Optional[Dict] = None,
        column_name: Optional[str] = None,
        is_empty: bool = False,
    ):
        super().__init__(
            alert_type=AlertType.DUPLICATES,
            values=values,
            column_name=column_name,
            fields={"n_duplicates"},
            is_empty=is_empty,
        )

    def _get_description(self) -> str:
        if self.values is not None:
            return f"Dataset has {self.values['n_duplicates']} ({fmt_percent(self.values['p_duplicates'])}) duplicate rows"
        else:
            return "Dataset has no duplicated rows"


class NearDuplicatesAlert(Alert):
    def __init__(
        self,
        values: Optional[Dict] = None,
        column_name: Optional[str] = None,
        is_empty: bool = False,
    ):
        super().__init__(
            alert_type=AlertType.NEAR_DUPLICATES,
            values=values,
            column_name=column_name,
            fields={"n_near_dups"},
            is_empty=is_empty,
        )

    def _get_description(self) -> str:
        if self.values is not None:
            return f"Dataset has {self.values['n_near_dups']} ({fmt_percent(self.values['p_near_dups'])}) near duplicate rows"
        else:
            return "Dataset has no near duplicated rows"


class EmptyAlert(Alert):
    def __init__(
        self,
        values: Optional[Dict] = None,
        column_name: Optional[str] = None,
        is_empty: bool = False,
    ):
        super().__init__(
            alert_type=AlertType.EMPTY,
            values=values,
            column_name=column_name,
            fields={"n"},
            is_empty=is_empty,
        )

    def _get_description(self) -> str:
        return "Dataset is empty"


class HighCardinalityAlert(Alert):
    def __init__(
        self,
        values: Optional[Dict] = None,
        column_name: Optional[str] = None,
        is_empty: bool = False,
    ):
        super().__init__(
            alert_type=AlertType.HIGH_CARDINALITY,
            values=values,
            column_name=column_name,
            fields={"n_distinct"},
            is_empty=is_empty,
        )

    def _get_description(self) -> str:
        if self.values is not None:
            return f"[{self.column_name}] has {self.values['n_distinct']:} ({fmt_percent(self.values['p_distinct'])}) distinct values"
        else:
            return f"[{self.column_name}] has a high cardinality"


class DirtyCategoryAlert(Alert):
    def __init__(
        self,
        values: Optional[Dict] = None,
        column_name: Optional[str] = None,
        is_empty: bool = False,
    ):
        super().__init__(
            alert_type=AlertType.DIRTY_CATEGORY,
            values=values,
            column_name=column_name,
            fields={"n_fuzzy_vals"},
            is_empty=is_empty,
        )

    def _get_description(self) -> str:
        if self.values is not None:
            return f"[{self.column_name}] has {self.values['n_fuzzy_vals']} fuzzy values: {fmt_percent(self.values['p_fuzzy_vals'])} per category"
        else:
            return f"[{self.column_name}] no dirty categories values."


class HighCorrelationAlert(Alert):
    def __init__(
        self,
        values: Optional[Dict] = None,
        column_name: Optional[str] = None,
        is_empty: bool = False,
    ):
        super().__init__(
            alert_type=AlertType.HIGH_CORRELATION,
            values=values,
            column_name=column_name,
            is_empty=is_empty,
        )

    def _get_description(self) -> str:
        if self.values is not None:
            description = f"[{self.column_name}] is highly {self.values['corr']} correlated with [{self.values['fields'][0]}]"
            if len(self.values["fields"]) > 1:
                description += f" and {len(self.values['fields']) - 1} other fields"
        else:
            return (
                f"[{self.column_name}] has a high correlation with one or more colums"
            )
        return description


class ImbalanceAlert(Alert):
    def __init__(
        self,
        values: Optional[Dict] = None,
        column_name: Optional[str] = None,
        is_empty: bool = False,
    ):
        super().__init__(
            alert_type=AlertType.IMBALANCE,
            values=values,
            column_name=column_name,
            fields={"imbalance"},
            is_empty=is_empty,
        )

    def _get_description(self) -> str:
        description = f"[{self.column_name}] is highly imbalanced"
        if self.values is not None:
            return description + f" ({self.values['imbalance']})"
        else:
            return description


class InfiniteAlert(Alert):
    def __init__(
        self,
        values: Optional[Dict] = None,
        column_name: Optional[str] = None,
        is_empty: bool = False,
    ):
        super().__init__(
            alert_type=AlertType.INFINITE,
            values=values,
            column_name=column_name,
            fields={"p_infinite", "n_infinite"},
            is_empty=is_empty,
        )

    def _get_description(self) -> str:
        if self.values is not None:
            return f"[{self.column_name}] has {self.values['n_infinite']} ({fmt_percent(self.values['p_infinite'])}) infinite values"
        else:
            return f"[{self.column_name}] has infinite values"


class MissingAlert(Alert):
    def __init__(
        self,
        values: Optional[Dict] = None,
        column_name: Optional[str] = None,
        is_empty: bool = False,
    ):
        super().__init__(
            alert_type=AlertType.MISSING,
            values=values,
            column_name=column_name,
            fields={"p_missing", "n_missing"},
            is_empty=is_empty,
        )

    def _get_description(self) -> str:
        if self.values is not None:
            return f"[{self.column_name}] {self.values['n_missing']} ({fmt_percent(self.values['p_missing'])}) missing values"
        else:
            return f"[{self.column_name}] has missing values"


class NonStationaryAlert(Alert):
    def __init__(
        self,
        values: Optional[Dict] = None,
        column_name: Optional[str] = None,
        is_empty: bool = False,
    ):
        super().__init__(
            alert_type=AlertType.NON_STATIONARY,
            values=values,
            column_name=column_name,
            is_empty=is_empty,
        )

    def _get_description(self) -> str:
        return f"[{self.column_name}] is non stationary"


class SeasonalAlert(Alert):
    def __init__(
        self,
        values: Optional[Dict] = None,
        column_name: Optional[str] = None,
        is_empty: bool = False,
    ):
        super().__init__(
            alert_type=AlertType.SEASONAL,
            values=values,
            column_name=column_name,
            is_empty=is_empty,
        )

    def _get_description(self) -> str:
        return f"[{self.column_name}] is seasonal"


class SkewedAlert(Alert):
    def __init__(
        self,
        values: Optional[Dict] = None,
        column_name: Optional[str] = None,
        is_empty: bool = False,
    ):
        super().__init__(
            alert_type=AlertType.SKEWED,
            values=values,
            column_name=column_name,
            fields={"skewness"},
            is_empty=is_empty,
        )

    def _get_description(self) -> str:
        description = f"[{self.column_name}] is highly skewed"
        if self.values is not None:
            return description + f"(\u03b31 = {self.values['skewness']})"
        else:
            return description


class TypeDateAlert(Alert):
    def __init__(
        self,
        values: Optional[Dict] = None,
        column_name: Optional[str] = None,
        is_empty: bool = False,
    ):
        super().__init__(
            alert_type=AlertType.TYPE_DATE,
            values=values,
            column_name=column_name,
            is_empty=is_empty,
        )

    def _get_description(self) -> str:
        return f"[{self.column_name}] only contains datetime values, but is categorical. Consider applying `pd.to_datetime()`"


class UniformAlert(Alert):
    def __init__(
        self,
        values: Optional[Dict] = None,
        column_name: Optional[str] = None,
        is_empty: bool = False,
    ):
        super().__init__(
            alert_type=AlertType.UNIFORM,
            values=values,
            column_name=column_name,
            is_empty=is_empty,
        )

    def _get_description(self) -> str:
        return f"[{self.column_name}] is uniformly distributed"


class UniqueAlert(Alert):
    def __init__(
        self,
        values: Optional[Dict] = None,
        column_name: Optional[str] = None,
        is_empty: bool = False,
    ):
        super().__init__(
            alert_type=AlertType.UNIQUE,
            values=values,
            column_name=column_name,
            fields={"n_distinct", "p_distinct", "n_unique", "p_unique"},
            is_empty=is_empty,
        )

    def _get_description(self) -> str:
        return f"[{self.column_name}] has unique values"


class UnsupportedAlert(Alert):
    def __init__(
        self,
        values: Optional[Dict] = None,
        column_name: Optional[str] = None,
        is_empty: bool = False,
    ):
        super().__init__(
            alert_type=AlertType.UNSUPPORTED,
            values=values,
            column_name=column_name,
            is_empty=is_empty,
        )

    def _get_description(self) -> str:
        return f"[{self.column_name}] is an unsupported type, check if it needs cleaning or further analysis"


class ZerosAlert(Alert):
    def __init__(
        self,
        values: Optional[Dict] = None,
        column_name: Optional[str] = None,
        is_empty: bool = False,
    ):
        super().__init__(
            alert_type=AlertType.ZEROS,
            values=values,
            column_name=column_name,
            fields={"n_zeros", "p_zeros"},
            is_empty=is_empty,
        )

    def _get_description(self) -> str:
        if self.values is not None:
            return f"[{self.column_name}] has {self.values['n_zeros']} ({fmt_percent(self.values['p_zeros'])}) zeros"
        else:
            return f"[{self.column_name}] has predominantly zeros"


class RejectedAlert(Alert):
    def __init__(
        self,
        values: Optional[Dict] = None,
        column_name: Optional[str] = None,
        is_empty: bool = False,
    ):
        super().__init__(
            alert_type=AlertType.REJECTED,
            values=values,
            column_name=column_name,
            is_empty=is_empty,
        )

    def _get_description(self) -> str:
        return f"[{self.column_name}] was rejected"


def check_table_alerts(table: dict) -> List[Alert]:
    """Checks the overall dataset for alerts.

    Args:
        table: Overall dataset statistics.

    Returns:
        A list of alerts.
    """
    alerts: List[Alert] = []
    if alert_value(table.get("n_duplicates", np.nan)):
        alerts.append(
            DuplicatesAlert(
                values=table,
            )
        )
    if table["n"] == 0:
        alerts.append(
            EmptyAlert(
                values=table,
            )
        )
    return alerts


def numeric_alerts(config: Settings, summary: dict) -> List[Alert]:
    alerts: List[Alert] = []

    # Skewness
    if skewness_alert(summary["skewness"], config.vars.num.skewness_threshold):
        alerts.append(SkewedAlert(summary))

    # Infinite values
    if alert_value(summary["p_infinite"]):
        alerts.append(InfiniteAlert(summary))

    # Zeros
    if alert_value(summary["p_zeros"]):
        alerts.append(ZerosAlert(summary))

    if (
        "chi_squared" in summary
        and summary["chi_squared"]["pvalue"] > config.vars.num.chi_squared_threshold
    ):
        alerts.append(UniformAlert())

    return alerts


def timeseries_alerts(config: Settings, summary: dict) -> List[Alert]:
    alerts: List[Alert] = numeric_alerts(config, summary)

    if not summary["stationary"]:
        alerts.append(NonStationaryAlert())

    if summary["seasonal"]:
        alerts.append(SeasonalAlert())

    return alerts


def categorical_alerts(config: Settings, summary: dict) -> List[Alert]:
    alerts: List[Alert] = []

    # High cardinality
    if summary.get("n_distinct", np.nan) > config.vars.cat.cardinality_threshold:
        alerts.append(HighCardinalityAlert(summary))

    if (
        "chi_squared" in summary
        and summary["chi_squared"]["pvalue"] > config.vars.cat.chi_squared_threshold
    ):
        alerts.append(UniformAlert())

    if summary.get("date_warning"):
        alerts.append(TypeDateAlert())

    # Constant length
    if "composition" in summary and summary["min_length"] == summary["max_length"]:
        alerts.append(ConstantLengthAlert())

    # Imbalance
    if (
        "imbalance" in summary
        and summary["imbalance"] > config.vars.cat.imbalance_threshold
    ):
        alerts.append(ImbalanceAlert(summary))
    return alerts


def boolean_alerts(config: Settings, summary: dict) -> List[Alert]:
    alerts: List[Alert] = []

    if (
        "imbalance" in summary
        and summary["imbalance"] > config.vars.bool.imbalance_threshold
    ):
        alerts.append(ImbalanceAlert())
    return alerts


def generic_alerts(summary: dict) -> List[Alert]:
    alerts: List[Alert] = []

    # Missing
    if alert_value(summary["p_missing"]):
        alerts.append(MissingAlert())

    return alerts


def supported_alerts(summary: dict) -> List[Alert]:
    alerts: List[Alert] = []

    if summary.get("n_distinct", np.nan) == summary["n"]:
        alerts.append(UniqueAlert())
    if summary.get("n_distinct", np.nan) == 1:
        alerts.append(ConstantAlert(summary))
    return alerts


def unsupported_alerts() -> List[Alert]:
    alerts: List[Alert] = [
        UnsupportedAlert(),
        RejectedAlert(),
    ]
    return alerts


def check_variable_alerts(config: Settings, col: str, description: dict) -> List[Alert]:
    """Checks individual variables for alerts.

    Args:
        col: The column name that is checked.
        description: The series description.

    Returns:
        A list of alerts.
    """
    alerts: List[Alert] = []

    alerts += generic_alerts(description)

    if description["type"] == "Unsupported":
        alerts += unsupported_alerts()
    else:
        alerts += supported_alerts(description)

        if description["type"] == "Categorical":
            alerts += categorical_alerts(config, description)
        if description["type"] == "Numeric":
            alerts += numeric_alerts(config, description)
        if description["type"] == "TimeSeries":
            alerts += timeseries_alerts(config, description)
        if description["type"] == "Boolean":
            alerts += boolean_alerts(config, description)

    for idx in range(len(alerts)):
        alerts[idx].column_name = col
        alerts[idx].values = description
    return alerts


def check_correlation_alerts(config: Settings, correlations: dict) -> List[Alert]:
    alerts: List[Alert] = []

    correlations_consolidated = {}
    for corr, matrix in correlations.items():
        if config.correlations[corr].warn_high_correlations:
            threshold = config.correlations[corr].threshold
            correlated_mapping = perform_check_correlation(matrix, threshold)
            for col, fields in correlated_mapping.items():
                set(fields).update(set(correlated_mapping.get(col, [])))
                correlations_consolidated[col] = fields

    if len(correlations_consolidated) > 0:
        for col, fields in correlations_consolidated.items():
            alerts.append(
                HighCorrelationAlert(
                    column_name=col,
                    values={"corr": "overall", "fields": fields},
                )
            )
    return alerts


def get_alerts(
    config: Settings, table_stats: dict, series_description: dict, correlations: dict
) -> List[Alert]:
    alerts: List[Alert] = check_table_alerts(table_stats)
    for col, description in series_description.items():
        alerts += check_variable_alerts(config, col, description)
    alerts += check_correlation_alerts(config, correlations)
    alerts.sort(key=lambda alert: str(alert.alert_type))
    return alerts


def alert_value(value: float) -> bool:
    return not pd.isna(value) and value > 0.01


def skewness_alert(v: float, threshold: int) -> bool:
    return not pd.isna(v) and (v < (-1 * threshold) or v > threshold)


def type_date_alert(series: pd.Series) -> bool:
    from dateutil.parser import ParserError, parse

    try:
        series.apply(parse)
    except ParserError:
        return False
    else:
        return True
