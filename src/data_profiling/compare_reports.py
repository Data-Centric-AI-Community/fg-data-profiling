import json
import warnings
from dataclasses import asdict
from typing import Any, List, Optional, Tuple, Union

import pandas as pd
from dacite import from_dict

from data_profiling.config import Correlation, Settings
from data_profiling.model import BaseDescription
from data_profiling.model.alerts import Alert
from data_profiling.model.roles import source_dtypes_of
from data_profiling.profile_report import ProfileReport


def _should_wrap(v1: Any, v2: Any) -> bool:
    if isinstance(v1, (list, dict)):
        return False

    if isinstance(v1, pd.DataFrame) and isinstance(v2, pd.DataFrame):
        return v1.equals(v2)
    if isinstance(v1, pd.Series) and isinstance(v2, pd.Series):
        return v1.equals(v2)

    try:
        return v1 == v2
    except ValueError:
        return False


def _update_merge_dict(d1: Any, d2: Any) -> dict:
    # Unwrap d1 and d2 in new dictionary to keep non-shared keys with **d1, **d2
    # Next unwrap a dict that treats shared keys
    # If two keys have an equal value, we take that value as new value
    # If the values are not equal, we recursively merge them

    return {
        **d1,
        **d2,
        **{
            k: [d1[k], d2[k]]
            if _should_wrap(d1[k], d2[k])
            else _update_merge_mixed(d1[k], d2[k])
            for k in {*d1} & {*d2}
        },
    }


def _update_merge_seq(d1: Any, d2: Any) -> Union[list, tuple]:
    # This case happens when values are merged
    # It bundle values in a list, making sure
    # to flatten them if they are already lists

    if isinstance(d1, list) and isinstance(d2, list):
        # This is the tuple for alerts
        return d1, d2
    elif isinstance(d1, tuple) and isinstance(d2, list):
        return (*d1, d2)
    else:
        return [
            *(d1 if isinstance(d1, list) else [d1]),
            *(d2 if isinstance(d2, list) else [d2]),
        ]


def _update_merge_mixed(d1: Any, d2: Any) -> Union[dict, list, tuple]:
    if isinstance(d1, dict) and isinstance(d2, dict):
        return _update_merge_dict(d1, d2)
    else:
        return _update_merge_seq(d1, d2)


def _update_merge(d1: Optional[dict], d2: dict) -> dict:
    # For convenience in the loop, allow d1 to be empty initially
    if d1 is None:
        return d2

    if not isinstance(d1, dict) or not isinstance(d2, dict):
        raise TypeError(
            "Both arguments need to be of type dictionary (ProfileReport.description_set)"
        )

    return _update_merge_dict(d1, d2)


def _placeholders(reports: List[BaseDescription]) -> None:
    """Generates placeholders in the dataset descriptions where needed"""

    keys = {v for r in reports for v in r.scatter}
    type_keys = {v for r in reports for v in r.table["types"]}
    for report in reports:
        # Interactions
        for k1 in keys:
            for k2 in keys:
                if k1 not in report.scatter:
                    report.scatter[k1] = {}
                if k2 not in report.scatter[k1]:
                    report.scatter[k1][k2] = ""

        # Types
        for type_key in type_keys:
            if type_key not in report.table["types"]:
                report.table["types"][type_key] = 0


def _update_titles(reports: List[ProfileReport]) -> None:
    """Redefine title of reports with the default one."""
    for idx, report in enumerate(reports):
        if report.config.title == "YData Profiling Report":
            report.config.title = f"Dataset {chr(65 + idx)}"


def _compare_title(titles: List[str]) -> str:
    if all(titles[0] == title for title in titles[1:]):
        return titles[0]
    else:
        title = ", ".join(titles[:-1])
        return f"<em>Comparing</em> {title} <em>and</em> {titles[-1]}"


def _compare_profile_report_preprocess(
    reports: List[ProfileReport],
    config: Optional[Settings] = None,
) -> Tuple[List[str], List[BaseDescription]]:
    # Use titles as labels
    labels = [report.config.title for report in reports]

    # Use color per report if not custom set
    if config is None:
        if len(reports[0].config.html.style.primary_colors) > 1:
            for idx, report in enumerate(reports):
                report.config.html.style.primary_colors = [
                    report.config.html.style.primary_colors[idx]
                ]
    else:
        if len(config.html.style.primary_colors) > 1:
            for idx, report in enumerate(reports):
                report.config.html.style.primary_colors = (
                    config.html.style.primary_colors
                )

    # Obtain description sets
    descriptions = [report.get_description() for report in reports]
    for label, description in zip(labels, descriptions):
        description.analysis.title = label

    return labels, descriptions


def _compare_dataset_description_preprocess(
    reports: List[BaseDescription],
) -> Tuple[List[str], List[BaseDescription]]:
    labels = [report.analysis.title for report in reports]
    return labels, reports


def validate_reports(
    reports: Union[List[ProfileReport], List[BaseDescription]], configs: List[dict]
) -> None:
    """Validate if the reports are comparable.

    Args:
        reports: two reports to compare
                 input may either be a ProfileReport, or the summary obtained from report.get_description()
    """
    if len(reports) < 2:
        raise ValueError("At least two reports are required for this comparison")

    if len(reports) > 2:
        warnings.warn(
            "Comparison of more than two reports is not supported. "
            "Reports may be produced, but may yield unexpected formatting."
        )

    report_types = [c.vars.timeseries.active for c in configs]  # type: ignore
    if all(report_types) != any(report_types):
        raise ValueError(
            "Comparison between timeseries and tabular reports is not supported."
        )

    if isinstance(reports[0], ProfileReport):
        is_df_available = [r.df is not None for r in reports]  # type: ignore
        if not all(is_df_available):
            raise ValueError("Reports where not initialized with a DataFrame.")

    if isinstance(reports[0], ProfileReport):
        features = [set(r.df.columns) for r in reports]  # type: ignore
    else:
        features = [set(r.variables.keys()) for r in reports]  # type: ignore

    if not all(features[0] == x for x in features):
        warnings.warn(
            "The datasets being profiled have a different set of columns. "
            "Only the left side profile will be calculated."
        )


def _apply_config(description: BaseDescription, config: Settings) -> BaseDescription:
    """Apply the configuration for visualilzation purposes.

    This handles the cases in which the report description
    was computed prior to comparison with a different config

    Args:
        description: report summary
        config: the settings object for the ProfileReport

    Returns:
        the updated description
    """
    description.missing = {
        k: v for k, v in description.missing.items() if config.missing_diagrams[k]
    }

    description.correlations = {
        k: v
        for k, v in description.correlations.items()
        if config.correlations.get(k, Correlation(calculate=False).calculate)
    }

    samples = [config.samples.head, config.samples.tail, config.samples.random]
    samples = [s > 0 for s in samples]
    description.sample = description.sample if any(samples) else []
    description.duplicates = (
        description.duplicates if config.duplicates.head > 0 else None
    )
    description.scatter = description.scatter if config.interactions.continuous else {}

    return description


def _is_alert_present(alert: Alert, alert_list: list) -> bool:
    return any(
        a.column_name == alert.column_name and a.alert_type == alert.alert_type
        for a in alert_list
    )


def _create_placehoder_alerts(report_alerts: tuple) -> tuple:
    from copy import copy

    fixed: list = [[] for _ in report_alerts]
    for idx, alerts in enumerate(report_alerts):
        for alert in alerts:
            fixed[idx].append(alert)
            for i, fix in enumerate(fixed):
                if i == idx:
                    continue
                if not _is_alert_present(alert, report_alerts[i]):
                    empty_alert = copy(alert)
                    empty_alert._is_empty = True
                    fix.append(empty_alert)
    return tuple(fixed)


def compare(
    reports: Union[List[ProfileReport], List[BaseDescription]],
    config: Optional[Settings] = None,
    compute: bool = False,
) -> ProfileReport:
    """
    Compare Profile reports

    Args:
        reports: two reports to compare
                 input may either be a ProfileReport, or the summary obtained from report.get_description()
        config: the settings object for the merged ProfileReport
        compute: recompute the profile report using config or the left report config
                 recommended in cases where the reports were created using different settings

    """
    if len(reports) == 0:
        raise ValueError("No reports available for comparison.")

    report_dtypes = [type(r) for r in reports]
    if len(set(report_dtypes)) > 1:
        raise TypeError(
            "The input must have the same data type for all reports. Comparing ProfileReport objects to summaries obtained from the get_description() method is not supported."
        )

    if isinstance(reports[0], ProfileReport):
        all_configs = [r.config for r in reports]  # type: ignore
    else:
        configs_str = [
            json.loads(r.package["data_profiling_config"]) for r in reports  # type: ignore
        ]
        all_configs = []
        for c_str in configs_str:
            c_setting = Settings()
            c_setting = c_setting.update(c_str)
            all_configs.append(c_setting)

    validate_reports(reports=reports, configs=all_configs)

    if isinstance(reports[0], ProfileReport):
        base_features = reports[0].df.columns  # type: ignore
        for report in reports[1:]:
            cols_2_compare = [col for col in base_features if col in report.df.columns]  # type: ignore
            report.df = report.df.loc[:, cols_2_compare]  # type: ignore
        reports = [r for r in reports if not r.df.empty]  # type: ignore
        if len(reports) == 1:
            return reports[0]  # type: ignore
    else:
        base_features = list(reports[0].variables.keys())
        non_empty_reports = 0
        for report in reports[1:]:
            cols_2_compare = [
                col for col in base_features if col in list(report.variables.keys())  # type: ignore
            ]
            if len(cols_2_compare) > 0:
                non_empty_reports += 1
        if non_empty_reports == 0:
            profile = ProfileReport(None, config=all_configs[0])
            profile._description_set = reports[0]
            return profile

    _config = None
    if config is None:
        _config = all_configs[0].copy()
    else:
        _config = config.copy()
        if isinstance(reports[0], ProfileReport):
            for report in reports:
                tsmode = report.config.vars.timeseries.active  # type: ignore
                title = report.config.title  # type: ignore
                report.config = config.copy()  # type: ignore
                report.config.title = title  # type: ignore
                report.config.vars.timeseries.active = tsmode  # type: ignore
                if compute:
                    report._description_set = None  # type: ignore

    if all(isinstance(report, ProfileReport) for report in reports):
        # Type ignore is needed as mypy does not pick up on the type narrowing
        # Consider using TypeGuard (3.10): https://docs.python.org/3/library/typing.html#typing.TypeGuard
        _update_titles(reports)  # type: ignore
        labels, descriptions = _compare_profile_report_preprocess(reports, _config)  # type: ignore
    elif all(isinstance(report, BaseDescription) for report in reports):
        labels, descriptions = _compare_dataset_description_preprocess(reports)  # type: ignore
    else:
        raise TypeError(
            "The input must have the same data type for all reports. Comparing ProfileReport objects to summaries obtained from the get_description() method is not supported."
        )

    _config.html.style._labels = labels
    source_dtypes = [
        source_dtypes_of(r.df) if isinstance(r, ProfileReport) else {}  # type: ignore
        for r in reports
    ]

    _placeholders(descriptions)

    descriptions_dict = [asdict(_apply_config(d, _config)) for d in descriptions]

    res: dict = _update_merge(None, descriptions_dict[0])
    for r in descriptions_dict[1:]:
        res = _update_merge(res, r)

    res["analysis"]["title"] = _compare_title(res["analysis"]["title"])
    res["alerts"] = _create_placehoder_alerts(res["alerts"])
    if not any(res["time_index_analysis"]):
        res["time_index_analysis"] = None
    profile = ProfileReport(None, config=_config)
    profile._description_set = from_dict(data_class=BaseDescription, data=res)
    # Kept for the agent output, which describes each report and the deltas
    profile._comparison_parts = list(zip(labels, descriptions, source_dtypes))
    return profile
