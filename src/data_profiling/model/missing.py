import importlib
import warnings
from itertools import combinations
from typing import Any, Callable, Dict, Optional, Sized

import numpy as np
import pandas as pd

from data_profiling.config import Settings


def get_missing_associations(config: Settings, df: Sized, summary: dict) -> dict:
    """Find exploratory missingness associations, retaining aggregate data only.

    This opt-in analysis runs once in the missing-values stage. Limits and
    thresholds are deliberately internal while the experimental design settles.
    A lack of findings is not evidence that values are missing at random.
    """
    result: dict = {"status": "complete", "findings": []}
    if not config.missing_diagrams.get("associations", False):
        return {"status": "disabled", "findings": []}
    if not isinstance(df, pd.DataFrame):
        warnings.warn("Missingness associations are not supported on Spark; skipping.")
        return {"status": "unsupported", "findings": []}

    n = len(df)
    targets = sorted(
        col for col, stats in summary.items() if 0 < stats["n_missing"] < n
    )
    if not targets:
        return result
    predictors = sorted(
        col
        for col, stats in summary.items()
        if stats["type"] in {"Categorical", "Boolean"} and 1 < stats["n_distinct"] <= 20
    )
    hypotheses = len(targets) * (len(targets) - 1) // 2 + sum(
        summary[col]["n_distinct"] * (len(targets) - (col in targets))
        for col in predictors
    )
    # Include space for factorized codes, selections and contingency workspace.
    if (
        len(set(targets) | set(predictors)) > 100
        or any(len(col) > 200 for col in set(targets) | set(predictors))
        or hypotheses > 5000
        or n * (len(targets) + 64) > 64 * 1024 * 1024
    ):
        warnings.warn("Missingness association work/memory limit exceeded; skipping.")
        return {"status": "skipped_limits", "findings": []}

    from scipy.stats import chi2_contingency, fisher_exact
    from statsmodels.stats.multitest import multipletests

    from data_profiling.model.pandas.correlations_pandas import _cramers_corrected_stat
    from data_profiling.model.typeset_relations import string_to_bool

    minimum = max(20, int(np.ceil(n * 0.01)))
    masks = {col: df[col].isna().to_numpy() for col in targets}
    candidates = []
    pvalues = []

    def evaluate(
        target: str,
        predictor: str,
        value: str,
        kind: str,
        a: int,
        b: int,
        c: int,
        d: int,
    ) -> None:
        # Columns of the 2x2 table are missing/observed, rows group/complement.
        support, complement = a + b, c + d
        if min(support, complement) < minimum or (a + c) * (b + d) == 0:
            return
        table = [[a, b], [c, d]]
        # Exact tests become costly for large, well-populated tables. Use the
        # same continuity-corrected chi-square convention as Cramer's V there,
        # retaining Fisher's exact test when any expected cell is sparse.
        minimum_expected = (
            min(support, complement) * min(a + c, b + d) / (support + complement)
        )
        if minimum_expected >= 5:
            pvalue = float(chi2_contingency(table, correction=True)[1])
        else:
            pvalue = float(fisher_exact(table)[1])
        pvalues.append(pvalue)
        if kind == "missingness":
            strength = (a * d - b * c) / np.sqrt(
                float(support * complement * (a + c) * (b + d))
            )
        else:
            strength = float(_cramers_corrected_stat(pd.DataFrame(table), True))
        uplift = a / support - c / complement
        if strength < 0.3 or uplift < 0.20 or a < minimum:
            return
        candidates.append(
            (
                len(pvalues) - 1,
                {
                    "kind": kind,
                    "target": target,
                    "predictor": predictor,
                    "value": value[:200],
                    "strength": strength,
                    "support": int(support),
                    "group_missing_rate": a / support,
                    "complement_missing_rate": c / complement,
                },
            )
        )

    for target, predictor in combinations(targets, 2):
        left, right = masks[target], masks[predictor]
        a = int(np.count_nonzero(left & right))
        b = int(summary[predictor]["n_missing"]) - a
        c = int(summary[target]["n_missing"]) - a
        d = n - a - b - c
        # Select the stronger conditional explanation without testing the pair
        # twice or making detection depend on alphabetical column ordering.
        if a / (a + b) - c / (c + d) < a / (a + c) - b / (b + d):
            target, predictor = predictor, target
            b, c = c, b
        evaluate(target, predictor, "missing", "missingness", a, b, c, d)

    for predictor in predictors:
        series = df[predictor]
        # Normalize boolean spellings with the existing type-system transform.
        # Custom schemas may also label native booleans; leave those untouched.
        if summary[predictor]["type"] == "Boolean":
            values = series.dropna().unique()
            if all(isinstance(value, str) for value in values):
                if any(
                    value.lower() not in config.vars.bool.mappings for value in values
                ):
                    warnings.warn(
                        "Boolean predictor contains unmapped values; skipping "
                        "its categorical missingness associations."
                    )
                    continue
                series = string_to_bool(series, {}, config.vars.bool.mappings)
            elif any(isinstance(value, str) for value in values):
                normalized = {
                    value: config.vars.bool.mappings.get(value.lower())
                    if isinstance(value, str)
                    else bool(value)
                    if value in (True, False)
                    else None
                    for value in values
                }
                if any(value is None for value in normalized.values()):
                    warnings.warn(
                        "Boolean predictor contains unmapped values; skipping "
                        "its categorical missingness associations."
                    )
                    continue
                series = series.map(normalized)
        codes, labels = pd.factorize(series, sort=False)
        valid = codes >= 0
        counts = np.bincount(codes[valid], minlength=len(labels))
        for target in targets:
            if target == predictor:
                continue
            missing_counts = np.bincount(
                codes[valid], weights=masks[target][valid], minlength=len(labels)
            ).astype(int)
            total_missing = int(missing_counts.sum())
            total = int(counts.sum())
            for index, label in enumerate(labels):
                a = int(missing_counts[index])
                b = int(counts[index]) - a
                c = total_missing - a
                redacted = config.vars.cat.redact or config.vars.text.redact
                value = (
                    f"[redacted category {index + 1}]"
                    if redacted
                    else f"{str(label)[:170]} [group {index + 1}]"
                )
                evaluate(
                    target, predictor, value, "categorical", a, b, c, total - a - b - c
                )

    if pvalues:
        rejected = multipletests(pvalues, alpha=0.05, method="fdr_bh")[0]
        findings = [finding for index, finding in candidates if rejected[index]]
        findings.sort(
            key=lambda item: (
                -item["strength"],
                -item["support"],
                item["target"],
                item["predictor"],
                item["value"],
            )
        )
        result["findings"] = findings[:50]
        if len(findings) > 50:
            warnings.warn(
                "Missingness associations truncated to the strongest 50 findings."
            )
    return result


class MissingDataBackend:
    """Helper class to select and cache the appropriate missing-data backend (Pandas or Spark)."""

    def __init__(self, df: Sized):
        """Determine backend once and store it for all missing-data computations."""
        if isinstance(df, pd.DataFrame):
            self.backend_module = "data_profiling.model.pandas.missing_pandas"
        else:
            self.backend_module = "data_profiling.model.spark.missing_spark"

        self.module = importlib.import_module(self.backend_module)

    def get_method(self, method_name: str) -> Callable:
        """Retrieve the appropriate missing-data function from the backend module."""
        try:
            return getattr(self.module, method_name)
        except AttributeError as ex:
            raise AttributeError(
                f"Missing-data function '{method_name}' is not available in {self.backend_module}."
            ) from ex


class MissingData:
    _method_name: str = ""

    def compute(
        self, config: Settings, df: Sized, backend: MissingDataBackend
    ) -> Optional[Sized]:
        """Computes correlation using the correct backend (Pandas or Spark)."""
        try:
            method = backend.get_method(self._method_name)
        except AttributeError as ex:
            raise NotImplementedError() from ex
        else:
            return method(config, df)


class MissingBar(MissingData):
    _method_name = "missing_bar"


class MissingMatrix(MissingData):
    _method_name = "missing_matrix"


class MissingHeatmap(MissingData):
    _method_name = "missing_heatmap"


def get_missing_active(config: Settings, table_stats: dict) -> Dict[str, Any]:
    """

    Args:
        config: report Settings object
        table_stats: The overall statistics for the DataFrame.

    Returns:

    """

    missing_map = {
        "bar": {
            "min_missing": 0,
            "name": "Count",
            "caption": "A simple visualization of nullity by column.",
            "function": MissingBar(),
        },
        "matrix": {
            "min_missing": 0,
            "name": "Matrix",
            "caption": "Nullity matrix is a data-dense display which lets you quickly visually pick out patterns in data completion.",
            "function": MissingMatrix(),
        },
        "heatmap": {
            "min_missing": 2,
            "name": "Heatmap",
            "caption": "The correlation heatmap measures nullity correlation: how strongly the presence or absence of one variable affects the presence of another.",
            "function": MissingHeatmap(),
        },
    }

    missing_map = {
        name: settings
        for name, settings in missing_map.items()
        if (
            config.missing_diagrams[name]
            and table_stats["n_vars_with_missing"] >= settings["min_missing"]
        )
        and (
            name != "heatmap"
            or (
                table_stats["n_vars_with_missing"] - table_stats["n_vars_all_missing"]
                >= settings["min_missing"]
            )
        )
    }

    return missing_map


def get_missing_diagram(
    config: Settings, df: pd.DataFrame, settings: Dict[str, Any]
) -> Optional[Dict[str, Any]]:
    """Gets the rendered diagrams for missing values.

    Args:
        config: report Settings object
        df: The DataFrame on which to calculate the missing values.
        settings: missing diagram name, caption and function

    Returns:
        A dictionary containing the base64 encoded plots for each diagram that is active in the config (matrix, bar, heatmap).
    """
    backend = MissingDataBackend(df)

    missing_func = settings.get("function")
    if missing_func is None:
        return None  # No function defined, skip execution

    try:
        result = missing_func.compute(config, df, backend)
    except ValueError as e:
        warnings.warn(
            f"""There was an attempt to generate the {settings['name']} missing values diagrams, but this failed.
        To hide this warning, disable the calculation
        (using `df.profile_report(missing_diagrams={{"{settings['name']}": False}}`)
        If this is problematic for your use case, please report this as an issue:
        https://github.com/Data-Centric-AI-Community/data-profiling/issues
        (include the error message: '{e}')"""
        )
        return None
    else:
        return {
            "name": settings["name"],
            "caption": settings["caption"],
            "matrix": result,
        }
