"""Record the inferred type of a fixed corpus of columns, and compare records.

The corpus targets the columns where more than one type relation matches:
numeric and date strings across cardinalities, URL and path strings with few
distinct values, month names, 0/1 flags, autocorrelated numbers.

Uses only ``ProfilingTypeSet(config).infer_type``, so it runs on any version
of the package. Typical use, from the repository root:

    git stash; python tests/typeset_snapshot.py record before.json; git stash pop
    python tests/typeset_snapshot.py record after.json
    python tests/typeset_snapshot.py compare before.json after.json

``record`` also accepts ``--data file.csv [...]`` to add the columns of real
datasets (CSV or Parquet) to the corpus.
"""
import argparse
import datetime
import json
import sys
from typing import Callable, Dict, List

import numpy as np
import pandas as pd

N_ROWS = 200
CARDINALITIES = [1, 2, 3, 5, 6, 10, 20, 28, 50, 51, 100, 200]
MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun"]
WEEKDAYS = ["Monday", "Tuesday", "Wednesday"]


def _cycle(values: List, n: int = N_ROWS) -> pd.Series:
    return pd.Series([values[i % len(values)] for i in range(n)])


def corpus() -> Dict[str, pd.Series]:
    generators: Dict[str, Callable[[int], List]] = {
        "int": lambda k: list(range(k)),
        "float": lambda k: [i / 4 for i in range(k)],
        "int_str": lambda k: [str(i) for i in range(k)],
        "float_str": lambda k: [f"{i / 4:.2f}" for i in range(k)],
        "padded_int_str": lambda k: [f"{i:05d}" for i in range(k)],
        "iso_date_str": lambda k: [
            (datetime.date(2024, 1, 1) + datetime.timedelta(days=i)).isoformat()
            for i in range(k)
        ],
        "datetime_str": lambda k: [
            str(datetime.datetime(2024, 1, 1, 8) + datetime.timedelta(hours=i))
            for i in range(k)
        ],
        "year_str": lambda k: [str(2000 + i) for i in range(k)],
        "url_str": lambda k: [f"https://example.com/page/{i}" for i in range(k)],
        "path_str": lambda k: [f"/data/file_{i}.csv" for i in range(k)],
        "word_str": lambda k: [f"item_{chr(97 + i % 26)}{i // 26}" for i in range(k)],
        "code_str": lambda k: [f"A{i:03d}" for i in range(k)],
    }
    series = {
        f"{name}_k{k}": _cycle(gen(k))
        for name, gen in generators.items()
        for k in CARDINALITIES
    }
    series.update(
        {
            "months": _cycle(MONTHS),
            "weekdays": _cycle(WEEKDAYS),
            "am_pm": _cycle(["AM", "PM"]),
            "yes_no": _cycle(["yes", "no"]),
            "y_n_nan": _cycle(["Y", "N", None]),
            "true_false_str": _cycle(["True", "False"]),
            "zero_one_int": _cycle([0, 1]),
            "zero_one_str": _cycle(["0", "1"]),
            "zero_one_float_nan": _cycle([0.0, 1.0, np.nan]),
            "bools": _cycle([True, False]),
            "bools_nan": _cycle([True, False, None]),
            "trend": pd.Series(np.arange(N_ROWS, dtype=float)),
            "trend_steps": pd.Series(np.repeat(np.arange(4), N_ROWS // 4)),
            "sine": pd.Series(np.sin(np.arange(N_ROWS) / 10)),
            "noise": pd.Series(np.random.default_rng(0).normal(size=N_ROWS)),
            "mixed_types": _cycle([1, "a", 2.5]),
            "empty_strings": _cycle([""]),
            "all_null": pd.Series([None] * N_ROWS),
        }
    )
    return series


def configs() -> Dict[str, object]:
    from data_profiling.config import Settings

    default = Settings()
    semantic = Settings()
    semantic.vars.url.active = True
    semantic.vars.path.active = True
    timeseries = Settings()
    timeseries.vars.timeseries.active = True
    return {"default": default, "url_path": semantic, "timeseries": timeseries}


def record(out: str, data: List[str]) -> None:
    from data_profiling.model.typeset import ProfilingTypeSet

    columns = corpus()
    for path in data:
        frame = (
            pd.read_parquet(path) if path.endswith(".parquet") else pd.read_csv(path)
        )
        for name, column in frame.items():
            columns[f"{path}:{name}"] = column.reset_index(drop=True)

    result = {}
    for config_name, config in configs().items():
        typeset = ProfilingTypeSet(config)
        result[config_name] = {
            name: str(typeset.infer_type(column.copy()))
            for name, column in columns.items()
        }
    with open(out, "w") as f:
        json.dump(result, f, indent=1, sort_keys=True)
    print(f"{sum(map(len, result.values()))} decisions written to {out}")


def compare(before: str, after: str) -> int:
    with open(before) as f:
        old = json.load(f)
    with open(after) as f:
        new = json.load(f)
    changed = 0
    for config_name in sorted(set(old) | set(new)):
        a, b = old.get(config_name, {}), new.get(config_name, {})
        for name in sorted(set(a) | set(b)):
            if a.get(name) != b.get(name):
                changed += 1
                print(f"{config_name:12} {name:32} {a.get(name)} -> {b.get(name)}")
    total = sum(map(len, new.values()))
    print(f"{changed} of {total} decisions changed")
    return 1 if changed else 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    rec = sub.add_parser("record")
    rec.add_argument("out")
    rec.add_argument("--data", nargs="*", default=[])
    cmp_ = sub.add_parser("compare")
    cmp_.add_argument("before")
    cmp_.add_argument("after")
    args = parser.parse_args()
    if args.command == "record":
        record(args.out, args.data)
        return 0
    return compare(args.before, args.after)


if __name__ == "__main__":
    sys.exit(main())
