import contextlib
import string
from collections import Counter
from typing import List, Tuple

import numpy as np
import pandas as pd

from data_profiling.config import Settings
from data_profiling.model.pandas.imbalance_pandas import column_imbalance_score
from data_profiling.model.pandas.utils_pandas import weighted_median
from data_profiling.model.summary_algorithms import (
    chi_square,
    describe_categorical_1d,
    histogram_compute,
    series_handle_nulls,
    series_hashable,
)


def get_character_counts_vc(vc: pd.Series) -> pd.Series:
    series = pd.Series(vc.index, index=vc, dtype=object)
    characters = series[series != ""].apply(list)
    characters = characters.explode()

    counts = pd.Series(characters.index, index=characters).dropna()
    if len(counts) > 0:
        counts = counts.groupby(level=0, sort=False).sum()
        counts = counts.sort_values(ascending=False)
        # FIXME: correct in split, below should be zero: print(counts.loc[''])
        counts = counts[counts.index.str.len() > 0]
    return counts


def get_character_counts(series: pd.Series) -> Counter:
    """Function to return the character counts

    Args:
        series: the Series to process

    Returns:
        A dict with character counts
    """
    return Counter(series.str.cat())


def counter_to_series(counter: Counter) -> pd.Series:
    if not counter:
        return pd.Series([], dtype=object)

    counter_as_tuples = counter.most_common()
    items, counts = zip(*counter_as_tuples)
    return pd.Series(counts, index=items)


def unicode_summary_vc(vc: pd.Series) -> dict:
    try:
        from tangled_up_in_unicode import (  # type: ignore
            block,
            block_abbr,
            category,
            category_long,
            script,
        )
    except ImportError:
        from unicodedata import category as _category  # pylint: disable=import-error

        category = _category  # type: ignore
        char_handler = lambda char: "(unknown)"  # noqa: E731
        block = char_handler
        block_abbr = char_handler
        category_long = char_handler
        script = char_handler

    # Unicode Character Summaries (category and script name)
    character_counts = get_character_counts_vc(vc)

    character_counts_series = character_counts
    summary = {
        "n_characters_distinct": len(character_counts_series),
        "n_characters" : np.sum(character_counts_series.to_numpy()),
        "character_counts": character_counts_series,
    }

    char_to_block = {key: block(key) for key in character_counts.keys()}
    char_to_category_short = {key: category(key) for key in character_counts.keys()}
    char_to_script = {key: script(key) for key in character_counts.keys()}

    summary.update(
        {
            "category_alias_values": {
                key: category_long(value)
                for key, value in char_to_category_short.items()
            },
            "block_alias_values": {
                key: block_abbr(value) for key, value in char_to_block.items()
            },
        }
    )

    # Retrieve original distribution
    block_alias_counts: Counter = Counter()
    per_block_char_counts: dict = {
        k: Counter() for k in summary["block_alias_values"].values()
    }
    for char, n_char in character_counts.items():
        block_name = summary["block_alias_values"][char]
        block_alias_counts[block_name] += n_char
        per_block_char_counts[block_name][char] = n_char
    summary["block_alias_counts"] = counter_to_series(block_alias_counts)
    summary["n_block_alias"] = len(summary["block_alias_counts"])
    summary["block_alias_char_counts"] = {
        k: counter_to_series(v) for k, v in per_block_char_counts.items()
    }

    script_counts: Counter = Counter()
    per_script_char_counts: dict = {k: Counter() for k in char_to_script.values()}
    for char, n_char in character_counts.items():
        script_name = char_to_script[char]
        script_counts[script_name] += n_char
        per_script_char_counts[script_name][char] = n_char
    summary["script_counts"] = counter_to_series(script_counts)
    summary["n_scripts"] = len(summary["script_counts"])
    summary["script_char_counts"] = {
        k: counter_to_series(v) for k, v in per_script_char_counts.items()
    }

    category_alias_counts: Counter = Counter()
    per_category_alias_char_counts: dict = {
        k: Counter() for k in summary["category_alias_values"].values()
    }
    for char, n_char in character_counts.items():
        category_alias_name = summary["category_alias_values"][char]
        category_alias_counts[category_alias_name] += n_char
        per_category_alias_char_counts[category_alias_name][char] += n_char
    summary["category_alias_counts"] = counter_to_series(category_alias_counts)
    if len(summary["category_alias_counts"]) > 0:
        summary["category_alias_counts"].index = summary[
            "category_alias_counts"
        ].index.str.replace("_", " ")
    summary["n_category"] = len(summary["category_alias_counts"])
    summary["category_alias_char_counts"] = {
        k: counter_to_series(v) for k, v in per_category_alias_char_counts.items()
    }

    with contextlib.suppress(AttributeError):
        summary["category_alias_counts"].index = summary[
            "category_alias_counts"
        ].index.str.replace("_", " ")

    return summary


def word_summary_vc(vc: pd.Series, stop_words: List[str] = []) -> dict:
    """Count the number of occurrences of each individual word across
    all lines of the data Series, then sort from the word with the most
    occurrences to the word with the least occurrences. If a list of
    stop words is given, they will be ignored.

    Args:
        vc: Series containing all unique categories as index and their
            frequency as value. Sorted from the most frequent down.
        stop_words: List of stop words to ignore, empty by default.

    Returns:
        A dict containing the results as a Series with unique words as
        index and the computed frequency as value
    """
    # TODO: configurable lowercase/punctuation etc.
    # TODO: remove punctuation in words

    series = pd.Series(vc.index, index=vc, dtype=object)
    word_lists = series.str.lower().str.split()
    words = word_lists.explode().str.strip(string.punctuation + string.whitespace)
    word_counts = pd.Series(words.index, index=words)
    # fix for pandas 1.0.5
    word_counts = word_counts[word_counts.index.notnull()]
    word_counts = word_counts.groupby(level=0, sort=False).sum()
    word_counts = word_counts.sort_values(ascending=False)

    # Remove stop words
    if len(stop_words) > 0:
        stop_words = [x.lower() for x in stop_words]
        word_counts = word_counts.loc[~word_counts.index.isin(stop_words)]

    return {"word_counts": word_counts} if not word_counts.empty else {}


def length_summary_vc(vc: pd.Series) -> dict:
    series = pd.Series(vc.index, index=vc, dtype=object)
    length = series.str.len()
    length_counts = pd.Series(length.index, index=length)
    length_counts = length_counts.groupby(level=0, sort=False).sum()
    length_counts = length_counts.sort_values(ascending=False)

    summary = {
        "max_length": np.max(length_counts.index),
        "mean_length": np.average(length_counts.index, weights=length_counts.values)
        if not length_counts.empty
        else np.nan,
        "median_length": weighted_median(
            length_counts.index.values, weights=length_counts.values
        )
        if not length_counts.empty
        else np.nan,
        "min_length": np.min(length_counts.index),
        "length_histogram": length_counts,
    }

    return summary


_displayed_catvar_banner = False


@describe_categorical_1d.register
@series_hashable
@series_handle_nulls
def pandas_describe_categorical_1d(
    config: Settings, series: pd.Series, summary: dict
) -> Tuple[Settings, pd.Series, dict]:
    """Describe a categorical series.

    Args:
        config: report Settings object
        series: The Series to describe.
        summary: The dict containing the series description so far.

    Returns:
        A dict containing calculated series description values.
    """
    # Global info banner
    global _displayed_catvar_banner

    # Make sure we deal with strings (Issue #100)
    series = series.astype(str)

    # Only run if at least 1 non-missing value
    value_counts = summary["value_counts_without_nan"]
    value_counts.index = value_counts.index.astype(str)

    summary["imbalance"] = column_imbalance_score(value_counts, len(value_counts))

    redact = config.vars.cat.redact
    if not redact:
        summary.update({"first_rows": series.head(5)})

    chi_squared_threshold = config.vars.num.chi_squared_threshold
    if chi_squared_threshold > 0.0:
    	summary["chi_squared"] = chi_square(
    	histogram=value_counts.to_numpy()
	)
    if config.vars.cat.length:
        summary.update(length_summary_vc(value_counts))
        summary.update(
            histogram_compute(
                config,
                summary["length_histogram"].index.values,
                len(summary["length_histogram"]),
                name="histogram_length",
                weights=summary["length_histogram"].values,
            )
        )

    if config.vars.cat.characters:
        summary.update(unicode_summary_vc(value_counts))

    if config.vars.cat.words:
        summary.update(word_summary_vc(value_counts, config.vars.cat.stop_words))

    if config.vars.cat.dirty_categories:  # noqa: SIM102
        if not _displayed_catvar_banner:
            _displayed_catvar_banner = True

    return config, series, summary
