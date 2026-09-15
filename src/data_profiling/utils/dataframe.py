"""Utils for pandas DataFrames."""
import hashlib
import re
import unicodedata
import warnings
from pathlib import Path
from typing import Any, Optional

import pandas as pd
from pandas.core.util.hashing import hash_pandas_object


def warn_read(extension: str) -> None:
    """Warn the user when an extension is not supported.

    Args:
        extension: The extension that is not supported.
    """
    warnings.warn(
        f"""There was an attempt to read a file with extension {extension}, we assume it to be in CSV format.
To prevent this warning from showing up, please rename the file to any of the extensions supported by pandas
(docs: https://pandas.pydata.org/pandas-docs/stable/user_guide/io.html)
If you think this extension should be supported, please report this as an issue:
https://github.com/Data-Centric-AI-Community/data-profiling/issues"""
    )


def is_supported_compression(file_extension: str) -> bool:
    """Determine if the given file extension indicates a compression format that pandas can handle automatically.

    Args:
        file_extension (str): the file extension to test

    Returns:
        bool: True if the extension indicates a compression format that pandas handles automatically and False otherwise

    Notes:
        Pandas can handle on the fly decompression from the following extensions: ‘.bz2’, ‘.gz’, ‘.zip’, or ‘.xz’
        (otherwise no decompression). If using ‘.zip’, the ZIP file must contain exactly one data file to be read in.
    """
    return file_extension.lower() in [".bz2", ".gz", ".xz", ".zip"]


def remove_suffix(text: str, suffix: str) -> str:
    """Removes the given suffix from the given string.

    Args:
        text (str): the string to remove the suffix from
        suffix (str): the suffix to remove from the string

    Returns:
        str: the string with the suffix removed, if the string ends with the suffix, otherwise the unmodified string

    Notes:
        In python 3.9+, there is a built-in string method called removesuffix() that can serve this purpose.
    """
    return text[: -len(suffix)] if suffix and text.endswith(suffix) else text


def uncompressed_extension(file_name: Path) -> str:
    """Returns the uncompressed extension of the given file name.

    Args:
        file_name (Path): the file name to get the uncompressed extension of

    Returns:
        str: the uncompressed extension, or the original extension if pandas doesn't handle it automatically
    """
    extension = file_name.suffix.lower()
    return (
        Path(remove_suffix(str(file_name).lower(), extension)).suffix
        if is_supported_compression(extension)
        else extension
    )


def read_pandas(file_name: Path, trusted_source: bool = False) -> pd.DataFrame:
    """Read DataFrame based on the file extension. This function is used when the file is in a standard format.
    Various file types are supported (.csv, .json, .jsonl, .data, .tsv, .xls, .xlsx, .xpt, .sas7bdat, .parquet)

    Args:
        file_name: the file to read
        trusted_source: Whether the file comes from a trusted source.

    Returns:
        DataFrame

    Notes:
        This function is based on pandas IO tools:
        https://pandas.pydata.org/pandas-docs/stable/user_guide/io.html
        https://pandas.pydata.org/pandas-docs/stable/reference/io.html

        This function is not intended to be flexible or complete. The main use case is to be able to read files without
        user input, which is currently used in the editor integration. For more advanced use cases, the user should load
        the DataFrame in code.
    """
    extension = uncompressed_extension(file_name)
    if extension == ".json":
        df = pd.read_json(str(file_name))
    elif extension == ".jsonl":
        df = pd.read_json(str(file_name), lines=True)
    elif extension == ".dta":
        df = pd.read_stata(str(file_name))
    elif extension == ".tsv":
        df = pd.read_csv(str(file_name), sep="\t")
    elif extension in [".xls", ".xlsx"]:
        df = pd.read_excel(str(file_name))
    elif extension in [".hdf", ".h5"]:
        df = pd.read_hdf(str(file_name))
    elif extension in [".sas7bdat", ".xpt"]:
        df = pd.read_sas(str(file_name))
    elif extension == ".parquet":
        df = pd.read_parquet(str(file_name))
    elif extension in [".pkl", ".pickle"]:
        if not trusted_source:
            warnings.warn(
                "Loading pickle files from untrusted sources can lead to remote code execution. "
                "Only load pickle files from trusted sources or set trusted_source=True if you accept the risk.",
                RuntimeWarning,
                stacklevel=2,
            )
        df = pd.read_pickle(str(file_name))
    elif extension == ".tar":
        raise ValueError(
            "tar compression is not supported directly by pandas, please use the 'tarfile' module"
        )
    else:
        if extension != ".csv":
            warn_read(extension)

        df = pd.read_csv(str(file_name))
    return df


def rename_index(df: pd.DataFrame) -> pd.DataFrame:
    """If the DataFrame contains a column or index named `index`, this will produce errors. We rename the {index,column}
    to be `df_index`.

    Args:
        df: DataFrame to process.

    Returns:
        The DataFrame with {index,column} `index` replaced by `df_index`, unchanged if the DataFrame does not contain such a string.
    """
    df.rename(columns={"index": "df_index"}, inplace=True)

    if "index" in df.index.names:
        df.index.names = [x if x != "index" else "df_index" for x in df.index.names]
    return df


def expand_mixed(df: pd.DataFrame, types: Any = None) -> pd.DataFrame:
    """Expand non-nested lists, dicts and tuples in a DataFrame into columns with a prefix.

    Args:
        types: list of types to expand (Default: list, dict, tuple)
        df: DataFrame

    Returns:
        DataFrame with the dict values as series, identifier by the combination of the series and dict keys.

    Notes:
        TODO: allow for list expansion by duplicating rows.
        TODO: allow for expansion of nested data structures.
    """
    if types is None:
        types = [list, dict, tuple]

    for column_name in df.columns:
        # All
        non_nested_enumeration = (
            df[column_name]
            .dropna()
            .map(lambda x: type(x) in types and not any(type(y) in types for y in x))
        )

        if non_nested_enumeration.all():
            # Expand and prefix
            expanded = pd.DataFrame(df[column_name].dropna().tolist())
            expanded = expanded.add_prefix(column_name + "_")

            # Add recursion
            expanded = expand_mixed(expanded)

            # Drop the expanded
            df.drop(columns=[column_name], inplace=True)

            df = pd.concat([df, expanded], axis=1)
    return df


# Change this if `hash_dataframe`'s implementation changes.
HASH_PREFIX = "2@"


def hash_dataframe(df: pd.DataFrame) -> str:
    """Hash a DataFrame (implementation might change in the future)

    Args:
        df: the DataFrame

    Returns:
        The DataFrame's hash
    """
    # hash_pandas_object returns a series of uint64s. Using their
    # binary representation would be more efficient, but it's not
    # necessarily portable across architectures. Using the human-readable
    # string values should be good enough.
    hash_values = "\n".join(hash_pandas_object(df).values.astype(str))
    digest = hashlib.sha256(hash_values.encode("utf-8")).hexdigest()
    return f"{HASH_PREFIX}{digest}"


def slugify(value: str, allow_unicode: bool = False) -> str:
    """
    Taken from https://github.com/django/django/blob/master/django/utils/text.py
    Convert to ASCII if 'allow_unicode' is False. Convert spaces or repeated
    dashes to single dashes. Remove characters that aren't alphanumerics,
    underscores, or hyphens. Convert to lowercase. Also strip leading and
    trailing whitespace, dashes, and underscores.
    """
    value = str(value)
    if allow_unicode:
        value = unicodedata.normalize("NFKC", value)
    else:
        value = (
            unicodedata.normalize("NFKD", value)
            .encode("ascii", "ignore")
            .decode("ascii")
        )
    value = re.sub(r"[^\w\s-]", "", value.lower())
    return re.sub(r"[-\s]+", "-", value).strip("-_")


def sort_column_names(dct: dict, sort: Optional[str]) -> dict:
    if sort is None:
        return dct

    sort = sort.lower()
    if sort.startswith("asc"):
        dct = dict(sorted(dct.items(), key=lambda x: x[0].casefold()))
    elif sort.startswith("desc"):
        dct = dict(sorted(dct.items(), key=lambda x: x[0].casefold(), reverse=True))
    else:
        raise ValueError('"sort" should be "ascending", "descending" or None.')
    return dct
