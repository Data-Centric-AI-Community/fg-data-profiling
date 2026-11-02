"""This file add the console interface to the package."""

import argparse
from pathlib import Path
from typing import Any, List, Optional

from data_profiling.__init__ import ProfileReport, __version__
from data_profiling.utils.dataframe import read_pandas


def parse_args(args: Optional[List[Any]] = None) -> argparse.Namespace:
    """Parse the command line arguments for the `data_profiling` binary.

    Args:
      args: List of input arguments. (Default value=None).

    Returns:
      Namespace with parsed arguments.

    """
    parser = argparse.ArgumentParser(
        description="Profile the variables in a CSV file and generate a HTML report."
    )

    # Version
    parser.add_argument(
        "--version", action="version", version=f"%(prog)s {__version__}"
    )

    # Console specific
    parser.add_argument(
        "-s",
        "--silent",
        help="Only generate but do not open report",
        action="store_true",
    )

    parser.add_argument(
        "-m",
        "--minimal",
        help="Minimal configuration for big data sets",
        action="store_true",
    )

    parser.add_argument(
        "-e",
        "--explorative",
        help="Explorative configuration featuring unicode, file and image analysis",
        action="store_true",
    )

    # Config
    parser.add_argument(
        "--pool_size", type=int, default=0, help="Number of CPU cores to use"
    )
    parser.add_argument(
        "--title",
        type=str,
        default="YData Profiling Report",
        help="Title for the report",
    )

    parser.add_argument(
        "--infer_dtypes",
        default=False,
        action="store_true",
        help="To infer dtypes of the dataframe",
    )

    parser.add_argument(
        "--no-infer_dtypes",
        dest="infer_dtypes",
        action="store_false",
        help="To read dtypes as read by pandas",
    )

    parser.add_argument(
        "--legacy-json",
        dest="legacy_json",
        default=False,
        action="store_true",
        help="For .json output, write the deprecated pre-agent JSON dump "
        "(removed in the next major release).",
    )

    parser.add_argument(
        "--config_file",
        type=str,
        default=None,
        help="Specify a yaml config file. Have a look at the 'config_default.yaml' as a starting point.",
    )

    parser.add_argument(
        "input_file",
        type=str,
        help="CSV file (or other file type supported by pandas) to profile",
    )
    parser.add_argument(
        "output_file",
        type=str,
        nargs="?",
        help="Output report file (.html, .json for AI agents, .md). If empty, replaces the input_file's extension with .html and uses that.",
        default=None,
    )

    return parser.parse_args(args)


def main(args: Optional[List[Any]] = None) -> None:
    """Run the `data_profiling` package.

    Args:
      args: Arguments for the programme (Default value=None).
    """

    # Parse the arguments
    parsed_args = parse_args(args)
    kwargs = vars(parsed_args)

    input_file = Path(kwargs.pop("input_file"))
    output_file = kwargs.pop("output_file")
    if output_file is None:
        output_file = str(input_file.with_suffix(".html"))

    silent = kwargs.pop("silent")
    legacy_json = kwargs.pop("legacy_json")

    # read the DataFrame
    df = read_pandas(input_file)

    # Generate the profiling report
    p = ProfileReport(
        df,
        **kwargs,
    )
    p.to_file(
        Path(output_file),
        silent=silent,
        legacy=True if legacy_json and Path(output_file).suffix == ".json" else None,
    )
