"""
    Auxiliary handler methods for data summary extraction
"""
from typing import Any, Callable, Dict, List, Sequence

from data_profiling.model.typeset_base import BaseTypeSet


def compose(functions: Sequence[Callable]) -> Callable:
    """
    Compose a sequence of functions.

    :param functions: sequence of functions
    :return: combined function applying all functions in order.
    """

    def composed_function(*args) -> List[Any]:
        result = args  # Start with the input arguments
        for func in functions:
            result = func(*result) if isinstance(result, tuple) else func(result)
        return result  # type: ignore

    return composed_function  # type: ignore


class Handler:
    """A generic handler

    Allows any custom mapping between data types and functions
    """

    def __init__(
        self, mapping: Dict[str, List[Callable]], typeset: BaseTypeSet, *args, **kwargs
    ):
        self.mapping = mapping
        self.typeset = typeset
        self._complete_dag()

    def _complete_dag(self) -> None:
        # A type runs its parent's functions first (e.g. Numeric gets the
        # Unsupported counts before its own statistics). Parents come first in
        # the topological order, so their lists are already complete.
        for type_ in self.typeset.topological_order():
            parent = self.typeset.parent_of(type_)
            if parent is None:
                continue
            self.mapping[str(type_)] = self.mapping.get(
                str(parent), []
            ) + self.mapping.get(str(type_), [])

    def handle(self, dtype: str, *args, **kwargs) -> dict:
        """
        Returns:
            object: a tuple containing the config, the dataset series and the summary extracted
        """
        funcs = self.mapping.get(dtype, [])
        op = compose(funcs)
        summary = op(*args)[-1]
        return summary


def get_render_map() -> Dict[str, Callable]:
    import data_profiling.report.structure.variables as render_algorithms

    render_map = {
        "Boolean": render_algorithms.render_boolean,
        "Numeric": render_algorithms.render_real,
        "Complex": render_algorithms.render_complex,
        "Text": render_algorithms.render_text,
        "DateTime": render_algorithms.render_date,
        "Categorical": render_algorithms.render_categorical,
        "URL": render_algorithms.render_url,
        "Path": render_algorithms.render_path,
        "File": render_algorithms.render_file,
        "Image": render_algorithms.render_image,
        "Unsupported": render_algorithms.render_generic,
        "TimeSeries": render_algorithms.render_timeseries,
    }

    return render_map
