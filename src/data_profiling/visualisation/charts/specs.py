"""Chart specifications.

A spec is a small, renderer-agnostic description of a chart: the data that is
drawn (already binned / aggregated / downsampled) plus presentation hints. The
model layer builds specs; a renderer (``svg.render``) turns them into markup.
Specs only hold plain Python values, so ``to_dict()`` is JSON-serialisable.
"""

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class ChartSpec:
    kind: str = field(init=False, default="chart")
    title: str = ""
    width: int = 640
    height: int = 280

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# --- Histograms ---------------------------------------------------------------


@dataclass
class HistSeries:
    counts: List[float]
    edges: List[float]
    color: str
    label: str = ""


@dataclass
class HistogramSpec(ChartSpec):
    series: List[HistSeries] = field(default_factory=list)
    date: bool = False
    mini: bool = False
    x_labels: bool = True
    y_label: str = "Frequency"

    def __post_init__(self) -> None:
        self.kind = "histogram"


# --- Bars ---------------------------------------------------------------------


@dataclass
class BarSeries:
    values: List[float]
    color: str
    label: str = ""
    annotations: Optional[List[str]] = None
    # Extra tooltip lines per category ("Key: value" strings)
    details: Optional[List[List[str]]] = None


@dataclass
class BarSpec(ChartSpec):
    """Bars over categorical labels (optionally grouped for comparisons)."""

    categories: List[str] = field(default_factory=list)
    series: List[BarSeries] = field(default_factory=list)
    horizontal: bool = True
    value_format: str = "count"  # "count" | "percent"
    value_max: Optional[float] = None
    value_label: str = ""
    value_name: str = "Count"  # tooltip label of the bar value

    def __post_init__(self) -> None:
        self.kind = "bar"


@dataclass
class StackedRow:
    label: str
    values: List[float]


@dataclass
class StackedBarSpec(ChartSpec):
    """100% stacked horizontal bar: one row per report (comparison)."""

    categories: List[str] = field(default_factory=list)
    colors: List[str] = field(default_factory=list)
    rows: List[StackedRow] = field(default_factory=list)
    legend: bool = True
    redact: bool = False  # sensitive mode: never emit category names

    def __post_init__(self) -> None:
        self.kind = "stacked_bar"


@dataclass
class PieSpec(ChartSpec):
    categories: List[str] = field(default_factory=list)
    values: List[float] = field(default_factory=list)
    colors: List[str] = field(default_factory=list)
    legend: bool = True
    redact: bool = False  # sensitive mode: never emit category names

    def __post_init__(self) -> None:
        self.kind = "pie"


# --- Matrices -----------------------------------------------------------------


@dataclass
class HeatmapSpec(ChartSpec):
    row_labels: List[str] = field(default_factory=list)
    col_labels: List[str] = field(default_factory=list)
    values: List[List[Optional[float]]] = field(default_factory=list)
    vmin: float = -1.0
    vmax: float = 1.0
    stops: List[str] = field(default_factory=list)
    bad_color: str = "#000000"
    bad_label: str = "invalid coefficient"
    mask: Optional[List[List[bool]]] = None  # True = cell not drawn
    annotate: bool = False
    annotation_style: str = "coefficient"  # "coefficient" | "number"
    colorbar: bool = True
    x_axis_title: str = ""
    cell_max: float = 28.0
    value_name: str = "Correlation"  # tooltip label of the cell value
    interpret: str = "correlation"  # "correlation" | "nullity" | "none"

    def __post_init__(self) -> None:
        self.kind = "heatmap"


@dataclass
class MissingMatrixSpec(ChartSpec):
    """Row-bucketed nullity matrix.

    ``bands[c]`` is a list of ``(row_start, row_end, fraction_present)`` runs for
    column ``c``; rows are bucketed so the markup size does not grow with the
    number of rows.
    """

    columns: List[str] = field(default_factory=list)
    nrows: int = 0
    bands: List[List[Tuple[int, int, float]]] = field(default_factory=list)
    color: str = "#0d6efd"
    labels: bool = True

    def __post_init__(self) -> None:
        self.kind = "missing_matrix"


# --- Scatter / hexbin ---------------------------------------------------------


@dataclass
class ScatterSpec(ChartSpec):
    x: List[float] = field(default_factory=list)
    y: List[float] = field(default_factory=list)
    color: str = "#0d6efd"
    x_label: str = ""
    y_label: str = ""
    # Hexbin mode (used above ``plot.scatter_threshold`` points)
    hexbin: bool = False
    gridsize: int = 30
    n_points: int = 0

    def __post_init__(self) -> None:
        self.kind = "scatter"


# --- Lines --------------------------------------------------------------------


@dataclass
class LineSeries:
    x: List[float]
    y: List[Optional[float]]
    color: str
    label: str = ""
    dash: bool = False


@dataclass
class LineSpec(ChartSpec):
    series: List[LineSeries] = field(default_factory=list)
    x_is_date: bool = False
    spans: List[Tuple[float, float, str]] = field(default_factory=list)
    legend: bool = False
    mini: bool = False
    y_label: str = ""

    def __post_init__(self) -> None:
        self.kind = "line"


@dataclass
class StemPanel:
    title: str
    values: List[float]
    conf_low: List[float]
    conf_high: List[float]
    color: str


@dataclass
class StemGridSpec(ChartSpec):
    """Grid of stem plots, e.g. ACF | PACF, one row per compared report."""

    rows: List[List[StemPanel]] = field(default_factory=list)
    row_labels: List[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.kind = "stem_grid"
