"""Configuration for the package."""
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import yaml
from pydantic.v1 import BaseModel, BaseSettings, Field, PrivateAttr


def _merge_dictionaries(dict1: dict, dict2: dict) -> dict:
    """
    Recursive merge dictionaries.

    :param dict1: Base dictionary to merge.
    :param dict2: Dictionary to merge on top of base dictionary.
    :return: Merged dictionary
    """
    for key, val in dict1.items():
        if isinstance(val, dict):
            dict2_node = dict2.setdefault(key, {})
            _merge_dictionaries(val, dict2_node)
        else:
            if key not in dict2:
                dict2[key] = val

    return dict2


class Dataset(BaseModel):
    """Metadata of the dataset"""

    description: str = ""
    creator: str = ""
    author: str = ""
    copyright_holder: str = ""
    copyright_year: str = ""
    url: str = ""


class NumVars(BaseModel):
    quantiles: List[float] = [0.05, 0.25, 0.5, 0.75, 0.95]
    skewness_threshold: int = 20
    low_categorical_threshold: int = 5
    # Set to zero to disable
    chi_squared_threshold: float = 0.999


class TextVars(BaseModel):
    length: bool = True
    words: bool = True
    characters: bool = True
    redact: bool = False
    # Spark backend only: number of most frequent words collected to the driver
    spark_words_limit: int = 1000


class CatVars(BaseModel):
    length: bool = True
    characters: bool = True
    words: bool = True
    # if var has more than threshold categories, it's a text var
    cardinality_threshold: int = 50
    # if var has more than threshold % distinct values, it's a text var
    percentage_cat_threshold: float = 0.5
    imbalance_threshold: float = 0.5
    n_obs: int = 5
    # Set to zero to disable
    chi_squared_threshold: float = 0.999
    coerce_str_to_date: bool = False
    redact: bool = False
    histogram_largest: int = 50
    stop_words: List[str] = []
    dirty_categories: bool = False
    dirty_categories_threshold: float = 0.85


class BoolVars(BaseModel):
    n_obs: int = 3
    imbalance_threshold: float = 0.5

    # string to boolean mapping dict
    mappings: Dict[str, bool] = {
        "t": True,
        "f": False,
        "yes": True,
        "no": False,
        "y": True,
        "n": False,
        "true": True,
        "false": False,
    }


class FileVars(BaseModel):
    active: bool = False


class PathVars(BaseModel):
    active: bool = False


class ImageVars(BaseModel):
    active: bool = False
    exif: bool = True
    hash: bool = True


class UrlVars(BaseModel):
    active: bool = False


class TimeseriesVars(BaseModel):
    active: bool = False
    sortby: Optional[str] = None
    autocorrelation: float = 0.7
    lags: List[int] = [1, 7, 12, 24, 30]
    significance: float = 0.05
    pacf_acf_lag: int = 100
    autolag: Optional[str] = "AIC"
    maxlag: Optional[int] = None


class Univariate(BaseModel):
    num: NumVars = NumVars()
    text: TextVars = TextVars()
    cat: CatVars = CatVars()
    image: ImageVars = ImageVars()
    bool: BoolVars = BoolVars()
    path: PathVars = PathVars()
    file: FileVars = FileVars()
    url: UrlVars = UrlVars()
    timeseries: TimeseriesVars = TimeseriesVars()


class MissingPlot(BaseModel):
    # Force labels when there are > 50 variables
    force_labels: bool = True
    cmap: str = "RdBu"


class ImageType(Enum):
    svg = "svg"
    png = "png"


class CorrelationPlot(BaseModel):
    cmap: str = "RdBu"
    bad: str = "#000000"


class Histogram(BaseModel):
    # Number of bins (set to 0 to automatically detect the bin size)
    bins: int = 50
    # Maximum number of bins (when bins=0)
    max_bins: int = 250
    x_axis_labels: bool = True
    density: bool = False


class CatFrequencyPlot(BaseModel):
    show: bool = True  # if false, the category frequency plot is turned off
    type: str = "bar"  # options: 'bar', 'pie'

    # The cat frequency plot is only rendered if the number of distinct values is
    # smaller or equal to "max_unique"
    max_unique: int = 10

    # Colors should be a list of matplotlib recognised strings:
    # --> https://matplotlib.org/stable/tutorials/colors/colors.html
    # --> matplotlib defaults are used by default
    colors: Optional[List[str]] = None


class Plot(BaseModel):
    missing: MissingPlot = MissingPlot()
    image_format: ImageType = ImageType.svg
    correlation: CorrelationPlot = CorrelationPlot()
    dpi: int = 800  # PNG dpi
    histogram: Histogram = Histogram()
    scatter_threshold: int = 1000
    cat_freq: CatFrequencyPlot = CatFrequencyPlot()
    font_path: Optional[Union[Path, str]] = None


class Theme(Enum):
    united = "united"
    flatly = "flatly"
    cosmo = "cosmo"
    simplex = "simplex"


class Style(BaseModel):
    # Primary color used for plotting and text where applicable.
    @property
    def primary_color(self) -> str:
        # This attribute may be deprecated in the future, please use primary_colors[0]
        return self.primary_colors[0]

    # Primary color used for comparisons (default: blue, red, green)
    primary_colors: List[str] = ["#0d6efd", "#dc3545", "#198754"]

    # Base64-encoded logo image
    logo: str = ""

    # HTML Theme (optional, default: None)
    theme: Optional[Theme] = None

    # Labels used for comparing reports (private attribute)
    _labels: List[str] = PrivateAttr(["_"])


class Html(BaseModel):
    # Styling options for the HTML report
    style: Style = Style()

    # Show navbar
    navbar_show: bool = True

    # Minify the html
    minify_html: bool = True

    # Offline support
    use_local_assets: bool = True

    # If True, single file, else directory with assets
    inline: bool = True

    # Assets prefix if inline = True
    assets_prefix: Optional[str] = None

    # Internal usage
    assets_path: Optional[str] = None

    full_width: bool = False


class Duplicates(BaseModel):
    head: int = 10
    key: str = "# duplicates"


class Correlation(BaseModel):
    key: str = ""
    calculate: bool = Field(default=True)
    warn_high_correlations: int = Field(default=10)
    threshold: float = Field(default=0.5)
    n_bins: int = Field(default=10)


class Correlations(BaseModel):
    pearson: Correlation = Correlation(key="pearson")
    spearman: Correlation = Correlation(key="spearman")
    auto: Correlation = Correlation(key="auto")


class Interactions(BaseModel):
    # Set to False to disable scatter plots
    continuous: bool = True

    targets: List[str] = []


class Samples(BaseModel):
    head: int = 10
    tail: int = 10
    random: int = 0


class Variables(BaseModel):
    descriptions: dict = {}


class IframeAttribute(Enum):
    src = "src"
    srcdoc = "srcdoc"


class Iframe(BaseModel):
    height: str = "800px"
    width: str = "100%"
    attribute: IframeAttribute = IframeAttribute.srcdoc


class Notebook(BaseModel):
    """When in a Jupyter notebook"""

    iframe: Iframe = Iframe()


class Report(BaseModel):
    # Numeric precision for displaying statistics
    precision: int = 8


class Settings(BaseSettings):
    # Default prefix to avoid collisions with environment variables
    class Config:
        env_prefix = "profile_"

    # Title of the document
    title: str = "YData Profiling Report"

    dataset: Dataset = Dataset()
    variables: Variables = Variables()
    infer_dtypes: bool = True

    # Show the description at each variable (in addition to the overview tab)
    show_variable_description: bool = True

    # Number of workers (0=multiprocessing.cpu_count())
    pool_size: int = 0

    # Show the progress bar
    progress_bar: bool = True

    # Per variable type description settings
    vars: Univariate = Univariate()

    # Sort the variables. Possible values: ascending, descending or None (leaves original sorting)
    sort: Optional[str] = None

    missing_diagrams: Dict[str, bool] = {
        "bar": True,
        "matrix": True,
        "heatmap": True,
    }

    correlation_table: bool = True

    correlations: Dict[str, Correlation] = {
        "auto": Correlation(key="auto", calculate=True),
        "spearman": Correlation(key="spearman", calculate=False),
        "pearson": Correlation(key="pearson", calculate=False),
        "phi_k": Correlation(key="phi_k", calculate=False),
        "cramers": Correlation(key="cramers", calculate=False),
        "kendall": Correlation(key="kendall", calculate=False),
    }

    interactions: Interactions = Interactions()

    categorical_maximum_correlation_distinct: int = 100
    # Use `deep` flag for memory_usage
    memory_deep: bool = False
    plot: Plot = Plot()
    duplicates: Duplicates = Duplicates()
    samples: Samples = Samples()

    reject_variables: bool = True

    # The number of observations to show
    n_obs_unique: int = 10
    n_freq_table_max: int = 10
    n_extreme_obs: int = 10

    # Report rendering
    report: Report = Report()
    html: Html = Html()
    notebook: Notebook = Notebook()

    def update(self, updates: dict) -> "Settings":
        update = _merge_dictionaries(self.dict(), updates)
        return self.parse_obj(self.copy(update=update))

    @staticmethod
    def from_file(config_file: Union[Path, str]) -> "Settings":
        """Create a Settings object from a yaml file.

        Args:
            config_file: yaml file path
        Returns:
            Settings
        """
        with open(config_file) as f:
            data = yaml.safe_load(f)

        return Settings.parse_obj(data)


class SparkSettings(Settings):
    """
    Setting class with the standard report configuration for Spark DataFrames
    All the supported analysis are set to true
    """

    vars: Univariate = Univariate()

    vars.num.low_categorical_threshold = 0

    infer_dtypes: bool = False

    correlations: Dict[str, Correlation] = {
        "spearman": Correlation(key="spearman", calculate=True),
        "pearson": Correlation(key="pearson", calculate=True),
    }

    correlation_table: bool = True

    interactions: Interactions = Interactions()
    interactions.continuous = False

    missing_diagrams: Dict[str, bool] = {
        "bar": False,
        "matrix": False,
        "dendrogram": False,
        "heatmap": False,
    }
    samples: Samples = Samples()
    samples.tail = 0
    samples.random = 0


class Config:
    arg_groups: Dict[str, Any] = {
        "sensitive": {
            "samples": None,
            "duplicates": None,
            "vars": {"cat": {"redact": True}, "text": {"redact": True}},
        },
        "flatly_theme": {
            "html": {
                "style": {
                    "theme": Theme.flatly,
                    "primary_color": "#2c3e50",
                }
            }
        },
        "united_theme": {
            "html": {
                "style": {
                    "theme": Theme.united,
                    "primary_color": "#d34615",
                }
            }
        },
        "explorative": {
            "vars": {
                "cat": {"characters": True, "words": True},
                "url": {"active": True},
                "path": {"active": True},
                "file": {"active": True},
                "image": {"active": True},
            },
            "n_obs_unique": 10,
            "n_extreme_obs": 10,
            "n_freq_table_max": 10,
            "memory_deep": True,
        },
    }

    _shorthands = {
        "dataset": {
            "creator": "",
            "author": "",
            "description": "",
            "copyright_holder": "",
            "copyright_year": "",
            "url": "",
        },
        "samples": {"head": 0, "tail": 0, "random": 0},
        "duplicates": {"head": 0},
        "interactions": {"targets": [], "continuous": False},
        "missing_diagrams": {
            "bar": False,
            "matrix": False,
            "heatmap": False,
        },
        "correlations": {
            "auto": {"calculate": False},
            "pearson": {"calculate": False},
            "spearman": {"calculate": False},
            "kendall": {"calculate": False},
            "phi_k": {"calculate": False},
            "cramers": {"calculate": False},
        },
        "correlation_table": True,
    }

    @staticmethod
    def get_arg_groups(key: str) -> dict:
        kwargs = Config.arg_groups[key]
        shorthand_args, _ = Config.shorthands(kwargs, split=False)
        return shorthand_args

    @staticmethod
    def shorthands(kwargs: dict, split: bool = True) -> Tuple[dict, dict]:
        shorthand_args = {}
        if not split:
            shorthand_args = kwargs
        for key, value in list(kwargs.items()):
            if value is None and key in Config._shorthands:
                shorthand_args[key] = Config._shorthands[key]
                if split:
                    del kwargs[key]

        if split:
            return shorthand_args, kwargs
        else:
            return shorthand_args, {}
