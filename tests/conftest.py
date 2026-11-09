import os
import shutil
import socket
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

try:
    from pyspark import SparkConf, SparkContext
    from pyspark.sql import SparkSession

    has_spark = True
except ImportError:
    has_spark = False

from data_profiling.config import Settings
from data_profiling.model.summarizer import ProfilingSummarizer
from data_profiling.model.typeset import ProfilingTypeSet

# Tests never report usage analytics.
os.environ.setdefault("YDATA_PROFILING_NO_ANALYTICS", "1")


def pytest_configure(config):
    config.addinivalue_line("markers", "linux: Test with linux")
    config.addinivalue_line("markers", "win32: Test with windows")
    config.addinivalue_line("markers", "darwin: Test with darwin")
    config.addinivalue_line(
        "markers", "network: test needs internet access (blocked otherwise)"
    )


# ---------------------------------------------------------------------------
# Offline guard
#
# The suite must not depend on remote datasets: a slow or unavailable server
# used to hang CI. Outbound connections are refused unless a test is marked
# ``@pytest.mark.network``. Loopback stays allowed (Spark talks to itself over
# 127.0.0.1).
# ---------------------------------------------------------------------------

_LOOPBACK = {"127.0.0.1", "::1", "localhost", "0.0.0.0"}
_real_connect = socket.socket.connect


def _guarded_connect(sock: socket.socket, address) -> None:  # noqa: ANN001
    host = address[0] if isinstance(address, tuple) else None
    if host is not None and host not in _LOOPBACK:
        raise RuntimeError(
            f"Network access is disabled in tests (tried to reach {host}). "
            "Build the data in the test, or mark it with @pytest.mark.network."
        )
    return _real_connect(sock, address)


@pytest.fixture(autouse=True)
def _block_network(request, monkeypatch):  # noqa: ANN001
    if request.node.get_closest_marker("network") is None:
        monkeypatch.setattr(socket.socket, "connect", _guarded_connect)
    yield


# ---------------------------------------------------------------------------
# Local stand-ins for the public datasets the tests used to download.
# Same column names and kinds of values; seeded, so results are reproducible.
# ---------------------------------------------------------------------------


def make_meteorites(n: int = 1000, seed: int = 7331) -> pd.DataFrame:
    """Shaped like NASA's Meteorite Landings CSV (as read by ``pd.read_csv``)."""
    rng = np.random.default_rng(seed)
    classes = ["L6", "H5", "L5", "H6", "H4", "LL5", "LL6", "L4", "CM2", "Iron, IIIAB"]
    years = rng.integers(1700, 2014, n)
    years[rng.random(n) < 0.02] = 860  # pre-1880 dates become NaT downstream
    lat = np.round(rng.uniform(-80, 80, n), 5)
    lon = np.round(rng.uniform(-180, 180, n), 5)
    missing_geo = rng.random(n) < 0.15
    lat[missing_geo] = np.nan
    lon[missing_geo] = np.nan
    mass = np.round(rng.lognormal(3, 2.5, n), 2)
    mass[rng.random(n) < 0.03] = np.nan
    return pd.DataFrame(
        {
            "name": [f"Meteorite {i:05d}" for i in range(n)],
            "id": np.arange(1, n + 1),
            "nametype": rng.choice(["Valid", "Relict"], n, p=[0.98, 0.02]),
            "recclass": rng.choice(classes, n),
            "mass (g)": mass,
            "fall": rng.choice(["Found", "Fell"], n, p=[0.9, 0.1]),
            "year": [f"01/01/{y:04d} 12:00:00 AM" for y in years],
            "reclat": lat,
            "reclong": lon,
            "GeoLocation": [
                None if np.isnan(a) else f"({a}, {b})" for a, b in zip(lat, lon)
            ],
        }
    )


def make_titanic(n: int = 300, seed: int = 1912) -> pd.DataFrame:
    """Shaped like the Kaggle Titanic training CSV."""
    rng = np.random.default_rng(seed)
    sex = rng.choice(["male", "female"], n, p=[0.65, 0.35])
    age = np.round(rng.normal(30, 14, n).clip(0.4, 80), 1)
    age[rng.random(n) < 0.2] = np.nan
    cabin = np.array([f"C{rng.integers(1, 150)}" for _ in range(n)], dtype=object)
    cabin[rng.random(n) < 0.77] = None
    embarked = rng.choice(["S", "C", "Q"], n, p=[0.72, 0.19, 0.09]).astype(object)
    embarked[:2] = None
    return pd.DataFrame(
        {
            "PassengerId": np.arange(1, n + 1),
            "Survived": rng.integers(0, 2, n),
            "Pclass": rng.choice([1, 2, 3], n, p=[0.24, 0.21, 0.55]),
            "Name": [f"Passenger, Mr. Person {i}" for i in range(n)],
            "Sex": sex,
            "Age": age,
            "SibSp": rng.choice([0, 1, 2, 3], n, p=[0.68, 0.23, 0.06, 0.03]),
            "Parch": rng.choice([0, 1, 2], n, p=[0.76, 0.13, 0.11]),
            "Ticket": [f"A/{rng.integers(1000, 99999)}" for _ in range(n)],
            "Fare": np.round(rng.lognormal(2.8, 1.0, n), 4),
            "Cabin": cabin,
            "Embarked": embarked,
        }
    )


@pytest.fixture
def meteorites_csv(tmp_path: Path) -> Path:
    path = tmp_path / "meteorites.csv"
    make_meteorites().to_csv(path, index=False)
    return path


@pytest.fixture
def titanic_csv(tmp_path: Path) -> Path:
    path = tmp_path / "titanic.csv"
    make_titanic().to_csv(path, index=False)
    return path


@pytest.fixture(scope="module")
def test_output_dir(tmpdir_factory):
    test_path = Path(str(tmpdir_factory.mktemp("test")))
    yield test_path
    shutil.rmtree(str(test_path))


@pytest.fixture(scope="function")
def summarizer(typeset):
    return ProfilingSummarizer(typeset)


@pytest.fixture(scope="function")
def summarizer_spark(typeset):
    return ProfilingSummarizer(typeset, use_spark=True)


@pytest.fixture(scope="function")
def config():
    return Settings()


@pytest.fixture(scope="function")
def typeset(config):
    return ProfilingTypeSet(config)


def pytest_runtest_setup(item):
    platforms = {"darwin", "linux", "win32"}
    supported_platforms = platforms.intersection(
        mark.name for mark in item.iter_markers()
    )
    plat = sys.platform
    if supported_platforms and plat not in supported_platforms:
        pytest.skip(f"cannot run on platform {plat}")


@pytest.fixture(scope="session")
def spark_context():
    """Fixture for SparkContext initialization.

    Ensures a single SparkContext instance is created for all tests.
    """

    if not has_spark:
        pytest.skip("Skipping Spark tests because PySpark is not installed.")

    conf = (
        SparkConf()
        .setAppName("pytest-pyspark-tests")
        .setMaster("local[*]")
        .set("spark.sql.ansi.enabled", "false")
        .set("spark.driver.host", "127.0.0.1")
        .set("spark.driver.bindAddress", "127.0.0.1")
        .set("spark.driver.port", "4040")
        .set("spark.blockManager.port", "4041")
    )

    # Check if SparkContext exists before creating a new one
    if SparkContext._active_spark_context:
        sc = SparkContext._active_spark_context
    else:
        sc = SparkContext(conf=conf)

    yield sc

    # Cleanup
    sc.stop()


@pytest.fixture(scope="session")
def spark_session(spark_context):
    """Fixture for SparkSession initialization.

    Ensures SparkSession is created with the existing SparkContext.
    """
    if not has_spark:
        pytest.skip("Skipping Spark tests because PySpark is not installed.")
    spark = (
        SparkSession.builder.master("local[*]")
        .config("spark.driver.host", "127.0.0.1")
        .config("spark.driver.bindAddress", "127.0.0.1")
        .appName("pytest")
        .config("spark.sql.ansi.enabled", "false")  # <-- restore permissive casts
        .getOrCreate()
    )

    yield spark

    # Cleanup
    spark.stop()
