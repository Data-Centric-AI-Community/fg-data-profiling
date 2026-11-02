"""Agent JSON and roles: Spark backend produces the same contract as pandas."""

import json

import numpy as np
import pandas as pd
import pytest

from data_profiling import ProfileReport, agent


@pytest.fixture
def frame():
    rng = np.random.default_rng(7)
    n = 400
    data = pd.DataFrame(
        {
            "customer_id": rng.permutation(np.arange(1000, 1000 + n)),
            "zip": [f"{v:05d}" for v in rng.integers(1000, 99999, n)],
            "price": rng.lognormal(3, 1, n).round(2),
            "country": rng.choice(["PT", "ES", "FR"], n),
            "review": [
                " ".join(rng.choice(["good", "bad", "slow"], int(rng.integers(3, 9))))
                for _ in range(n)
            ],
        }
    )
    data.loc[rng.choice(n, 40, replace=False), "price"] = np.nan
    return data


def test_agent_json_spark_matches_pandas(spark_session, frame):
    sdf = spark_session.createDataFrame(frame)
    spark_payload = json.loads(ProfileReport(sdf, progress_bar=False).to_json())
    pandas_payload = json.loads(
        ProfileReport(frame, progress_bar=False, infer_dtypes=False).to_json()
    )

    jsonschema = pytest.importorskip("jsonschema")
    jsonschema.validate(spark_payload, agent.json_schema())

    for col in frame.columns:
        s, p = spark_payload["columns"][col], pandas_payload["columns"][col]
        assert s["role"] == p["role"], col
        assert s.get("missing_pct", 0) == pytest.approx(p.get("missing_pct", 0))

    spark_tags = {(f["tag"], tuple(f["cols"])) for f in spark_payload["findings"]}
    assert ("missing", ("price",)) in spark_tags


def test_roles_spark(spark_session, frame):
    report = ProfileReport(spark_session.createDataFrame(frame), progress_bar=False)
    roles = {c: r["role"] for c, r in report.get_roles().items()}
    assert roles["customer_id"] == "identifier"
    assert roles["zip"] == "code"
    assert roles["country"] == "dimension"
    assert roles["review"] == "free_text"
    assert "dp-role dp-role-" in report.to_html()
