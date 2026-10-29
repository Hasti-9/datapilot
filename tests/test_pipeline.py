import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

os.environ["DATAPILOT_OFFLINE"] = "1"  # deterministic, no API calls

from datapilot.cleaning import clean_dataframe
from datapilot.graph import DataPilot
from datapilot.ml import train_and_compare
from datapilot.sql import SQLEngine

SAMPLE = Path(__file__).parents[1] / "sample_data" / "customer_churn.csv"


@pytest.fixture(scope="module")
def pilot():
    return DataPilot(str(SAMPLE), "churn", cv_folds=3)


def test_cleaning_fixes_messy_values():
    raw = pd.DataFrame({"Price ($)": ["$1,200", " 30 ", "N/A", "45"], "Flag": ["yes", "no", "yes", None],
                        "Empty": [None] * 4})
    df, rep = clean_dataframe(raw)
    assert list(df.columns) == ["price", "flag"]
    assert pd.api.types.is_numeric_dtype(df["price"]) and df["price"].isna().sum() == 0
    assert df["price"].iloc[0] == 1200
    assert pd.api.types.is_bool_dtype(df["flag"])
    assert rep.steps


def test_analysis_graph_runs_end_to_end(pilot):
    s = pilot.state
    assert [t.split(":")[0] for t in s["trace"]] == [
        "ingest", "clean", "eda", "visualize", "ml", "build_kb", "insights"]
    assert len(s["df"]) == 2000  # 25 duplicates removed
    assert "customer_id" not in s["ml"].features  # identifier excluded
    assert s["ml"].task == "classification"
    best = s["ml"].leaderboard.iloc[0]
    baseline = s["ml"].leaderboard.query("model.str.startswith('Baseline')").iloc[0]
    assert best["test_f1"] > baseline["test_f1"]
    assert s["charts"] and s["insights"]


def test_regression_task():
    rng = np.random.default_rng(0)
    x = rng.normal(size=400)
    df = pd.DataFrame({"x": x, "g": rng.choice(list("abc"), 400), "y": 3 * x + rng.normal(0, 0.1, 400)})
    res = train_and_compare(df, "y", cv_folds=3)
    assert res.task == "regression"
    assert res.leaderboard.iloc[0]["test_r2"] > 0.9
    assert res.feature_importance.iloc[0]["feature"] == "x"


def test_sql_guardrails(pilot):
    eng = pilot.sql_engine
    for bad in ["DROP TABLE data", "SELECT 1; DELETE FROM data", "SELECT * FROM read_csv('x.csv')"]:
        with pytest.raises(ValueError):
            eng.run(bad)
    assert eng.run("SELECT COUNT(*) AS n FROM data").iat[0, 0] == 2000


def test_sql_heuristic_fallback(pilot):
    res = pilot.sql_engine.ask("average monthly charges by contract type")
    assert res.error is None, res.error
    assert res.result.shape == (3, 2)


def test_chat_routing(pilot):
    assert pilot.ask("how many rows per region")["route"] == "sql"
    assert pilot.ask("which feature importance matters most for the model")["route"] == "explain"
    out = pilot.ask("what cleaning was done to the data?")
    assert out["route"] == "rag" and out["sources"]
