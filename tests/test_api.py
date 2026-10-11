"""API behaviour and the end-to-end workflow with mocked market data."""
import time
from datetime import date

import pytest

from app.core.errors import ProviderRateLimitError


def wait_task(client, task_id, timeout=180):
    t0 = time.time()
    while time.time() - t0 < timeout:
        t = client.get(f"/tasks/{task_id}").json()
        if t["status"] in ("succeeded", "failed"):
            return t
        time.sleep(0.2)
    raise AssertionError("task timeout")


def test_health_and_stock_list(client):
    r = client.get("/health")
    assert r.status_code == 200 and r.json()["status"] == "ok" and r.json()["database"] is True
    assert "naive" in r.json()["algorithms"]
    r = client.get("/stocks", params={"q": "reli"})
    assert r.status_code == 200 and r.json()["stocks"][0]["symbol"] == "RELIANCE"
    assert "seed" in r.json()["note"]


def test_history_for_valid_symbol_includes_source_and_freshness(client, prices):
    r = client.get("/stocks/testco/history", params={"start": "2022-01-01"})
    assert r.status_code == 200
    body = r.json()
    assert body["symbol"] == "TESTCO" and body["currency"] == "INR" and body["count"] > 100
    assert body["data"]["source"] == "fake" and body["data"]["last_date"] == str(prices.index[-1].date())
    assert "not real-time" in body["note"]
    assert set(body["rows"][0]) >= {"date", "open", "high", "low", "close", "volume"}


@pytest.mark.parametrize("path,status,code", [
    ("/stocks/NOSUCH/history", 404, "symbol_not_found"),
    ("/stocks/BAD SYM/history", 422, "invalid_symbol"),
    ("/stocks/TESTCO/history?start=notadate", 422, "validation_error"),
    ("/stocks/TESTCO/models/999/backtest", 404, "not_found"),
    ("/tasks/does-not-exist", 404, "not_found"),
])
def test_structured_errors(client, path, status, code):
    r = client.get(path)
    assert r.status_code == status
    assert r.json()["error"]["code"] == code and r.json()["error"]["message"]


def test_provider_rate_limit_maps_to_429(client, provider):
    provider.fail_with = ProviderRateLimitError("slow down")
    r = client.get("/stocks/TESTCO/history")
    assert r.status_code == 429 and r.json()["error"]["code"] == "rate_limited"


def test_request_validation(client):
    assert client.post("/stocks/TESTCO/train", json={"horizons": [0]}).status_code == 422
    assert client.post("/stocks/TESTCO/train", json={"horizons": [99]}).status_code == 422
    assert client.post("/stocks/TESTCO/train", json={"algorithms": ["magic"]}).status_code == 422
    assert client.post("/stocks/TESTCO/train", json={"start": "2024-01-01", "end": "2023-01-01"}).status_code == 422
    assert client.post("/stocks/TESTCO/predict", json={"horizons": []}).status_code == 422
    assert client.post("/data/refresh", json={"symbols": []}).status_code == 422


def test_predict_without_trained_model_is_409(client):
    client.get("/stocks/TESTCO/history")
    r = client.post("/stocks/TESTCO/predict", json={"horizons": [1]})
    assert r.status_code == 409 and r.json()["error"]["code"] == "no_model"


def test_full_workflow_train_predict_history_performance(client, provider, prices):
    from tests.conftest import make_prices
    full = make_prices(701)                      # the last session "has not happened yet"
    provider.frames["TESTCO"] = full.iloc[:-1]
    # 1. data -> 2. async training
    r = client.post("/stocks/TESTCO/train", json={"algorithms": ["ridge", "random_forest", "xgboost"],
                                                    "horizons": [1, 5]})
    assert r.status_code == 202
    task_id = r.json()["task_id"]
    dup = client.post("/stocks/TESTCO/train", json={"horizons": [1]})
    t = wait_task(client, task_id)
    assert t["status"] == "succeeded", t
    assert dup.status_code in (202, 409)      # a concurrent request is rejected, never double-run
    assert t["result"]["summary"]["horizons"]["1"]["selected"]

    # 3. models + metrics + backtest
    m = client.get("/stocks/TESTCO/models").json()
    assert m["count"] == 4 * 2 and "feature_names" in m
    sel = client.get("/stocks/TESTCO/models", params={"selected_only": True, "horizon": 1}).json()["models"]
    assert len(sel) == 1 and sel[0]["is_selected"]
    bt = client.get(f"/stocks/TESTCO/models/{sel[0]['id']}/backtest").json()
    assert bt["kind"] == "backtest" and len(bt["rows"]) == sel[0]["metrics"]["n"]

    # 4. forecast, persisted + idempotent
    p = client.post("/stocks/TESTCO/predict", json={"horizons": [1, 5]})
    assert p.status_code == 200, p.text
    body = p.json()
    assert body["kind"] == "forecast" and len(body["predictions"]) == 2 and body["disclaimer"]
    assert all(x["predicted_price"] > 0 for x in body["predictions"])
    hist = client.get("/stocks/TESTCO/predictions").json()
    assert hist["count"] == 2 and all(x["outcome"] is None for x in hist["predictions"])
    client.post("/stocks/TESTCO/predict", json={"horizons": [1, 5]})
    assert client.get("/stocks/TESTCO/predictions").json()["count"] == 2

    # 5. nothing matured yet
    perf = client.get("/stocks/TESTCO/performance").json()
    assert perf["pending"] == 2 and perf["resolved"] == 0

    # 6. new session arrives -> record the actual and the error
    provider.frames["TESTCO"] = full
    up = client.post("/stocks/TESTCO/outcomes/update").json()
    assert up["newly_resolved"] == 1 and up["still_pending"] == 1
    perf = client.get("/stocks/TESTCO/performance").json()
    assert perf["resolved"] == 1
    o = perf["outcomes"][0]["outcome"]
    assert o["abs_error"] == pytest.approx(abs(o["error"])) and "pct_error" in o
    assert perf["by_model"][0]["naive_metrics"]["n"] == 1


def test_background_refresh_task(client):
    r = client.post("/data/refresh", json={"symbols": ["testco", "NOSUCH"]})
    assert r.status_code == 202
    t = wait_task(client, r.json()["task_id"])
    assert t["status"] == "succeeded"
    assert t["result"]["TESTCO"]["n_rows"] > 0 and "error" in t["result"]["NOSUCH"]


def test_universe_import_endpoint(client, tmp_path):
    f = tmp_path / "EQUITY_L.csv"
    f.write_text("SYMBOL,NAME OF COMPANY, SERIES\nAAA,Aaa Ltd,EQ\nBBB,Bbb Ltd,EQ\n")
    r = client.post("/data/universe/import", json={"source": str(f)})
    assert r.status_code == 200 and r.json()["newly_inserted"] == 2
    assert client.post("/data/universe/import", json={"source": str(tmp_path / "x.csv")}).status_code == 502
