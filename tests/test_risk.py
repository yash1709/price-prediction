"""Risk tools: stock risk card, position sizing, portfolio risk (synthetic data, no network)."""
import math

import numpy as np
import pandas as pd
import pytest

from app.services.risk import shrunk_corr
from tests.conftest import make_prices


@pytest.fixture
def two_stocks(provider, prices):
    provider.frames["OTHER"] = make_prices(700, seed=5, start_price=300)
    provider.frames["TWIN"] = prices.copy()          # identical price history to TESTCO
    return provider


def test_stock_risk_card_structure_and_consistency(client, two_stocks):
    r = client.get("/stocks/TESTCO/risk")
    assert r.status_code == 200, r.text
    b = r.json()
    v = b["volatility"]
    m1 = b["expected_moves"]["1"]["80"]
    assert v["forecast_daily_pct"] == pytest.approx((m1["upper_pct"] - m1["lower_pct"]) / (2 * 1.2815515655446004))
    # regression guard: the displayed volatility must be on the same scale as a standard volatility
    # estimate (the raw model output is ~0.5x that and must never be shown as volatility)
    assert 0.6 < v["forecast_daily_pct"] / v["ewma_daily_pct"] < 1.8
    assert v["forecast_annualised_pct"] == pytest.approx(v["forecast_daily_pct"] * math.sqrt(252))
    assert set(v["realised_annualised_pct"]) == {"20d", "60d", "252d"}
    for h in ("1", "5"):
        m = b["expected_moves"][h]
        assert m["95"]["lower"] < m["80"]["lower"] < m["50"]["lower"] < b["last_close"] < m["50"]["upper"] \
            < m["80"]["upper"] < m["95"]["upper"]
    bd = b["bad_day_thresholds"]
    assert bd["1_in_40"]["threshold_pct"] < bd["1_in_10"]["threshold_pct"] < 0     # rarer = more extreme
    dd = b["drawdown"]
    assert dd["max_1y_pct"] <= dd["current_from_1y_peak_pct"] <= 0 and dd["max_3y_pct"] <= dd["max_1y_pct"]
    assert 0 <= b["range_52w"]["position_pct"] <= 100
    assert b["beta_vs_nifty_1y"] is not None and b["avg_daily_traded_value_inr"] > 0
    assert client.get("/stocks/NOSUCH/risk").status_code == 404


def test_position_size_matches_formula_and_caps(client, two_stocks):
    risk = client.get("/stocks/TESTCO/risk").json()
    body = {"symbol": "TESTCO", "portfolio_value": 1_000_000, "risk_pct": 1.0, "horizon": 1, "tail": "1_in_10"}
    p = client.post("/risk/position-size", json=body).json()
    loss = abs(risk["expected_moves"]["1"]["80"]["lower_pct"]) / 100
    assert p["risk_budget_inr"] == pytest.approx(10_000)
    assert p["max_position_inr"] == pytest.approx(10_000 / loss)
    assert p["shares"] == int(p["max_position_inr"] // risk["last_close"])
    assert p["loss_if_bad_move_inr"] == pytest.approx(p["position_value_inr"] * loss)
    assert p["loss_if_bad_move_inr"] <= 10_000 + 1e-6                   # never exceeds the budget
    assert p["capped_at_portfolio"] is False
    # a rarer, more extreme bad move => smaller position
    p40 = client.post("/risk/position-size", json={**body, "tail": "1_in_40"}).json()
    assert p40["max_position_inr"] < p["max_position_inr"]
    # a huge risk budget is capped at the portfolio value (no leverage)
    big = client.post("/risk/position-size", json={**body, "risk_pct": 100}).json()
    assert big["capped_at_portfolio"] is True and big["max_position_inr"] == pytest.approx(1_000_000)


@pytest.mark.parametrize("patch", [{"portfolio_value": -5}, {"risk_pct": 0}, {"risk_pct": 150},
                                   {"tail": "bogus"}, {"horizon": 3}])
def test_position_size_validation(client, two_stocks, patch):
    body = {"symbol": "TESTCO", "portfolio_value": 500_000, "risk_pct": 1.0, "horizon": 1,
            "tail": "1_in_10", **patch}
    assert client.post("/risk/position-size", json=body).status_code == 422


def test_portfolio_invariants(client, two_stocks):
    r = client.post("/risk/portfolio", json={"holdings": [
        {"symbol": "TESTCO", "amount": 600_000}, {"symbol": "other", "amount": 400_000}]})
    assert r.status_code == 200, r.text
    b = r.json()
    hs = {h["symbol"]: h for h in b["holdings"]}
    assert sum(h["weight_pct"] for h in hs.values()) == pytest.approx(100)
    assert sum(h["risk_contribution_pct"] for h in hs.values()) == pytest.approx(100)
    w = {s: h["weight_pct"] / 100 for s, h in hs.items()}
    s = {s: h["daily_volatility_pct"] for s, h in hs.items()}
    sp = b["portfolio"]["daily_volatility_pct"]
    assert abs(w["TESTCO"] * s["TESTCO"] - w["OTHER"] * s["OTHER"]) - 1e-9 <= sp \
        <= w["TESTCO"] * s["TESTCO"] + w["OTHER"] * s["OTHER"] + 1e-9    # between perfect +/- correlation
    assert b["portfolio"]["diversification_ratio"] >= 1 - 1e-9
    cm = np.array(b["correlation_matrix"]["values"])
    assert np.allclose(cm, cm.T) and np.allclose(np.diag(cm), 1)
    assert "not_provided" in b and "calibrated portfolio forecast range" in b["not_provided"]
    assert not any("range" in k for k in b["portfolio"])                 # no portfolio ranges offered


def test_identical_stocks_give_no_diversification(client, two_stocks):
    b = client.post("/risk/portfolio", json={"holdings": [
        {"symbol": "TESTCO", "amount": 300_000}, {"symbol": "TWIN", "amount": 700_000}]}).json()
    assert b["portfolio"]["diversification_ratio"] == pytest.approx(1.0)      # perfectly correlated
    p = b["portfolio"]
    assert p["daily_volatility_pct"] == pytest.approx(p["weighted_avg_holding_volatility_pct"])
    assert any("highly correlated" in w for w in b["warnings"])


def test_single_holding_and_duplicate_merge_and_validation(client, two_stocks):
    one = client.post("/risk/portfolio", json={"holdings": [{"symbol": "TESTCO", "amount": 100_000}]}).json()
    assert one["portfolio"]["daily_volatility_pct"] == pytest.approx(one["holdings"][0]["daily_volatility_pct"])
    assert any("Fewer than 5" in w for w in one["warnings"])
    merged = client.post("/risk/portfolio", json={"holdings": [
        {"symbol": "TESTCO", "amount": 40_000}, {"symbol": "testco.ns", "amount": 60_000}]}).json()
    assert len(merged["holdings"]) == 1 and merged["holdings"][0]["amount_inr"] == pytest.approx(100_000)
    assert client.post("/risk/portfolio", json={"holdings": []}).status_code == 422
    assert client.post("/risk/portfolio", json={"holdings": [{"symbol": "TESTCO", "amount": 0}]}).status_code == 422
    assert client.post("/risk/portfolio", json={"holdings": [{"symbol": "NOSUCH", "amount": 5}]}).status_code == 404
    too_many = [{"symbol": f"S{i}", "amount": 1} for i in range(26)]
    assert client.post("/risk/portfolio", json={"holdings": too_many}).status_code == 422


def test_shrunk_corr_properties():
    rng = np.random.default_rng(0)
    base = rng.normal(size=400)
    r = pd.DataFrame({"a": base + rng.normal(scale=0.5, size=400), "b": base + rng.normal(scale=0.5, size=400),
                      "c": rng.normal(size=400)})
    raw = r.corr().values
    sh = shrunk_corr(r)
    assert np.allclose(sh, sh.T) and np.allclose(np.diag(sh), 1)
    off = ~np.eye(3, dtype=bool)
    avg = raw[off].mean()
    for i in range(3):
        for j in range(3):
            if i != j:
                assert min(raw[i, j], avg) - 1e-12 <= sh[i, j] <= max(raw[i, j], avg) + 1e-12
                assert sh[i, j] == pytest.approx(0.8 * raw[i, j] + 0.2 * avg)
    assert np.allclose(shrunk_corr(r[["a"]]), [[1.0]])
