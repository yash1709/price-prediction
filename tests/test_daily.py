"""Daily forward-testing job: forecasts, idempotence, outcome resolution, failure handling."""
import json
from datetime import datetime, timezone

import pandas as pd
import pytest

from app.models.intervals import IntervalService
from app.services.daily import DailyConfig, coverage_stats, run_daily
from tests.conftest import FakeProvider, make_market, make_prices

NOW = datetime(2024, 7, 1, 12, 0, tzinfo=timezone.utc)
CFG = DailyConfig(pause_seconds=0, stale_days=60)


@pytest.fixture
def full():
    return {"AAA": make_prices(706, seed=1), "BBB": make_prices(706, seed=2, start_price=250)}


@pytest.fixture
def svc(settings, full):
    market = make_market(full["AAA"].index)
    return IntervalService(settings, lambda: market)


def _run(provider, symbols, out, svc, settings, now=NOW):
    return run_daily(provider, symbols, out, svc, settings, now=now, cfg=CFG, say=lambda m: None)


def test_first_run_publishes_nested_ranges_and_reports_failures(tmp_path, settings, full, svc):
    provider = FakeProvider({k: v.iloc[:-6] for k, v in full.items()})
    syms = [("AAA", "Alpha Ltd"), ("BBB", "Beta Ltd"), ("MISSING", "Nope Ltd")]
    s = _run(provider, syms, tmp_path, svc, settings)
    assert s["symbols_ok"] == 2 and s["status"] == "degraded" and s["n_failed"] == 1
    assert s["failed"][0]["symbol"] == "MISSING"
    f = pd.read_csv(tmp_path / "forecasts.csv")
    assert len(f) == 4 and set(f["horizon"]) == {1, 5}
    assert (f["as_of_date"] == str(full["AAA"].index[-7].date())).all()
    assert (f["point_forecast"] == f["reference_close"]).all()          # point forecast = no change
    assert (f["lo95"] < f["lo80"]).all() and (f["lo80"] < f["lo50"]).all()
    assert (f["lo50"] < f["reference_close"]).all() and (f["reference_close"] < f["hi50"]).all()
    assert (f["hi50"] < f["hi80"]).all() and (f["hi80"] < f["hi95"]).all()
    latest = pd.read_csv(tmp_path / "latest_forecasts.csv")
    assert set(latest["name"]) == {"Alpha Ltd", "Beta Ltd"}
    assert json.loads((tmp_path / "summary.json").read_text())["coverage"]["1"] == {"n": 0}


def test_rerun_same_day_is_idempotent(tmp_path, settings, full, svc):
    provider = FakeProvider({k: v.iloc[:-6] for k, v in full.items()})
    syms = [("AAA", None), ("BBB", None)]
    _run(provider, syms, tmp_path, svc, settings)
    s2 = _run(provider, syms, tmp_path, svc, settings)
    assert s2["new_forecasts"] == 0 and s2["forecasts_total"] == 4 and s2["outcomes_total"] == 0


def test_outcomes_resolve_when_target_sessions_arrive(tmp_path, settings, full, svc):
    provider = FakeProvider({k: v.iloc[:-6] for k, v in full.items()})
    syms = [("AAA", None), ("BBB", None)]
    _run(provider, syms, tmp_path, svc, settings)
    old_asof = full["AAA"].index[-7]
    provider.frames = dict(full)                    # six more sessions have happened
    s = _run(provider, syms, tmp_path, svc, settings)
    assert s["new_outcomes"] == 4 and s["new_forecasts"] == 4   # old forecasts scored, new ones made
    o = pd.read_csv(tmp_path / "outcomes.csv")
    f = pd.read_csv(tmp_path / "forecasts.csv")
    for r in o.itertuples():
        df = full[r.symbol]
        pos = df.index.get_loc(old_asof)
        assert r.actual_date == str(df.index[pos + r.horizon].date())
        assert r.actual_close == pytest.approx(df["close"].iloc[pos + r.horizon])
        ret = df["adj_close"].iloc[pos + r.horizon] / df["adj_close"].iloc[pos] - 1
        assert r.ret_pct == pytest.approx(ret * 100)
        fr = f[(f.as_of_date == r.as_of_date) & (f.symbol == r.symbol) & (f.horizon == r.horizon)].iloc[0]
        ref = fr.reference_close
        assert bool(r.inside80) == ((fr.lo80 / ref - 1) <= ret <= (fr.hi80 / ref - 1))
        assert bool(r.inside95) >= bool(r.inside80) >= bool(r.inside50)       # ranges are nested
    cov = s["coverage"]["1"]
    assert cov["n"] == 2 and 0 <= cov["inside80_pct"] <= 100
    s3 = _run(provider, syms, tmp_path, svc, settings)                        # nothing new: stable
    assert s3["new_outcomes"] == 0 and s3["outcomes_total"] == 4
    assert (tmp_path / "coverage_by_day.csv").exists()


def test_horizon5_waits_for_five_sessions(tmp_path, settings, full, svc):
    provider = FakeProvider({k: v.iloc[:-3] for k, v in full.items()})
    syms = [("AAA", None)]
    _run(provider, syms, tmp_path, svc, settings)
    provider.frames = {"AAA": full["AAA"].iloc[:-1]}          # 2 new sessions: h=1 matured, h=5 not
    s = _run(provider, syms, tmp_path, svc, settings)
    o = pd.read_csv(tmp_path / "outcomes.csv")
    assert set(o["horizon"]) == {1} and s["new_outcomes"] == 1


def test_stale_data_is_skipped_and_everything_failing_is_status_failed(tmp_path, settings, full, svc):
    provider = FakeProvider({"AAA": full["AAA"].iloc[:-6]})
    s = run_daily(provider, [("AAA", None), ("X", None)], tmp_path, svc, settings,
                  now=datetime(2024, 9, 30, tzinfo=timezone.utc),       # months after the last bar
                  cfg=DailyConfig(pause_seconds=0, stale_days=6), say=lambda m: None)
    assert s["status"] == "failed" and s["symbols_ok"] == 0
    assert {x["reason"] for x in s["failed"]} >= {"stale or empty data"}


def test_coverage_stats_math():
    o = pd.DataFrame({"actual_date": ["d1", "d1", "d2", "d2"], "ret_pct": [1.0, -2.0, 0.5, 3.0],
                      "inside50": [True, False, True, False], "inside80": [True, True, True, False],
                      "inside95": [True, True, True, True], "width80_pct": [4.0, 4.0, 6.0, 6.0]})
    c = coverage_stats(o)
    assert c["n"] == 4 and c["inside50_pct"] == 50 and c["inside80_pct"] == 75 and c["inside95_pct"] == 100
    assert c["mean_width80_pct"] == 5 and c["no_change_mape_pct"] == pytest.approx(1.625)
