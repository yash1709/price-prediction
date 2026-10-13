"""Tests for the calibration export scripts and the artifacts they ship.

`scripts/export_vol_models.py` is run end to end on tiny synthetic inputs in a temp folder.
`scripts/export_band_calibration.py` needs the cached prices of ~800 stocks, so it is not run here;
instead the artifacts it writes (`smallcap_ranks.csv`, the `bands` section of the calibration JSON)
are checked for internal consistency and for being usable by `IntervalService`."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest
from xgboost import XGBRegressor

from app.core.config import Settings
from app.models.intervals import LEVELS, PRETRAINED, IntervalService, supported_horizons
from tests.conftest import make_prices

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "app" / "data"
CAL = json.loads((PRETRAINED / "interval_calibration.json").read_text(encoding="utf-8"))
RANKS = pd.read_csv(DATA / "smallcap_ranks.csv")
LV = [str(lv) for lv in LEVELS]
BAND_RANK_RANGE = {"r501_1000": (1, 500), "r1001_1500": (501, 1000)}  # same as interval_tail_check.BANDS


def _load_script(name: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _z(scale: float = 1.0) -> dict:
    return {"0.5": [-0.7 * scale, 0.7 * scale], "0.8": [-1.3 * scale, 1.3 * scale],
            "0.95": [-2.1 * scale, 2.1 * scale]}


# ----------------------------------------------------------------- export_vol_models.py
@pytest.fixture
def export_env(tmp_path, monkeypatch):
    """Fake experiment outputs for horizon 1 (a tiny real XGBoost model) and a patched script."""
    reports, out = tmp_path / "reports", tmp_path / "pretrained"
    reports.mkdir()
    rng = np.random.default_rng(0)
    features = ["f_a", "f_b", "log_ewma"]
    X = pd.DataFrame(rng.normal(size=(60, 3)), columns=features).astype("float32")
    model = XGBRegressor(n_estimators=3, max_depth=2).fit(X, rng.normal(size=60))
    joblib.dump({"model": model, "features": features, "eps": 1e-6}, reports / "vol_model_h1_experiment.joblib")
    (reports / "interval_calibration_candidates.json").write_text(json.dumps({
        "created": "2026-01-01", "levels": [0.5, 0.8, 0.95],
        "horizons": {"1": {"calibration_window": ["2025-01-01", "2025-12-31"],
                           "z_quantiles": {"fhs_xgb": _z(), "fhs_ewma": _z(1.1)}}}}), encoding="utf-8")
    cov = {lv: {"coverage_pct": float(lv) * 100 - 0.1} for lv in LV}
    (reports / "interval_experiment.json").write_text(json.dumps({
        "oos_start": "2022-01-01", "oos_end": "2026-01-01", "stocks": 190, "selection": {"1": "fhs_xgb"},
        "horizons": {"1": {"methods": {"fhs_xgb": cov, "fhs_ewma": cov}}}}), encoding="utf-8")
    script = _load_script("export_vol_models")
    monkeypatch.setattr(script, "REPORTS", reports)
    monkeypatch.setattr(script, "OUT", out)
    return script, out, model, X, features


def test_export_vol_models_writes_a_loadable_model_and_calibration(export_env, capsys):
    script, out, model, X, features = export_env
    script.main()
    meta = json.loads((out / "interval_calibration.json").read_text(encoding="utf-8"))
    assert set(meta["horizons"]) == {"1"}
    h1 = meta["horizons"]["1"]
    assert h1["features"] == features and h1["eps"] == 1e-6
    assert h1["calibration_window"] == ["2025-01-01", "2025-12-31"]
    assert h1["z_quantiles"]["fhs_xgb"] == _z()
    assert meta["stocks"] == 190 and meta["oos_start"] == "2022-01-01" and meta["levels"] == [0.5, 0.8, 0.95]
    assert h1["validation"]["fhs_xgb"]["0.8"] == {"coverage_pct": pytest.approx(79.9)}
    reloaded = XGBRegressor()
    reloaded.load_model(str(out / "vol_model_h1.json"))
    np.testing.assert_allclose(reloaded.predict(X[features]), model.predict(X[features]), rtol=1e-5)
    assert "exported model" in capsys.readouterr().out


def test_export_vol_models_fails_loudly_when_a_model_file_is_missing(export_env):
    script, out, *_ = export_env
    (script.REPORTS / "vol_model_h1_experiment.joblib").unlink()
    with pytest.raises(FileNotFoundError):
        script.main()
    assert not (out / "interval_calibration.json").exists()  # no half-written calibration


# --------------------------------------------- shipped calibration (written by both scripts)
def test_shipped_calibration_covers_the_supported_horizons_with_loadable_models():
    assert set(CAL["horizons"]) == {"1", "5"}
    assert supported_horizons() == [1, 5]
    for h, info in CAL["horizons"].items():
        model = XGBRegressor()
        model.load_model(str(PRETRAINED / f"vol_model_h{h}.json"))
        assert model.get_booster().num_features() == len(info["features"]), h
        assert len(set(info["features"])) == len(info["features"]), f"duplicate features at h={h}"


def _assert_valid_quantiles(zq: dict, where: str) -> None:
    for method in ("fhs_xgb", "fhs_ewma"):
        z = zq[method]
        assert set(z) == set(LV), f"{where}/{method}"
        for lv in LV:
            lo, hi = z[lv]
            assert lo < 0 < hi, f"{where}/{method}/{lv}: range must straddle zero"
        # higher confidence levels must give strictly wider, nested ranges
        assert z["0.95"][0] < z["0.8"][0] < z["0.5"][0], f"{where}/{method} lower bounds not nested"
        assert z["0.5"][1] < z["0.8"][1] < z["0.95"][1], f"{where}/{method} upper bounds not nested"


def test_shipped_z_quantiles_are_valid_for_nifty200_and_every_band():
    for h, info in CAL["horizons"].items():
        _assert_valid_quantiles(info["z_quantiles"], f"horizons/{h}")
    assert set(CAL["bands"]) == set(BAND_RANK_RANGE)
    for band, by_h in CAL["bands"].items():
        assert set(by_h) == set(CAL["horizons"]), band
        for h, entry in by_h.items():
            _assert_valid_quantiles(entry["z_quantiles"], f"bands/{band}/{h}")
            assert entry["n_rows"] > 0 and entry["n_stocks"] > 0
            assert entry["calibration_window"] == CAL["horizons"][h]["calibration_window"]


def test_band_ranges_are_wider_than_nifty200_as_the_readme_states():
    for band, by_h in CAL["bands"].items():
        for h, entry in by_h.items():
            n200 = CAL["horizons"][h]["z_quantiles"]["fhs_xgb"]["0.8"]
            band80 = entry["z_quantiles"]["fhs_xgb"]["0.8"]
            assert band80[1] - band80[0] > n200[1] - n200[0], f"{band} h={h}"


# --------------------------------------------------------------------- smallcap_ranks.csv
def test_smallcap_ranks_file_is_well_formed():
    assert list(RANKS.columns) == ["symbol", "band", "rank_outside_nifty500", "market_cap_inr", "as_of"]
    assert RANKS["symbol"].is_unique and not RANKS.isna().any().any()
    assert set(RANKS["band"]) == set(BAND_RANK_RANGE)
    assert (RANKS["market_cap_inr"] > 0).all()
    for band, (lo, hi) in BAND_RANK_RANGE.items():
        ranks = RANKS.loc[RANKS["band"] == band, "rank_outside_nifty500"]
        assert ranks.between(lo, hi).all(), f"{band}: rank outside {lo}-{hi}"
        assert ranks.is_unique, f"{band}: duplicate ranks"


def test_smallcap_ranks_never_include_nifty500_stocks():
    n500 = set(pd.read_csv(DATA / "nifty500.csv")["Symbol"].astype(str).str.upper())
    assert not (set(RANKS["symbol"].str.upper()) & n500)


def test_larger_market_cap_means_better_rank_across_the_whole_file():
    by_rank = RANKS.sort_values("rank_outside_nifty500")
    assert by_rank["market_cap_inr"].is_monotonic_decreasing


# -------------------------------------------------- the app picks the band calibration up
def test_interval_service_uses_the_band_calibration_for_ranked_stocks(tmp_path):
    settings = Settings(data_dir=tmp_path / "data", _env_file=None)
    svc = IntervalService(settings, market_fetcher=lambda: None)  # no market data -> fhs_ewma path
    df = make_prices(n=400)
    sym = RANKS.loc[RANKS["band"] == "r501_1000", "symbol"].iloc[0]
    banded = svc.compute(df, 1, symbol=sym)
    plain = svc.compute(df, 1)
    assert banded is not None and plain is not None
    assert banded.method == plain.method == "fhs_ewma"
    assert banded.sigma_daily == plain.sigma_daily  # same volatility estimate, different calibration
    band_z = CAL["bands"]["r501_1000"]["1"]["z_quantiles"]["fhs_ewma"]
    n200_z = CAL["horizons"]["1"]["z_quantiles"]["fhs_ewma"]
    for lv in LEVELS:
        assert banded.ranges[lv] == pytest.approx(tuple(z * banded.sigma_daily for z in band_z[str(lv)]))
        assert plain.ranges[lv] == pytest.approx(tuple(z * plain.sigma_daily for z in n200_z[str(lv)]))
    assert banded.ranges != plain.ranges
