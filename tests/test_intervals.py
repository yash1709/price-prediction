"""Forecast ranges: causality, structure, fallback and calibration sanity (no network)."""
import numpy as np
import pandas as pd
import pytest

from app.features.engineering import FEATURE_NAMES
from app.models.intervals import (
    IntervalService, _meta, feature_row, supported_horizons, vol_series,
)
from tests.conftest import make_market, make_prices


def _svc(market):
    from app.core.config import Settings
    return IntervalService(Settings(_env_file=None), lambda: market)


def test_supported_horizons_and_artifacts_present():
    assert supported_horizons() == [1, 5]
    meta = _meta()
    for h in ("1", "5"):
        info = meta["horizons"][h]
        assert set(info["z_quantiles"]) >= {"fhs_xgb", "fhs_ewma"}
        assert len(info["features"]) == 35 and info["features"][: len(FEATURE_NAMES)] == FEATURE_NAMES


def test_volatility_features_use_no_future_data(prices):
    full = vol_series(prices)
    for t in (100, 300, 600):
        pd.testing.assert_series_equal(full.iloc[t], vol_series(prices.iloc[: t + 1]).iloc[-1], check_names=False)


def test_model_features_ignore_market_data_after_the_forecast_date(prices, market):
    info = _meta()["horizons"]["1"]
    t = 400
    df = prices.iloc[: t + 1]
    a = feature_row(df, market, info["features"])
    b = feature_row(df, market.iloc[: t + 1], info["features"])          # market truncated at t
    c = feature_row(df, market.assign(nifty=lambda m: np.where(m.index > df.index[-1], 1.0, m["nifty"]),
                                      vix=lambda m: np.where(m.index > df.index[-1], 99.0, m["vix"])),
                    info["features"])                                  # absurd future market values
    pd.testing.assert_frame_equal(a, b)
    pd.testing.assert_frame_equal(a, c)


def test_ranges_are_nested_ordered_and_scale_with_horizon(prices, market):
    svc = _svc(market)
    r1, r5 = svc.compute(prices, 1), svc.compute(prices, 5)
    for r in (r1, r5):
        assert r.method == "fhs_xgb" and r.sigma_daily > 0
        (l50, h50), (l80, h80), (l95, h95) = r.ranges[0.5], r.ranges[0.8], r.ranges[0.95]
        assert l95 < l80 < l50 < 0 < h50 < h80 < h95
        assert l95 >= -0.95
    assert (r5.ranges[0.8][1] - r5.ranges[0.8][0]) > (r1.ranges[0.8][1] - r1.ranges[0.8][0])


def test_small_cap_band_uses_its_own_calibration(prices, market):
    from tests.test_tiers import SMALL_SYM, SMALL_SYM_B2
    meta, svc = _meta(), _svc(market)
    for h in (1, 5):
        base = svc.compute(prices, h)                                       # Nifty-200 calibration
        for sym, band in ((SMALL_SYM, "r501_1000"), (SMALL_SYM_B2, "r1001_1500")):
            res = svc.compute(prices, h, symbol=sym)
            z = meta["bands"][band][str(h)]["z_quantiles"]["fhs_xgb"]["0.8"]
            assert res.sigma_daily == pytest.approx(base.sigma_daily)       # same volatility model ...
            lo, hi = res.ranges[0.8]
            assert lo == pytest.approx(z[0] * res.sigma_daily * np.sqrt(h))  # ... different quantiles
            assert hi == pytest.approx(z[1] * res.sigma_daily * np.sqrt(h))
            assert (hi - lo) > (base.ranges[0.8][1] - base.ranges[0.8][0])   # small caps get wider ranges
        assert svc.compute(prices, h, symbol="RELIANCE").ranges == base.ranges   # Nifty 200 unchanged
        assert svc.compute(prices, h, symbol="ZZZNOPE").ranges == base.ranges    # unknown symbol unchanged
    assert set(meta["bands"]) == {"r501_1000", "r1001_1500"}
    assert _svc(None).compute(prices, 5, symbol=SMALL_SYM).method == "fhs_ewma"  # fallback also band-calibrated


def test_fallback_to_ewma_without_or_with_stale_market(prices, market):
    assert _svc(None).compute(prices, 1).method == "fhs_ewma"
    stale = market.iloc[:-40]                                   # market feed ends ~2 months early
    assert _svc(stale).compute(prices, 1).method == "fhs_ewma"

    def boom():
        raise RuntimeError("network down")
    from app.core.config import Settings
    assert IntervalService(Settings(_env_file=None), boom).compute(prices, 5).method == "fhs_ewma"


def test_unsupported_horizon_or_short_history_gives_no_range(prices, market):
    svc = _svc(market)
    assert svc.compute(prices, 3) is None
    assert svc.compute(prices.iloc[:40], 1) is None


def test_ewma_fallback_covers_close_to_nominal_on_simulated_returns():
    """Gaussian returns with slowly changing volatility: the 80 % range should cover ~80 %."""
    rng = np.random.default_rng(7)
    n = 4000
    vol = 0.012 * np.exp(0.5 * np.sin(np.arange(n) / 150))
    ret = rng.normal(0, 1, n) * vol
    idx = pd.bdate_range(end="2024-06-28", periods=n)
    adj = pd.Series(1000 * np.exp(np.cumsum(ret)), index=idx)
    df = pd.DataFrame({"adj_close": adj})
    ewma = vol_series(df)["ewma"]
    zlo, zhi = _meta()["horizons"]["1"]["z_quantiles"]["fhs_ewma"]["0.8"]
    nxt = adj.pct_change().shift(-1)
    ok = ewma.notna() & nxt.notna() & (np.arange(n) > 300)
    inside = ((nxt[ok] >= zlo * ewma[ok]) & (nxt[ok] <= zhi * ewma[ok])).mean()
    assert 0.74 <= inside <= 0.86
