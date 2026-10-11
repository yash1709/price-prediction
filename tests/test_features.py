"""Feature calculations, target shifting and the no-future-information guarantees."""
import numpy as np
import pandas as pd
import pytest

from app.core.errors import InsufficientDataError
from app.features.engineering import (
    FEATURE_NAMES, LOOKBACK, build_dataset, compute_features, compute_target, latest_features,
)
from tests.conftest import make_prices


def test_sma_and_return_features_match_manual_calculation(prices):
    f = compute_features(prices)
    c = prices["adj_close"]
    t = 100
    assert f["close_to_sma_20"].iloc[t] == pytest.approx(c.iloc[t] / c.iloc[t - 19:t + 1].mean() - 1)
    assert f["close_to_sma_5"].iloc[t] == pytest.approx(c.iloc[t] / c.iloc[t - 4:t + 1].mean() - 1)
    assert f["ret_lag_1"].iloc[t] == pytest.approx(c.iloc[t] / c.iloc[t - 1] - 1)
    assert f["ret_lag_3"].iloc[t] == pytest.approx(c.iloc[t - 2] / c.iloc[t - 3] - 1)
    assert f["mom_10"].iloc[t] == pytest.approx(c.iloc[t] / c.iloc[t - 10] - 1)
    assert f["hl_range"].iloc[t] == pytest.approx((prices["high"].iloc[t] - prices["low"].iloc[t]) / c.iloc[t])
    assert f["gap_open"].iloc[t] == pytest.approx(prices["open"].iloc[t] / c.iloc[t - 1] - 1)
    r = c.pct_change()
    assert f["vol_10"].iloc[t] == pytest.approx(r.iloc[t - 9:t + 1].std())


def test_rsi_bounds_and_extremes(prices):
    f = compute_features(prices)
    rsi = f["rsi_14"].dropna()
    assert ((rsi >= 0) & (rsi <= 100)).all()
    up = prices.copy()
    up["adj_close"] = up["close"] = np.linspace(100, 200, len(up))
    up["open"] = up["high"] = up["low"] = up["close"]
    assert compute_features(up)["rsi_14"].dropna().iloc[-1] == pytest.approx(100.0)


def test_macd_matches_ewm_definition(prices):
    f = compute_features(prices)
    c = prices["adj_close"]
    macd = c.ewm(span=12, adjust=False).mean() - c.ewm(span=26, adjust=False).mean()
    assert f["macd_norm"].iloc[200] == pytest.approx(macd.iloc[200] / c.iloc[200])
    sig = macd.ewm(span=9, adjust=False).mean()
    assert f["macd_hist_norm"].iloc[200] == pytest.approx((macd.iloc[200] - sig.iloc[200]) / c.iloc[200])


def test_features_have_warmup_nans_then_complete(prices):
    f = compute_features(prices)
    assert list(f.columns) == FEATURE_NAMES
    assert f.iloc[LOOKBACK:].notna().all().all()
    assert f.iloc[:10].isna().any(axis=1).all()


@pytest.mark.parametrize("h", [1, 5, 10])
def test_target_is_future_return_and_last_rows_unknown(prices, h):
    y = compute_target(prices, h)
    c = prices["adj_close"]
    assert y.iloc[50] == pytest.approx(c.iloc[50 + h] / c.iloc[50] - 1)
    assert y.iloc[-h:].isna().all() and y.iloc[:-h].notna().all()


def test_build_dataset_excludes_rows_without_known_target_and_aligns_dates(prices):
    ds = build_dataset(prices, 5)
    assert ds.X.index[-1] == prices.index[-6]               # last 5 rows have no target
    row = ds.X.index[10]
    pos = prices.index.get_loc(row)
    assert ds.target_date.iloc[10] == prices.index[pos + 5]
    assert ds.actual_price.iloc[10] == pytest.approx(prices["adj_close"].iloc[pos + 5])
    assert ds.X.notna().all().all() and ds.y.notna().all()


def test_features_never_use_future_data(prices):
    """Row t of the features computed on the full history must equal row t computed on the
    history truncated at t -> no feature can depend on anything after t."""
    full = compute_features(prices)
    for t in (80, 200, 400, 650):
        trunc = compute_features(prices.iloc[: t + 1])
        pd.testing.assert_series_equal(full.iloc[t], trunc.iloc[t], check_names=False, rtol=1e-9)


def test_perturbing_the_future_does_not_change_past_features(prices):
    t = 300
    altered = prices.copy()
    altered.iloc[t + 1:, :] = altered.iloc[t + 1:, :] * 3.7   # wildly different future
    a, b = compute_features(prices), compute_features(altered)
    pd.testing.assert_frame_equal(a.iloc[: t + 1], b.iloc[: t + 1])


def test_latest_features_use_last_completed_session(prices):
    ts, row = latest_features(prices)
    assert ts == prices.index[-1] and list(row.columns) == FEATURE_NAMES
    pd.testing.assert_series_equal(row.iloc[0], compute_features(prices).iloc[-1], check_names=False)


def test_insufficient_history_raises():
    with pytest.raises(InsufficientDataError):
        build_dataset(make_prices(40), 1)
    with pytest.raises(InsufficientDataError):
        latest_features(make_prices(40))


def test_split_adjustment_keeps_returns_continuous():
    df = make_prices(120)
    split = df.copy()
    split.loc[split.index[60:], ["open", "high", "low", "close"]] *= 0.5   # 2:1 split, unadjusted
    split.loc[split.index[:60], ["open", "high", "low", "close"]] *= 1.0
    split["adj_close"] = df["close"] * 0.5                                  # fully adjusted series
    split.loc[split.index[:60], "close"] = df["close"].iloc[:60]            # raw close before split
    f = compute_features(split)
    assert f["ret_lag_1"].abs().max() < 0.1   # no -50% artefact from the raw price jump
