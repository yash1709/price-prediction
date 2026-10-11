"""Leakage-safe feature engineering and target construction.

Definitions (documented in README):
  * Observation at trading day t uses ONLY bars dated <= t (rolling/ewm windows are backward-looking).
  * Prices are split/dividend adjusted (`adj_close`; open/high/low scaled by the same factor) so that
    returns are continuous across corporate actions.
  * Target for horizon h: relative return  y_t = P[t+h] / P[t] - 1  (P = adjusted close).
    The next-trading-day close forecast is  P[t] * (1 + y_hat).  Models predict a return because it
    is (roughly) stationary and scale-free; predicting price levels makes tree models unable to
    extrapolate and makes linear models mostly copy the last price.
  * Lookback: the longest window is 50 sessions (SMA50), so the first 50 rows have no features.
  * Features are scale-free (ratios/returns) so they transfer across price levels.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from app.core.errors import InsufficientDataError

LOOKBACK = 50  # sessions needed before the first valid feature row

FEATURE_NAMES: list[str] = (
    [f"ret_lag_{k}" for k in range(1, 6)]
    + [f"mom_{k}" for k in (5, 10, 20)]
    + [f"close_to_sma_{k}" for k in (5, 10, 20, 50)]
    + ["sma_5_20_ratio"]
    + [f"vol_{k}" for k in (5, 10, 20)]
    + ["rsi_14", "macd_norm", "macd_signal_norm", "macd_hist_norm",
       "hl_range", "gap_open", "volume_chg", "volume_ratio_20"]
)


def adjusted_ohlc(df: pd.DataFrame) -> pd.DataFrame:
    """Return open/high/low/close/volume on an adjusted basis (factor = adj_close / close)."""
    factor = (df["adj_close"] / df["close"]).where(df["close"] > 0)
    out = pd.DataFrame(index=df.index)
    for col in ("open", "high", "low"):
        out[col] = df[col] * factor
    out["close"] = df["adj_close"]
    out["volume"] = df["volume"]
    return out


def _rsi(close: pd.Series, n: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / n, adjust=False, min_periods=n).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / n, adjust=False, min_periods=n).mean()
    rs = gain / loss.replace(0, np.nan)
    rsi = 100 - 100 / (1 + rs)
    return rsi.where(loss != 0, 100.0).where(gain.notna() & loss.notna())


def compute_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute FEATURE_NAMES for every row. Row t depends only on rows <= t (causal)."""
    a = adjusted_ohlc(df)
    c, v = a["close"], a["volume"]
    ret = c.pct_change()
    f = pd.DataFrame(index=df.index)
    for k in range(1, 6):
        f[f"ret_lag_{k}"] = ret.shift(k - 1)  # ret_lag_1 = today's return (known at close of t)
    for k in (5, 10, 20):
        f[f"mom_{k}"] = c / c.shift(k) - 1
    smas = {k: c.rolling(k, min_periods=k).mean() for k in (5, 10, 20, 50)}
    for k, s in smas.items():
        f[f"close_to_sma_{k}"] = c / s - 1
    f["sma_5_20_ratio"] = smas[5] / smas[20] - 1
    for k in (5, 10, 20):
        f[f"vol_{k}"] = ret.rolling(k, min_periods=k).std()
    f["rsi_14"] = _rsi(c, 14)
    ema12 = c.ewm(span=12, adjust=False, min_periods=12).mean()
    ema26 = c.ewm(span=26, adjust=False, min_periods=26).mean()
    macd = ema12 - ema26
    signal = macd.ewm(span=9, adjust=False, min_periods=9).mean()
    f["macd_norm"] = macd / c
    f["macd_signal_norm"] = signal / c
    f["macd_hist_norm"] = (macd - signal) / c
    f["hl_range"] = (a["high"] - a["low"]) / c
    f["gap_open"] = a["open"] / c.shift(1) - 1
    vol = v.where(v > 0)
    f["volume_chg"] = (vol / vol.shift(1) - 1).clip(-5, 5)
    f["volume_ratio_20"] = (vol / vol.rolling(20, min_periods=10).mean() - 1).clip(-5, 5)
    return f.replace([np.inf, -np.inf], np.nan)[FEATURE_NAMES]


def compute_target(df: pd.DataFrame, horizon: int) -> pd.Series:
    """y_t = P[t+horizon] / P[t] - 1; the last `horizon` rows are NaN (future unknown)."""
    if horizon < 1:
        raise ValueError("horizon must be >= 1")
    c = df["adj_close"]
    return (c.shift(-horizon) / c - 1).rename(f"target_ret_h{horizon}")


@dataclass
class Dataset:
    X: pd.DataFrame            # features, index = as-of date t
    y: pd.Series               # target return for horizon
    ref_price: pd.Series       # adjusted close at t
    target_date: pd.Series     # date of bar t+h
    horizon: int

    @property
    def actual_price(self) -> pd.Series:
        return self.ref_price * (1 + self.y)


def build_dataset(df: pd.DataFrame, horizon: int = 1) -> Dataset:
    """Training dataset: rows with complete features AND a known target, chronologically ordered."""
    if not df.index.is_monotonic_increasing:
        raise ValueError("Price frame must be sorted by date ascending.")
    X = compute_features(df)
    y = compute_target(df, horizon)
    c = df["adj_close"]
    target_date = pd.Series(df.index, index=df.index).shift(-horizon)
    valid = X.notna().all(axis=1) & y.notna()
    if valid.sum() == 0:
        raise InsufficientDataError(
            f"Not enough history to build features (need > {LOOKBACK + horizon} sessions, "
            f"have {len(df)}).")
    return Dataset(X[valid], y[valid], c[valid], target_date[valid], horizon)


def latest_features(df: pd.DataFrame) -> tuple[pd.Timestamp, pd.DataFrame]:
    """Feature row for the most recent completed session (target unknown by construction)."""
    X = compute_features(df)
    last = X.iloc[[-1]]
    if last.isna().any(axis=1).iloc[0]:
        raise InsufficientDataError(
            f"Latest feature row is incomplete (need > {LOOKBACK} sessions of clean history).")
    return last.index[0], last
