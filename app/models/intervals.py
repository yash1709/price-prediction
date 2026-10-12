"""Forecast ranges (prediction intervals) for the h-day return, centred on the latest close.

Method (selected by the pre-registered rule in scripts/interval_experiment.py, validated
out-of-sample on ~211k stock-days: 80 % ranges covered 79.9 % of outcomes):
  * `fhs_xgb`  : a pooled XGBoost model predicts the stock's daily volatility from its own features and
                 Nifty/VIX features; the range is  sigma * sqrt(h) * z  where z are empirical quantiles
                 of (realised return / predicted sigma) on a held-out calibration window.
  * `fhs_ewma` : fallback when market data is unavailable: EWMA (lambda 0.94) volatility with the same
                 kind of empirical calibration (also passed the coverage criteria).
Calibrated for horizons of 1 and 5 trading days only; other horizons get no range.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from statistics import NormalDist
from typing import Callable

import numpy as np
import pandas as pd

from app.core.config import Settings
from app.features.engineering import compute_features
from app.features.market import fetch_market_closes, market_features

log = logging.getLogger(__name__)
PRETRAINED = Path(__file__).with_name("pretrained")
LEVELS = (0.5, 0.8, 0.95)
MAX_MARKET_LAG_DAYS = 7
MIN_LOG_RETURN = -0.95  # a lower bound can never go below -95 % of the reference price


Z80 = NormalDist().inv_cdf(0.9)  # 1.2816: half-width of a normal 80 % interval in sigmas


def implied_daily_vol(lower_ret: float, upper_ret: float, horizon: int) -> float:
    """Normal-equivalent DAILY volatility implied by a calibrated 80 % range for `horizon` days.

    Use this (not `sigma_daily`) wherever a volatility figure is shown: the raw model output is on
    an arbitrary scale that the calibration step corrects (z multipliers ~2.3), so on its own it is
    NOT a volatility estimate."""
    return (upper_ret - lower_ret) / (2 * Z80) / float(np.sqrt(horizon))


@dataclass
class IntervalResult:
    method: str
    horizon: int
    sigma_daily: float                         # raw model scale (uncalibrated); do not display
    ranges: dict[float, tuple[float, float]]   # level -> (lower return, upper return)
    calibration_window: list[str]

    @property
    def implied_daily_vol(self) -> float:
        lo, hi = self.ranges[0.8]
        return implied_daily_vol(lo, hi, self.horizon)


@lru_cache
def _meta() -> dict:
    return json.loads((PRETRAINED / "interval_calibration.json").read_text(encoding="utf-8"))


@lru_cache
def _model(h: int):
    from xgboost import XGBRegressor
    m = XGBRegressor()
    m.load_model(str(PRETRAINED / f"vol_model_h{h}.json"))
    return m


def supported_horizons() -> list[int]:
    return sorted(int(h) for h in _meta()["horizons"])


def vol_series(df: pd.DataFrame) -> pd.DataFrame:
    """Causal volatility estimates per row (adjusted returns): EWMA, 60-session std, long-run RMS."""
    r = df["adj_close"].pct_change(fill_method=None)
    r2 = r ** 2
    return pd.DataFrame({
        "ewma": r2.ewm(alpha=0.06, adjust=False, min_periods=30).mean() ** 0.5,
        "roll60": r.rolling(60, min_periods=40).std(),
        "lr252": r2.rolling(252, min_periods=120).mean() ** 0.5,
    })


def feature_row(df: pd.DataFrame, market: pd.DataFrame | None, features: list[str]) -> pd.DataFrame | None:
    """Model input for the latest bar, or None if the market features cannot be supplied."""
    f = compute_features(df)
    v = vol_series(df)
    for k in ("ewma", "roll60", "lr252"):
        f[f"log_{k}"] = np.log(np.clip(v[k], 1e-4, None))
    if market is not None:
        mf = market_features(market, df.index)
        f = f.join(mf)
    last = f.iloc[[-1]]
    if any(c not in last.columns for c in features):
        return None
    return last[features].astype("float32")


class IntervalService:
    """Computes ranges for a price history. `market_fetcher` returns a DataFrame with `nifty` and
    `vix` closes (or None); injectable so tests never touch the network."""

    def __init__(self, settings: Settings,
                 market_fetcher: Callable[[], pd.DataFrame | None] | None = None):
        self._fetch = market_fetcher or (lambda: fetch_market_closes(settings.data_path / "market"))

    def compute(self, df: pd.DataFrame, horizon: int) -> IntervalResult | None:
        if horizon not in supported_horizons() or len(df) < 60:
            return None
        info = _meta()["horizons"][str(horizon)]
        v = vol_series(df).iloc[-1]
        if not np.isfinite(v["ewma"]) or v["ewma"] <= 0:
            return None
        market = self._market_for(df)
        row = feature_row(df, market, info["features"]) if market is not None else None
        sq = np.sqrt(horizon)
        if row is not None:
            pred = float(_model(horizon).predict(row)[0])
            sigma, method = float(np.sqrt(np.exp(pred))), "fhs_xgb"
        else:
            sigma, method = float(v["ewma"]), "fhs_ewma"
        z = info["z_quantiles"][method]
        ranges = {}
        for lv in LEVELS:
            zlo, zhi = z[str(lv)]
            ranges[lv] = (max(zlo * sigma * sq, MIN_LOG_RETURN), zhi * sigma * sq)
        return IntervalResult(method, horizon, sigma, ranges, info["calibration_window"])

    def market_for(self, df: pd.DataFrame) -> pd.DataFrame | None:
        """Nifty/VIX closes if available and recent enough for `df`, else None."""
        return self._market_for(df)

    def _market_for(self, df: pd.DataFrame) -> pd.DataFrame | None:
        try:
            m = self._fetch()
        except Exception as exc:  # noqa: BLE001 - fall back to the EWMA method
            log.warning("market data unavailable: %s", exc)
            return None
        if m is None or m.empty or "nifty" not in m or "vix" not in m:
            return None
        if (df.index[-1] - m.dropna().index[-1]).days > MAX_MARKET_LAG_DAYS:
            return None
        return m
