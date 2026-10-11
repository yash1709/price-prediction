"""Market-wide features (Nifty 50, India VIX, optionally lagged US-hours series) on the Indian
trading calendar. Shared by the research scripts and the live forecast-range code so the feature
definitions cannot drift apart."""
from __future__ import annotations

import logging
import time
from pathlib import Path

import pandas as pd

log = logging.getLogger(__name__)
LIVE_TICKERS = {"nifty": "^NSEI", "vix": "^INDIAVIX"}


def market_features(mkt: pd.DataFrame, dates: pd.DatetimeIndex) -> pd.DataFrame:
    """Causal market features on `dates`: row t uses only observations that are known at t."""
    f = pd.DataFrame(index=dates)
    if "nifty" in mkt:
        n = mkt["nifty"].dropna()
        r = n.pct_change()
        g = pd.DataFrame({"nifty_ret_1": r, "nifty_mom_5": n / n.shift(5) - 1,
                          "nifty_mom_20": n / n.shift(20) - 1, "nifty_vol_20": r.rolling(20).std(),
                          "nifty_to_sma_50": n / n.rolling(50).mean() - 1})
        f = f.join(g.reindex(dates, method="ffill"))  # Indian close: known at t
    if "vix" in mkt:
        v = mkt["vix"].dropna()
        g = pd.DataFrame({"vix": v, "vix_chg_1": v.pct_change(), "vix_chg_5": v.pct_change(5)})
        f = f.join(g.reindex(dates, method="ffill"))
    for name in ("usdinr", "crude"):  # US-hours series: only values dated BEFORE t are known at t
        if name in mkt:
            s = mkt[name].dropna()
            g = pd.DataFrame({f"{name}_ret_1": s.pct_change(), f"{name}_mom_5": s / s.shift(5) - 1}).shift(1)
            f = f.join(g.reindex(dates, method="ffill"))
    return f


def fetch_market_closes(cache_dir: Path, ttl_hours: float = 6.0) -> pd.DataFrame | None:
    """Daily closes for Nifty 50 and India VIX (Yahoo Finance), cached on disk. Returns None if
    either series is unavailable (callers then fall back to a model that needs no market data)."""
    import yfinance as yf

    cache_dir.mkdir(parents=True, exist_ok=True)
    out = {}
    for name, ticker in LIVE_TICKERS.items():
        f = cache_dir / f"live_{name}.csv"
        fresh = f.exists() and (time.time() - f.stat().st_mtime) < ttl_hours * 3600
        s = None
        if not fresh:
            try:
                h = yf.Ticker(ticker).history(period="2y", interval="1d", auto_adjust=False)
                if not h.empty:
                    idx = pd.DatetimeIndex(h.index)
                    s = pd.Series(h["Close"].values, index=(idx.tz_localize(None) if idx.tz else idx).normalize(),
                                  name=name)
                    s = s[~s.index.duplicated(keep="last")].sort_index()
                    s.to_csv(f)
            except Exception as exc:  # noqa: BLE001 - logged; stale cache or None is the fallback
                log.warning("Market data fetch failed for %s: %s", ticker, exc)
        if s is None and f.exists():
            s = pd.read_csv(f, index_col=0, parse_dates=True).iloc[:, 0]
        if s is None or s.empty:
            return None
        out[name] = s
    return pd.DataFrame(out)
