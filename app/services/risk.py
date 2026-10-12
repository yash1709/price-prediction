"""Risk tools built only on validated components (no price prediction).

* stock_risk     : realised/forecast volatility, calibrated expected-move ranges (the validated
                   product), bad-day thresholds, drawdown, beta vs Nifty, 52-week position, liquidity.
* position_size  : largest position whose loss on a "bad day" stays within a risk budget.
* portfolio_risk : volatility, per-holding risk contribution, diversification and correlations.
                   NO calibrated portfolio ranges: that variant failed its pre-registered coverage test
                   at the 5-day horizon (scripts/portfolio_risk_experiment.py), so it is not offered.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd

from app.core.errors import AppError, InsufficientDataError
from app.core.symbols import normalize_symbol
from app.data.tiers import validation_status
from app.models.intervals import vol_series
from app.services.context import AppContext
from app.services.workflow import fetch_data

ANN = math.sqrt(252)
CORR_WINDOW = 250
SHRINK = 0.2
MIN_OVERLAP = 120
MAX_HOLDINGS = 25
TAILS = {"1_in_10": ("80", "about 1 day in 10 (observed 10.3% out-of-sample)"),
         "1_in_40": ("95", "about 1 day in 37 (observed 2.7% out-of-sample)")}


class InvalidInputError(AppError):
    code, status_code = "invalid_input", 422


def shrunk_corr(r: pd.DataFrame) -> np.ndarray:
    """Trailing correlation matrix shrunk 20% toward its average off-diagonal value."""
    c = r.corr(min_periods=MIN_OVERLAP).values
    n = c.shape[0]
    if n == 1:
        return np.ones((1, 1))
    off = c[~np.eye(n, dtype=bool)]
    avg = float(np.nanmean(off)) if np.isfinite(off).any() else 0.3
    c = np.where(np.isnan(c), avg, c)
    out = (1 - SHRINK) * c + SHRINK * avg
    np.fill_diagonal(out, 1.0)
    return out


def _load(ctx: AppContext, symbol: str, refresh: bool = True) -> pd.DataFrame:
    symbol = normalize_symbol(symbol)
    if refresh:
        fetch_data(ctx, symbol)
    with ctx.repo() as repo:
        df = repo.load_bars(symbol)
    if len(df) < 130:
        raise InsufficientDataError(f"{symbol} has only {len(df)} sessions of history; need at least 130.")
    return df


def _drawdown(adj: pd.Series) -> tuple[float, float]:
    """(maximum drawdown, current drawdown from the running peak) in percent."""
    peak = adj.cummax()
    dd = adj / peak - 1
    return float(dd.min() * 100), float(dd.iloc[-1] * 100)


def stock_risk(ctx: AppContext, symbol: str, refresh: bool = True) -> dict:
    symbol = normalize_symbol(symbol)
    df = _load(ctx, symbol, refresh)
    adj, close = df["adj_close"], df["close"]
    r = adj.pct_change(fill_method=None)
    ref = float(close.iloc[-1])
    vs = vol_series(df).iloc[-1]
    realised = {f"{n}d": float(r.tail(n).std() * ANN * 100) for n in (20, 60, 252) if len(r) > n}

    moves: dict[str, dict] = {}
    method = implied = None
    for h in (1, 5):
        res = ctx.intervals.compute(df, h, symbol=symbol)
        if res is None:
            continue
        method = res.method
        if h == 1:
            implied = res.implied_daily_vol   # calibration-consistent volatility (NOT the raw model sigma)
        moves[str(h)] = {str(int(lv * 100)): {
            "lower": ref * (1 + lo), "upper": ref * (1 + hi),
            "lower_pct": lo * 100, "upper_pct": hi * 100} for lv, (lo, hi) in res.ranges.items()}
    bad_days = {}
    if "1" in moves:
        bad_days = {k: {"threshold_pct": moves["1"][pct]["lower_pct"], "threshold_price": moves["1"][pct]["lower"],
                        "meaning": f"A 1-day fall worse than this happens {desc}."}
                    for k, (pct, desc) in TAILS.items()}

    last252 = adj.tail(252)
    mdd1, cur = _drawdown(last252)
    mdd3, _ = _drawdown(adj.tail(756))
    hi52, lo52 = float(close.tail(252).max()), float(close.tail(252).min())
    beta = None
    market = ctx.intervals.market_for(df)
    if market is not None:
        m = market["nifty"].dropna().pct_change().reindex(df.index).tail(252)
        pair = pd.concat([r.tail(252), m], axis=1).dropna()
        if len(pair) >= MIN_OVERLAP and pair.iloc[:, 1].var() > 0:
            beta = float(pair.cov().iloc[0, 1] / pair.iloc[:, 1].var())
    traded = float((close * df["volume"]).tail(20).mean())
    with ctx.repo() as repo:
        stock = repo.get_stock(symbol)
        series = stock.series if stock else None
    return {
        "symbol": symbol, "as_of_date": str(df.index[-1].date()), "last_close": ref,
        "validation": validation_status(symbol, series),
        "volatility": {
            "forecast_daily_pct": None if implied is None else implied * 100,
            "forecast_annualised_pct": None if implied is None else implied * ANN * 100,
            "forecast_method": method, "ewma_daily_pct": float(vs["ewma"] * 100),
            "realised_annualised_pct": realised,
            "note": "Forecast volatility is the normal-equivalent figure implied by the calibrated 1-day "
                    "80% range (width / 2.563), so it is consistent with the ranges shown. Heavy-tailed "
                    "days make the true standard deviation somewhat larger.",
        },
        "expected_moves": moves, "bad_day_thresholds": bad_days,
        "drawdown": {"current_from_1y_peak_pct": cur, "max_1y_pct": mdd1, "max_3y_pct": mdd3},
        "range_52w": {"high": hi52, "low": lo52,
                      "position_pct": float((ref - lo52) / (hi52 - lo52) * 100) if hi52 > lo52 else None},
        "extremes_1y": {"worst_day_pct": float(r.tail(252).min() * 100), "best_day_pct": float(r.tail(252).max() * 100)},
        "beta_vs_nifty_1y": beta,
        "avg_daily_traded_value_inr": traded,
        "notes": ["Ranges describe how far the price may move, not which direction.",
                  "Ranges were validated for 1- and 5-day horizons; they cannot foresee gaps from "
                  "unexpected news beyond what volatility history implies."],
    }


def position_size(risk: dict, portfolio_value: float, risk_pct: float, horizon: int = 1,
                  tail: str = "1_in_10") -> dict:
    """Largest position whose loss, if the stock hits its `tail` bad-move level, equals risk_pct of
    the portfolio. Position <= portfolio (no leverage)."""
    if portfolio_value <= 0 or not (0 < risk_pct <= 100):
        raise InvalidInputError("portfolio_value must be > 0 and risk_pct in (0, 100].")
    if tail not in TAILS:
        raise InvalidInputError(f"tail must be one of {sorted(TAILS)}.")
    moves = risk["expected_moves"].get(str(horizon))
    if not moves:
        raise InvalidInputError(f"No calibrated ranges for horizon {horizon}; use 1 or 5.")
    pct_key, desc = TAILS[tail]
    loss_frac = float(abs(moves[pct_key]["lower_pct"]) / 100)
    ref = float(risk["last_close"])
    budget = portfolio_value * risk_pct / 100
    position = min(budget / loss_frac, portfolio_value)
    shares = int(position // ref)
    value = shares * ref
    return {
        "symbol": risk["symbol"], "horizon_days": horizon, "tail": tail,
        "bad_move_pct": -loss_frac * 100, "bad_move_frequency": desc,
        "risk_budget_inr": budget, "max_position_inr": position, "max_position_pct_of_portfolio": position / portfolio_value * 100,
        "shares": shares, "position_value_inr": value,
        "loss_if_bad_move_inr": value * loss_frac,
        "loss_if_bad_move_pct_of_portfolio": value * loss_frac / portfolio_value * 100,
        "capped_at_portfolio": bool(budget / loss_frac > portfolio_value),
        "formula": "max position = (portfolio value x risk %) / |bad-move %|, capped at the portfolio value",
        "caveats": ["One stock only: it ignores other holdings and how they move together.",
                    "A bad move can be bigger than the threshold (it is exceeded about as often as stated).",
                    "Overnight gaps, circuit limits and illiquidity can make real losses larger."],
    }


@dataclass
class Holding:
    symbol: str
    amount: float


def portfolio_risk(ctx: AppContext, holdings: list[Holding], refresh: bool = True) -> dict:
    if not (1 <= len(holdings) <= MAX_HOLDINGS):
        raise InvalidInputError(f"Provide between 1 and {MAX_HOLDINGS} holdings.")
    merged: dict[str, float] = {}
    for h in holdings:
        if not (h.amount > 0) or not math.isfinite(h.amount):
            raise InvalidInputError(f"Amount for {h.symbol} must be a positive number.")
        s = normalize_symbol(h.symbol)
        merged[s] = merged.get(s, 0.0) + h.amount
    frames = {s: _load(ctx, s, refresh) for s in merged}
    adj = pd.DataFrame({s: f["adj_close"] for s, f in frames.items()}).sort_index()
    r = adj.pct_change(fill_method=None).tail(CORR_WINDOW)
    syms = list(merged)
    total = sum(merged.values())
    w = np.array([merged[s] / total for s in syms])
    sig = np.array([float(vol_series(frames[s]).iloc[-1]["ewma"]) for s in syms])
    if not np.isfinite(sig).all():
        raise InsufficientDataError("Volatility could not be estimated for every holding.")
    corr = shrunk_corr(r[syms])
    cov = sig[:, None] * corr * sig[None, :]
    var_p = float(w @ cov @ w)
    sig_p = math.sqrt(var_p)
    contrib = w * (cov @ w) / var_p if var_p > 0 else w
    sample_corr = r[syms].corr(min_periods=MIN_OVERLAP)
    pairs = []
    for i in range(len(syms)):
        for j in range(i + 1, len(syms)):
            c = sample_corr.iloc[i, j]
            if np.isfinite(c):
                pairs.append({"a": syms[i], "b": syms[j], "correlation": float(c)})
    pairs.sort(key=lambda p: -abs(p["correlation"]))
    eff_n = float(1 / np.sum(w ** 2))
    warnings = []
    if len(syms) < 5:
        warnings.append("Fewer than 5 holdings: the portfolio is highly concentrated.")
    if w.max() > 0.4:
        warnings.append(f"{syms[int(w.argmax())]} is {w.max() * 100:.0f}% of the portfolio.")
    avg_corr = float(np.mean([p["correlation"] for p in pairs])) if pairs else None
    if avg_corr is not None and avg_corr > 0.6:
        warnings.append(f"Holdings are highly correlated (average {avg_corr:.2f}); diversification is limited.")
    last = {s: float(frames[s]["close"].iloc[-1]) for s in syms}
    return {
        "total_amount_inr": total, "as_of_date": str(max(f.index[-1] for f in frames.values()).date()),
        "portfolio": {
            "daily_volatility_pct": sig_p * 100, "annualised_volatility_pct": sig_p * ANN * 100,
            "weighted_avg_holding_volatility_pct": float((w * sig).sum() * 100),
            "diversification_ratio": float((w * sig).sum() / sig_p) if sig_p > 0 else None,
            "effective_number_of_holdings": eff_n, "average_pairwise_correlation": avg_corr,
        },
        "holdings": [{"symbol": s, "amount_inr": merged[s], "weight_pct": float(w[i] * 100),
                      "last_close": last[s], "daily_volatility_pct": float(sig[i] * 100),
                      "annualised_volatility_pct": float(sig[i] * ANN * 100),
                      "risk_contribution_pct": float(contrib[i] * 100)} for i, s in enumerate(syms)],
        "most_correlated_pairs": pairs[:5],
        "correlation_matrix": {"symbols": syms, "values": sample_corr.fillna(0).round(3).values.tolist()},
        "warnings": warnings,
        "method": "EWMA volatilities (lambda 0.94) + trailing-250-session correlations shrunk 20% toward average",
        "not_provided": "No calibrated portfolio forecast range: that variant failed its out-of-sample "
                        "coverage test at the 5-day horizon, so it is deliberately not offered.",
        "observations_used": int(r[syms].dropna(how="all").shape[0]),
    }
