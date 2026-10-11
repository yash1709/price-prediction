"""Pooled cross-sectional model experiment (one model across all stocks) vs the naive baseline.

Pre-registered success criterion (fixed before running): the selected pooled system is "better than
naive" only if, over all out-of-sample test blocks combined, the MAE improvement vs naive has a
Newey-West t-stat > 2 AND is positive in at least 2 of the 3 blocks.

Protocol (per horizon h, per test block k of 126 sessions):
  * sub-train  : as-of dates <= val_start - h - 1          -> fit candidates, score on validation
  * validation : the 126 sessions before the (purged) test block -> choose model and shrink alpha
  * full-train : as-of dates <= test_start - h - 1          -> refit, predict the test block (frozen)
  Nothing in the test block is used for fitting or selection. alpha=0 means "naive" (fallback).

Features: the 24 per-stock features + market (Nifty, India VIX; USD/INR and crude lagged one day
because they close after NSE) + cross-sectional (relative strength, ranks, sector averages).

Usage:  python scripts/pooled_experiment.py [--horizons 1 5] [--blocks 3] [--quick]
Needs the price cache from scripts/benchmark_universe.py (data/nse.db) and its results file.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore", message="An input array is constant")  # naive fallback => const preds

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.linear_model import Ridge  # noqa: E402
from sklearn.pipeline import Pipeline  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402
from sqlalchemy import text  # noqa: E402
from xgboost import XGBRegressor  # noqa: E402

from app.core.config import get_settings  # noqa: E402
from app.database.session import make_engine  # noqa: E402
from app.features.engineering import FEATURE_NAMES, compute_features  # noqa: E402

REPORTS = ROOT / "reports"
MARKET_DIR = ROOT / "data" / "market"
ALPHAS = [0.0, 0.25, 0.5, 0.75, 1.0]
BLOCK = 126
VAL = 126


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


# ------------------------------------------------------------------------------------------
# data
# ------------------------------------------------------------------------------------------
def fetch_market() -> pd.DataFrame:
    """Daily closes: Nifty 50, India VIX, USD/INR, WTI crude (cached as CSV)."""
    import yfinance as yf
    MARKET_DIR.mkdir(parents=True, exist_ok=True)
    out = {}
    for name, tk in {"nifty": "^NSEI", "vix": "^INDIAVIX", "usdinr": "USDINR=X", "crude": "CL=F"}.items():
        f = MARKET_DIR / f"{name}.csv"
        if f.exists():
            s = pd.read_csv(f, index_col=0, parse_dates=True).iloc[:, 0]
        else:
            h = yf.Ticker(tk).history(period="12y", interval="1d", auto_adjust=False)
            if h.empty:
                log(f"WARNING: no data for {tk}; feature group '{name}' skipped")
                continue
            idx = pd.DatetimeIndex(h.index)
            s = pd.Series(h["Close"].values, index=(idx.tz_localize(None) if idx.tz else idx).normalize(),
                          name=name)
            s = s[~s.index.duplicated(keep="last")].sort_index()
            s.to_csv(f)
        out[name] = s
    return pd.DataFrame(out)


from app.features.market import market_features  # noqa: E402  (shared with the live app)


def build_panel(symbols: list[str], sectors: dict[str, str], horizons: list[int]) -> pd.DataFrame:
    s = get_settings()
    engine = make_engine(s.resolved_database_url)
    with engine.connect() as conn:
        raw = pd.read_sql(text("select symbol,date,open,high,low,close,adj_close,volume from price_bars"),
                          conn, parse_dates=["date"])
    raw = raw[raw["symbol"].isin(symbols)]
    parts = []
    for sym, g in raw.groupby("symbol"):
        g = g.drop(columns="symbol").set_index("date").sort_index()
        f = compute_features(g)
        f["adj"] = g["adj_close"]
        for h in horizons:
            f[f"y{h}"] = g["adj_close"].shift(-h) / g["adj_close"] - 1
        f["symbol"] = sym
        parts.append(f)
    p = pd.concat(parts)
    p.index.name = "date"
    p = p.reset_index().set_index(["date", "symbol"]).sort_index()

    # --- cross-sectional features (same-date information only) ---
    wide = {c: p[c].unstack("symbol") for c in ("ret_lag_1", "mom_5", "mom_20", "vol_20")}
    cnt = wide["ret_lag_1"].notna().sum(axis=1)
    ok = cnt >= 30
    mkt_ret = wide["ret_lag_1"].mean(axis=1).where(ok)
    new = {
        "xs_mkt_ret_1": mkt_ret, "xs_dispersion_1": wide["ret_lag_1"].std(axis=1).where(ok),
        "xs_breadth_1": (wide["ret_lag_1"] > 0).sum(axis=1).div(cnt).where(ok),
    }
    cols = {}
    cols["rel_ret_1"] = wide["ret_lag_1"].sub(mkt_ret, axis=0)
    cols["rel_mom_5"] = wide["mom_5"].sub(wide["mom_5"].mean(axis=1), axis=0)
    cols["rel_mom_20"] = wide["mom_20"].sub(wide["mom_20"].mean(axis=1), axis=0)
    cols["rank_mom_20"] = wide["mom_20"].rank(axis=1, pct=True)
    cols["rank_vol_20"] = wide["vol_20"].rank(axis=1, pct=True)
    syms = list(wide["ret_lag_1"].columns)
    for src, name in (("ret_lag_1", "sector_ret_1"), ("mom_5", "sector_mom_5")):
        w = wide[src]
        out = pd.DataFrame(np.nan, index=w.index, columns=syms)
        for sec in set(sectors.get(x, "Unknown") for x in syms):
            members = [x for x in syms if sectors.get(x, "Unknown") == sec]
            if len(members) < 3:
                continue
            sub = w[members]
            tot, c = sub.fillna(0).sum(axis=1), sub.notna().sum(axis=1)
            ex = (tot.values[:, None] - sub.fillna(0).values) / (c.values[:, None] - 1)
            out[members] = ex
        cols[name] = out
    for name, w in cols.items():
        p[name] = w.stack(future_stack=True).reindex(p.index)
    for name, s_ in new.items():
        p[name] = s_.reindex(p.index.get_level_values("date")).values
    dates = pd.DatetimeIndex(sorted(p.index.get_level_values("date").unique()))
    mf = market_features(fetch_market(), dates)
    for c in mf.columns:
        p[c] = mf[c].reindex(p.index.get_level_values("date")).values
    return p


# ------------------------------------------------------------------------------------------
# models
# ------------------------------------------------------------------------------------------
def make_models(quick: bool) -> dict:
    n = 60 if quick else 300
    common = dict(n_estimators=n, learning_rate=0.03, max_depth=4, subsample=0.7, colsample_bytree=0.7,
                  min_child_weight=200, reg_lambda=10.0, random_state=42, n_jobs=6, verbosity=0,
                  tree_method="hist")
    return {
        "pooled_ridge": lambda: Pipeline([("s", StandardScaler()), ("m", Ridge(alpha=100.0))]),
        "pooled_xgb_l2": lambda: XGBRegressor(objective="reg:squarederror", **common),
        "pooled_xgb_l1": lambda: XGBRegressor(objective="reg:absoluteerror", **common),
    }


# ------------------------------------------------------------------------------------------
# statistics
# ------------------------------------------------------------------------------------------
def nw_t(x, lag: int) -> float | None:
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    n = len(x)
    if n < 10:
        return None
    e = x - x.mean()
    s = e @ e / n
    for l in range(1, lag + 1):
        s += 2 * (1 - l / (lag + 1)) * (e[l:] @ e[:-l]) / n
    se = np.sqrt(max(s, 0) / n)
    return float(x.mean() / se) if se > 0 else None


def evaluate(df: pd.DataFrame, h: int) -> dict:
    """df index (date, symbol); columns pred (return), y (return), adj (reference price)."""
    err_m, err_n = (df["pred"] - df["y"]).abs(), df["y"].abs()
    daily_diff = (err_n - err_m).groupby(level="date").mean()  # >0: model better than naive
    lag = max(5, 2 * h)
    ref = df["adj"]
    per = pd.DataFrame({"m": (ref * err_m).groupby(level="symbol").mean(),
                        "n": (ref * err_n).groupby(level="symbol").mean()})
    imp = (1 - per["m"] / per["n"]) * 100
    g = df.assign(r=df.groupby(level="date")["pred"].rank(pct=True))
    ic = g.groupby(level="date").apply(lambda d: d["pred"].corr(d["y"], method="spearman") if len(d) > 30 else np.nan)
    spread = g.groupby(level="date").apply(
        lambda d: d.loc[d["r"] > 0.8, "y"].mean() - d.loc[d["r"] <= 0.2, "y"].mean() if len(d) > 30 else np.nan)
    nz = (df["pred"] != 0) & (df["y"] != 0)
    return {
        "n_rows": int(len(df)), "n_dates": int(daily_diff.notna().sum()), "n_stocks": int(len(per)),
        "mae_improvement_pct": float((1 - err_m.mean() / err_n.mean()) * 100),
        "mae_diff_mean": float(daily_diff.mean()), "mae_diff_nw_t": nw_t(daily_diff, lag),
        "stocks_beating_naive": int((imp > 0).sum()), "stocks_losing_to_naive": int((imp < 0).sum()),
        "median_stock_mae_improvement_pct": float(imp.median()),
        "ic_mean": float(ic.mean()), "ic_nw_t": nw_t(ic, lag),
        "quintile_spread_mean_pct": float(spread.mean() * 100), "quintile_spread_nw_t": nw_t(spread, lag),
        "directional_accuracy_pct": float((np.sign(df.loc[nz, "pred"]) == np.sign(df.loc[nz, "y"])).mean() * 100)
        if nz.any() else None,
    }, daily_diff


# ------------------------------------------------------------------------------------------
# experiment
# ------------------------------------------------------------------------------------------
def run(args) -> dict:
    res = [json.loads(l) for l in (REPORTS / "benchmark_results.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    symbols = sorted({r["symbol"] for r in res if r["status"] == "ok"})
    if args.quick:
        symbols = symbols[:40]
    idx_csv = pd.read_csv(ROOT / "data" / "universe_index.csv")
    sectors = dict(zip(idx_csv["Symbol"], idx_csv["Industry"]))
    log(f"{len(symbols)} stocks; building panel…")
    panel = build_panel(symbols, sectors, args.horizons)
    feat_cols = [c for c in panel.columns if c not in ("adj", "symbol") and not c.startswith("y")]
    panel = panel.dropna(subset=feat_cols)
    dates = pd.DatetimeIndex(sorted(panel.index.get_level_values("date").unique()))
    ndate = panel.groupby(level="date").size()
    dates = ndate[ndate >= 30].index
    panel = panel[panel.index.get_level_values("date").isin(dates)]
    pos = pd.Series(np.arange(len(dates)), index=dates)
    P = pos.reindex(panel.index.get_level_values("date")).values
    N = len(dates)
    log(f"panel: {len(panel):,} rows, {len(feat_cols)} features, {N} dates "
        f"({dates[0].date()} → {dates[-1].date()})")
    X = panel[feat_cols].astype("float32")
    models = make_models(args.quick)
    out = {"features": feat_cols, "n_stocks": len(symbols), "dates": [str(dates[0].date()), str(dates[-1].date())],
           "horizons": {}}

    for h in args.horizons:
        y = panel[f"y{h}"].values
        yfin = ~np.isnan(y)
        blocks = []
        for k in range(args.blocks):
            t0 = N - BLOCK * (args.blocks - k)
            t1 = t0 + BLOCK if k < args.blocks - 1 else N
            blocks.append((t0, t1))
        per_block, all_sel, all_raw = [], [], {m: [] for m in models}
        for k, (t0, t1) in enumerate(blocks):
            val_hi = t0 - h - 1
            val_lo = val_hi - VAL + 1
            sub = (P <= val_lo - h - 1) & yfin
            val = (P >= val_lo) & (P <= val_hi) & yfin
            full = (P <= t0 - h - 1) & yfin
            test = (P >= t0) & (P < t1) & yfin

            def fit(mask, name):
                lo, hi = np.quantile(y[mask], [0.005, 0.995])
                mdl = models[name]()
                mdl.fit(X[mask], np.clip(y[mask], lo, hi))
                return mdl

            # ---- selection on validation (models fit on sub-train only) ----
            val_loss = {}
            for name in models:
                m = fit(sub, name)
                pv = np.asarray(m.predict(X[val]), dtype=float)
                for a in ALPHAS:
                    val_loss[(name, a)] = float(np.abs(a * pv - y[val]).mean())
            naive_val = float(np.abs(y[val]).mean())
            best = min(val_loss, key=lambda ka: (val_loss[ka], ka[1]))
            sel_name, sel_alpha = (best if val_loss[best] < naive_val else ("naive", 0.0))
            log(f"h={h} block {k + 1}/{args.blocks}: val naive={naive_val:.5f} best={best} "
                f"{val_loss[best]:.5f} -> selected {sel_name} alpha={sel_alpha}")

            # ---- refit on full train; frozen predictions on test ----
            preds = {}
            for name in models:
                m = fit(full, name)
                preds[name] = np.asarray(m.predict(X[test]), dtype=float)
            base = panel.loc[test, ["adj"]].assign(y=y[test])
            sel_pred = np.zeros(test.sum()) if sel_name == "naive" else sel_alpha * preds[sel_name]
            blk = {"test_dates": [str(dates[t0].date()), str(dates[min(t1, N) - 1].date())],
                   "selected": {"model": sel_name, "alpha": sel_alpha},
                   "validation_mae_naive": naive_val,
                   "validation_mae_best": val_loss[best]}
            blk["selected_eval"], d = evaluate(base.assign(pred=sel_pred), h)
            all_sel.append(base.assign(pred=sel_pred))
            blk["raw_models"] = {}
            for name in models:
                blk["raw_models"][name], _ = evaluate(base.assign(pred=preds[name]), h)
                all_raw[name].append(base.assign(pred=preds[name]))
            per_block.append(blk)
            log(f"   test MAE vs naive: selected {blk['selected_eval']['mae_improvement_pct']:+.3f}% | raw "
                + ", ".join(f"{n} {v['mae_improvement_pct']:+.3f}%" for n, v in blk["raw_models"].items()))

        comb_sel, _ = evaluate(pd.concat(all_sel), h)
        comb_raw = {n: evaluate(pd.concat(v), h)[0] for n, v in all_raw.items()}
        blocks_pos = sum(b["selected_eval"]["mae_improvement_pct"] > 0 for b in per_block)
        t = comb_sel["mae_diff_nw_t"]
        passed = bool(t is not None and t > 2 and blocks_pos >= 2 and comb_sel["mae_improvement_pct"] > 0)
        out["horizons"][str(h)] = {"blocks": per_block, "combined_selected": comb_sel,
                                   "combined_raw_models": comb_raw, "blocks_positive": blocks_pos,
                                   "criterion_passed": passed}
        log(f"h={h} COMBINED selected: MAE vs naive {comb_sel['mae_improvement_pct']:+.3f}% "
            f"(t={t}), blocks positive {blocks_pos}/{len(per_block)}, IC {comb_sel['ic_mean']:+.4f} "
            f"(t={comb_sel['ic_nw_t']}) -> criterion {'PASSED' if passed else 'NOT met'}")
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--horizons", type=int, nargs="+", default=[1, 5])
    ap.add_argument("--blocks", type=int, default=3)
    ap.add_argument("--quick", action="store_true", help="40 stocks, small models (smoke test)")
    a = ap.parse_args()
    t0 = time.time()
    out = run(a)
    out["runtime_minutes"] = round((time.time() - t0) / 60, 1)
    name = "pooled_experiment_quick.json" if a.quick else "pooled_experiment.json"
    (REPORTS / name).write_text(json.dumps(out, indent=2), encoding="utf-8")
    log(f"wrote reports/{name} ({out['runtime_minutes']} min)")


if __name__ == "__main__":
    main()
