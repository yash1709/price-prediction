"""Forecast-range (prediction-interval) experiment with out-of-sample coverage tests.

Intervals are for the h-day return, applied around the latest close (centre = "no change"; the
experiments showed no model beats that centre). Methods:
  uncond      per-stock empirical quantiles of past h-day returns (trailing 500 sessions)
  normal_roll normal, sigma = 60-session rolling std of daily returns * sqrt(h)
  normal_ewma normal, sigma = EWMA (lambda 0.94) * sqrt(h)
  fhs_ewma    EWMA sigma, tail multipliers = empirical quantiles of y/sigma on a calibration window
  fhs_blend   sqrt(0.5 EWMA^2 + 0.5 long-run^2) sigma, empirical-quantile calibration
  fhs_xgb     pooled XGBoost volatility model (per-stock + Nifty/VIX features only, shippable), same
              empirical-quantile calibration
Walk-forward: every 126 sessions from START_POS; calibration window = the 252 sessions before the
(purged) test block; the XGB model is fit strictly before the calibration window.

Pre-registered selection rule (fixed before running): a method is eligible if overall 80 % coverage
is within +-3 pp of 80 % at BOTH horizons and within +-5 pp in every calendar year; the winner is the
eligible method with the lowest 80 % interval (Winkler) score, which must also beat `uncond` with a
Newey-West t > 2. If nothing is eligible, no interval is shipped.

Usage: python scripts/interval_experiment.py [--quick]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import joblib  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.stats import norm  # noqa: E402
from xgboost import XGBRegressor  # noqa: E402

from pooled_experiment import build_panel, log, nw_t  # noqa: E402

REPORTS = ROOT / "reports"
START_POS = 750
BLOCK = 126
CAL = 252
LEVELS = [0.5, 0.8, 0.95]
METHODS = ["uncond", "normal_roll", "normal_ewma", "fhs_ewma", "fhs_blend", "fhs_xgb"]
EPS = 1e-6


def long(w: pd.DataFrame, index) -> np.ndarray:
    return w.stack(future_stack=True).reindex(index).values


def vol_inputs(panel: pd.DataFrame, horizons):
    adjw = panel["adj"].unstack("symbol")
    r = adjw.pct_change(fill_method=None)
    r2 = r ** 2
    ewma = r2.ewm(alpha=0.06, adjust=False, min_periods=30).mean() ** 0.5
    lr = r2.rolling(252, min_periods=120).mean() ** 0.5
    d = {
        "roll60": long(r.rolling(60, min_periods=40).std(), panel.index),
        "ewma": long(ewma, panel.index),
        "lr252": long(lr, panel.index),
        "blend": long((0.5 * ewma ** 2 + 0.5 * lr ** 2) ** 0.5, panel.index),
    }
    for h in horizons:
        d[f"rv{h}"] = long(r2.rolling(h).mean().shift(-h), panel.index)  # mean r^2 over t+1..t+h
        bk = adjw / adjw.shift(h) - 1
        for lv in LEVELS:
            for side, q in (("lo", (1 - lv) / 2), ("hi", 1 - (1 - lv) / 2)):
                d[f"uq{h}_{lv}_{side}"] = long(bk.rolling(500, min_periods=250).quantile(q), panel.index)
    return d


def xgb_model(quick):
    return XGBRegressor(n_estimators=60 if quick else 300, learning_rate=0.05, max_depth=4,
                        subsample=0.7, colsample_bytree=0.7, min_child_weight=100, reg_lambda=10.0,
                        random_state=42, n_jobs=6, verbosity=0, tree_method="hist")


def interval_score(y, lo, hi, level):
    a = 1 - level
    return (hi - lo) + 2 / a * np.clip(lo - y, 0, None) + 2 / a * np.clip(y - hi, 0, None)


def run(args):
    res = [json.loads(l) for l in (REPORTS / "benchmark_results.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    symbols = sorted({r["symbol"] for r in res if r["status"] == "ok"})
    if args.quick:
        symbols = symbols[:40]
    idx = pd.read_csv(ROOT / "data" / "universe_index.csv")
    log(f"{len(symbols)} stocks; building panel…")
    panel = build_panel(symbols, dict(zip(idx["Symbol"], idx["Industry"])), [1, 5])
    feat_all = [c for c in panel.columns if c not in ("adj", "symbol") and not c.startswith("y")]
    panel = panel.dropna(subset=feat_all)
    n = panel.groupby(level="date").size()
    dates = pd.DatetimeIndex(n[n >= 30].index)
    panel = panel[panel.index.get_level_values("date").isin(dates)]
    N = len(dates)
    pos = pd.Series(np.arange(N), index=dates)
    P = pos.reindex(panel.index.get_level_values("date")).values
    years = panel.index.get_level_values("date").year.values
    V = vol_inputs(panel, [1, 5])
    # shippable feature set: per-stock features + Nifty/VIX (no cross-sectional/sector/US-hours series)
    from app.features.engineering import FEATURE_NAMES
    xcols = list(FEATURE_NAMES) + [c for c in feat_all if c.startswith("nifty_") or c.startswith("vix")]
    X = panel[xcols].astype("float32").copy()
    for nme in ("ewma", "roll60", "lr252"):
        X[f"log_{nme}"] = np.log(np.clip(V[nme], 1e-4, None)).astype("float32")
    xcols = list(X.columns)
    log(f"panel {len(panel):,} rows, {N} dates; XGB vol model uses {len(xcols)} features")
    out = {"stocks": len(symbols), "oos_start": str(dates[START_POS].date()),
           "oos_end": str(dates[-1].date()), "x_features": xcols, "horizons": {}}
    calib_out = {"created": time.strftime("%Y-%m-%d"), "levels": LEVELS, "horizons": {}}
    sig_names = {"normal_roll": "roll60", "normal_ewma": "ewma", "fhs_ewma": "ewma", "fhs_blend": "blend"}

    for h in (1, 5):
        y = panel[f"y{h}"].values
        yfin = ~np.isnan(y)
        rv = V[f"rv{h}"]
        sq = np.sqrt(h)
        rec = {m: {f"{lv}_{s}": np.full(len(panel), np.nan) for lv in LEVELS for s in ("lo", "hi")} for m in METHODS}
        sigma_h_used = {m: np.full(len(panel), np.nan) for m in METHODS}
        starts = list(range(START_POS, N, BLOCK))
        for t0 in starts + [N]:                       # N = final fit for production calibration
            final = t0 == N
            t1 = min(t0 + BLOCK, N)
            cal_hi = t0 - h - 1 if not final else N - h - 1
            cal_lo = cal_hi - CAL + 1
            cal = (P >= cal_lo) & (P <= cal_hi) & yfin
            test = (P >= t0) & (P < t1) if not final else np.zeros(len(P), bool)
            sig_h = {m: V[s] * sq for m, s in sig_names.items()}
            # --- XGB volatility model: fit strictly before the calibration window ---
            tr = (P <= cal_lo - h - 1) & yfin & ~np.isnan(rv)
            xm = xgb_model(args.quick)
            xm.fit(X[tr], np.log(rv[tr] + EPS))
            need = cal | test
            sx = np.full(len(panel), np.nan)
            sx[need] = np.sqrt(np.exp(np.asarray(xm.predict(X[need]), dtype=float))) * sq
            sig_h["fhs_xgb"] = sx
            calib_block = {}
            for m in ("normal_roll", "normal_ewma", "fhs_ewma", "fhs_blend", "fhs_xgb"):
                s = sig_h[m]
                if m.startswith("fhs"):
                    ok = cal & np.isfinite(s) & (s > 0)
                    z = y[ok] / s[ok]
                    qs = {lv: (float(np.quantile(z, (1 - lv) / 2)), float(np.quantile(z, 1 - (1 - lv) / 2)))
                          for lv in LEVELS}
                    calib_block[m] = {str(lv): list(v) for lv, v in qs.items()}
                else:
                    qs = {lv: (float(norm.ppf((1 - lv) / 2)), float(norm.ppf(1 - (1 - lv) / 2))) for lv in LEVELS}
                for lv in LEVELS:
                    rec[m][f"{lv}_lo"][test] = qs[lv][0] * s[test]
                    rec[m][f"{lv}_hi"][test] = qs[lv][1] * s[test]
                sigma_h_used[m][test] = s[test]
            for lv in LEVELS:
                for sd in ("lo", "hi"):
                    rec["uncond"][f"{lv}_{sd}"][test] = V[f"uq{h}_{lv}_{sd}"][test]
            if final:
                calib_out["horizons"][str(h)] = {
                    "calibration_window": [str(dates[cal_lo].date()), str(dates[cal_hi].date())],
                    "z_quantiles": calib_block}
                joblib.dump({"model": xm, "features": xcols, "horizon": h, "eps": EPS},
                            REPORTS / f"vol_model_h{h}_experiment.joblib")
            else:
                log(f"h={h} block {starts.index(t0) + 1}/{len(starts)}: cal {dates[cal_lo].date()}.."
                    f"{dates[cal_hi].date()} train rows {tr.sum():,}")

        # ----- evaluation on all OOS rows with known targets -----
        oos = (P >= START_POS) & yfin & np.isfinite(rec["uncond"]["0.8_lo"]) & np.isfinite(rec["fhs_xgb"]["0.8_lo"]) \
            & np.isfinite(rec["fhs_blend"]["0.8_lo"])
        ew = V["ewma"]
        t_lo, t_hi = np.nanquantile(ew[oos], [1 / 3, 2 / 3])
        date_idx = panel.index.get_level_values("date")
        hres = {"n_rows": int(oos.sum()), "methods": {}}
        base_is = None
        for m in METHODS:
            mres = {}
            for lv in LEVELS:
                lo, hi = rec[m][f"{lv}_lo"][oos], rec[m][f"{lv}_hi"][oos]
                yy = y[oos]
                hit = (yy >= lo) & (yy <= hi)
                sc = interval_score(yy, lo, hi, lv)
                mres[str(lv)] = {
                    "coverage_pct": float(hit.mean() * 100),
                    "miss_below_pct": float((yy < lo).mean() * 100),
                    "miss_above_pct": float((yy > hi).mean() * 100),
                    "mean_width_pct": float((hi - lo).mean() * 100),
                    "mean_interval_score": float(sc.mean()),
                    "coverage_by_year_pct": {str(yr): float(hit[years[oos] == yr].mean() * 100)
                                             for yr in sorted(set(years[oos]))},
                    "coverage_by_vol_tercile_pct": {
                        nm: float(hit[msk].mean() * 100) for nm, msk in
                        (("low", ew[oos] <= t_lo), ("mid", (ew[oos] > t_lo) & (ew[oos] <= t_hi)),
                         ("high", ew[oos] > t_hi))},
                }
                # per-stock coverage dispersion
                syms = panel.index.get_level_values("symbol")[oos]
                cov_s = pd.Series(hit).groupby(syms.values).mean() * 100
                mres[str(lv)]["share_stocks_within_10pp"] = float(((cov_s - lv * 100).abs() <= 10).mean() * 100)
                if lv == 0.8:
                    d = pd.Series(sc).groupby(date_idx[oos].values).mean()
                    mres["_daily_is_80"] = d
            hres["methods"][m] = mres
        base = hres["methods"]["uncond"]["_daily_is_80"]
        for m in METHODS:
            d = hres["methods"][m].pop("_daily_is_80")
            diff = base - d  # >0 : method better than uncond
            hres["methods"][m]["is80_improvement_vs_uncond_pct"] = float(
                (1 - d.mean() / base.mean()) * 100)
            hres["methods"][m]["is80_diff_nw_t_vs_uncond"] = nw_t(diff.values, max(5, 2 * h))
        out["horizons"][str(h)] = hres
        for m in METHODS:
            r8 = hres["methods"][m]["0.8"]
            log(f"h={h} {m:<12} cov80={r8['coverage_pct']:.1f}% width={r8['mean_width_pct']:.2f}% "
                f"IS80 vs uncond {hres['methods'][m]['is80_improvement_vs_uncond_pct']:+.1f}% "
                f"(t={hres['methods'][m]['is80_diff_nw_t_vs_uncond']})")
    # ---- pre-registered selection ----
    elig = {}
    for m in METHODS:
        ok = True
        for h in ("1", "5"):
            r8 = out["horizons"][h]["methods"][m]["0.8"]
            ok &= abs(r8["coverage_pct"] - 80) <= 3
            ok &= all(abs(c - 80) <= 5 for c in r8["coverage_by_year_pct"].values())
        elig[m] = bool(ok)
    cands = [m for m in METHODS if elig[m] and m != "uncond"]
    def tot_is(m):
        return sum(out["horizons"][h]["methods"][m]["0.8"]["mean_interval_score"] / out["horizons"][h]["methods"]["uncond"]["0.8"]["mean_interval_score"] for h in ("1", "5"))
    winner = None
    for m in sorted(cands, key=tot_is):
        if all((out["horizons"][h]["methods"][m]["is80_diff_nw_t_vs_uncond"] or 0) > 2 for h in ("1", "5")):
            winner = m
            break
    out["selection"] = {"eligible": elig, "winner": winner}
    log(f"eligible: {elig}; WINNER (pre-registered rule): {winner}")
    return out, calib_out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--quick", action="store_true")
    a = ap.parse_args()
    t = time.time()
    out, calib = run(a)
    out["runtime_minutes"] = round((time.time() - t) / 60, 1)
    sfx = "_quick" if a.quick else ""
    (REPORTS / f"interval_experiment{sfx}.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    (REPORTS / f"interval_calibration_candidates{sfx}.json").write_text(json.dumps(calib, indent=2), encoding="utf-8")
    log(f"wrote reports/interval_experiment{sfx}.json ({out['runtime_minutes']} min)")


if __name__ == "__main__":
    main()
