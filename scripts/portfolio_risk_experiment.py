"""Walk-forward test of portfolio forecast ranges on random portfolios of Nifty 200 stocks.

Method under test (what the app ships if it passes): portfolio sigma = sqrt(w' S w) with
S = D C D, D = EWMA daily volatilities (lambda 0.94), C = trailing-250-session correlation matrix
shrunk 20 % toward its average off-diagonal value; range = z * sigma * sqrt(h) where z are empirical
quantiles of (realised portfolio return / forecast sigma) on a 252-session calibration window.
Variants for comparison: normal quantiles instead of empirical; independence (correlations ignored);
perfect correlation (sum of weighted volatilities).

Pre-registered rule (fixed before running): `full_fhs` is shipped only if its overall 80 % coverage is
within +-3 pp of 80 % at both horizons AND within +-5 pp in every calendar year AND within +-5 pp in
every portfolio-size bucket (2-4, 5-9, 10-15 stocks). Otherwise no portfolio ranges are shipped.

Usage: python scripts/portfolio_risk_experiment.py [--portfolios 300] [--quick]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.stats import norm  # noqa: E402
from sqlalchemy import text  # noqa: E402

from app.core.config import get_settings  # noqa: E402
from app.database.session import make_engine  # noqa: E402

REPORTS = ROOT / "reports"
OUT = ROOT / "app" / "models" / "pretrained"
START_POS, BLOCK, CAL, STEP = 750, 126, 252, 5
LEVELS = [0.5, 0.8, 0.95]
CORR_WINDOW, SHRINK = 250, 0.2
METHODS = ["full_fhs", "full_normal", "indep_fhs", "perfect_fhs"]


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


from app.services.risk import shrunk_corr  # noqa: E402  (the exact function the app uses)


def run(args):
    res = [json.loads(l) for l in (REPORTS / "benchmark_results.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    symbols = sorted({r["symbol"] for r in res if r["status"] == "ok"})
    if args.quick:
        symbols = symbols[:60]
    engine = make_engine(get_settings().resolved_database_url)
    with engine.connect() as conn:
        raw = pd.read_sql(text("select symbol,date,adj_close from price_bars"), conn, parse_dates=["date"])
    adj = raw[raw["symbol"].isin(symbols)].pivot(index="date", columns="symbol", values="adj_close").sort_index()
    adj = adj[adj.notna().sum(axis=1) >= 30]
    dates = adj.index
    N = len(dates)
    r = adj.pct_change(fill_method=None)
    ewma = (r ** 2).ewm(alpha=0.06, adjust=False, min_periods=30).mean() ** 0.5
    log(f"{adj.shape[1]} stocks, {N} dates ({dates[0].date()} → {dates[-1].date()})")

    rng = np.random.default_rng(42)
    cols = list(adj.columns)
    portfolios = []
    for _ in range(args.portfolios):
        k = int(rng.integers(2, 16))
        members = rng.choice(len(cols), size=k, replace=False)
        w = rng.dirichlet(np.ones(k))
        portfolios.append((members, w))
    bucket = lambda k: "2-4" if k <= 4 else ("5-9" if k <= 9 else "10-15")

    out = {"stocks": adj.shape[1], "portfolios": args.portfolios, "oos_start": str(dates[START_POS].date()),
           "oos_end": str(dates[-1].date()), "shrink": SHRINK, "corr_window": CORR_WINDOW,
           "horizons": {}}
    calib_out = {"created": time.strftime("%Y-%m-%d"), "levels": LEVELS, "shrink": SHRINK,
                 "corr_window": CORR_WINDOW, "ewma_alpha": 0.06, "horizons": {}}
    sample_pos = list(range(CORR_WINDOW + 1, N - 1, STEP))
    rv = r.values
    ev = ewma.values
    for h in (1, 5):
        yv = (adj.shift(-h) / adj - 1).values
        sq = np.sqrt(h)
        rows = []  # per (portfolio, date): pos, k, y, sigma by method
        for pos in sample_pos:
            C = shrunk_corr(r.iloc[pos - CORR_WINDOW: pos + 1])
            for pi, (mem, w) in enumerate(portfolios):
                s = ev[pos, mem]
                y = yv[pos, mem]
                if not (np.isfinite(s).all() and np.isfinite(y).all()):
                    continue
                Sg = s[:, None] * C[np.ix_(mem, mem)] * s[None, :]
                rows.append((pos, pi, len(mem), float(w @ y), float(np.sqrt(w @ Sg @ w)),
                             float(np.sqrt(np.sum((w * s) ** 2))), float(np.sum(w * s))))
        d = pd.DataFrame(rows, columns=["pos", "pi", "k", "y", "sig_full", "sig_indep", "sig_perfect"])
        d["year"] = dates[d["pos"].values].year
        d["bucket"] = d["k"].map(bucket)
        log(f"h={h}: {len(d):,} portfolio-date observations")
        lo = {m: {lv: np.full(len(d), np.nan) for lv in LEVELS} for m in METHODS}
        hi = {m: {lv: np.full(len(d), np.nan) for lv in LEVELS} for m in METHODS}
        sig_col = {"full": "sig_full", "indep": "sig_indep", "perfect": "sig_perfect"}
        starts = list(range(START_POS, N, BLOCK))
        for t0 in starts + [N]:
            final = t0 == N
            cal_hi = (N - h - 1) if final else (t0 - h - 1)
            cal_lo = cal_hi - CAL + 1
            cal = (d["pos"] >= cal_lo) & (d["pos"] <= cal_hi)
            test = (d["pos"] >= t0) & (d["pos"] < min(t0 + BLOCK, N)) if not final else pd.Series(False, index=d.index)
            zs = {}
            for fam in ("full", "indep", "perfect"):
                z = d.loc[cal, "y"] / (d.loc[cal, sig_col[fam]] * sq)
                zs[fam] = {lv: (float(np.quantile(z, (1 - lv) / 2)), float(np.quantile(z, 1 - (1 - lv) / 2)))
                           for lv in LEVELS}
            if final:
                calib_out["horizons"][str(h)] = {
                    "calibration_window": [str(dates[cal_lo].date()), str(dates[cal_hi].date())],
                    "z_quantiles": {str(lv): list(zs["full"][lv]) for lv in LEVELS}}
                continue
            for m in METHODS:
                fam = m.split("_")[0]
                for lv in LEVELS:
                    if m.endswith("normal"):
                        zl, zh = norm.ppf((1 - lv) / 2), norm.ppf(1 - (1 - lv) / 2)
                    else:
                        zl, zh = zs[fam][lv]
                    sg = d.loc[test, sig_col[fam]].values * sq
                    lo[m][lv][test.values] = zl * sg
                    hi[m][lv][test.values] = zh * sg
        oos = (d["pos"] >= START_POS).values & np.isfinite(lo["full_fhs"][0.8])
        hres = {"n_obs": int(oos.sum()), "methods": {}}
        for m in METHODS:
            mres = {}
            for lv in LEVELS:
                hit = (d["y"].values >= lo[m][lv]) & (d["y"].values <= hi[m][lv])
                mres[str(lv)] = {
                    "coverage_pct": float(hit[oos].mean() * 100),
                    "mean_width_pct": float((hi[m][lv] - lo[m][lv])[oos].mean() * 100),
                    "by_year": {str(y): float(hit[oos & (d["year"].values == y)].mean() * 100)
                                for y in sorted(set(d["year"].values[oos]))},
                    "by_size": {b: float(hit[oos & (d["bucket"].values == b)].mean() * 100)
                                for b in ("2-4", "5-9", "10-15")},
                }
            hres["methods"][m] = mres
            r8 = mres["0.8"]
            log(f"h={h} {m:<12} cov50 {mres['0.5']['coverage_pct']:.1f} cov80 {r8['coverage_pct']:.1f} "
                f"cov95 {mres['0.95']['coverage_pct']:.1f} width80 {r8['mean_width_pct']:.2f}% "
                f"by size {r8['by_size']}")
        out["horizons"][str(h)] = hres
    ok = True
    for h in ("1", "5"):
        r8 = out["horizons"][h]["methods"]["full_fhs"]["0.8"]
        ok &= abs(r8["coverage_pct"] - 80) <= 3
        ok &= all(abs(v - 80) <= 5 for v in r8["by_year"].values())
        ok &= all(abs(v - 80) <= 5 for v in r8["by_size"].values())
    out["selection"] = {"full_fhs_eligible": bool(ok)}
    log(f"pre-registered rule: full_fhs eligible = {ok}")
    return out, calib_out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--portfolios", type=int, default=300)
    ap.add_argument("--quick", action="store_true")
    a = ap.parse_args()
    t = time.time()
    out, calib = run(a)
    out["runtime_minutes"] = round((time.time() - t) / 60, 1)
    sfx = "_quick" if a.quick else ""
    (REPORTS / f"portfolio_risk_experiment{sfx}.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    if not a.quick and out["selection"]["full_fhs_eligible"]:
        OUT.mkdir(parents=True, exist_ok=True)
        (OUT / "portfolio_calibration.json").write_text(json.dumps(calib, indent=2), encoding="utf-8")
        log("wrote app/models/pretrained/portfolio_calibration.json")
    log(f"done ({out['runtime_minutes']} min)")


if __name__ == "__main__":
    main()
