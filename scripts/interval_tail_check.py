"""Out-of-sample coverage of the forecast ranges on small stocks: market-cap ranks 501-1500.

Question: do the shipped ranges keep their stated coverage on stocks far smaller than the Nifty 500?
Ranking: current market cap (Yahoo snapshot, reports/market_cap_snapshot.csv) of every stock outside the
Nifty 500; rank 501 = the largest of them. Bands: 501-1000 and 1001-1500 (rank counted after the Nifty 500).
Stocks without a market cap, or with fewer than 400 cached sessions, are excluded and counted.

Design: identical to scripts/interval_midcap_check.py. Before every 126-session test block the volatility
model and the calibration are fitted on Nifty 200 stocks ONLY (252-session purged window); the shipped
ranges are applied to the Nifty 200 (reference) and to each band. Methods: fhs_xgb (shipped), fhs_xgb_grp
(the band's own calibration window, fitted before each block), fhs_ewma, uncond, normal_ewma.
Pre-registered rule (fixed before running), per band and method: eligible if overall 80 % coverage is
within +-3 pp at BOTH horizons and within +-5 pp in every calendar year.
Decision per band: fhs_xgb eligible -> the band can be called "tested"; else fhs_xgb_grp eligible -> a
band-specific calibration is viable; else the band keeps the "not validated" warning.

Usage: python scripts/interval_tail_check.py [--quick] [--skip-fetch] [--max-rank 1000]
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

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.stats import norm  # noqa: E402

from app.features.engineering import FEATURE_NAMES  # noqa: E402
from interval_experiment import (  # noqa: E402
    BLOCK, CAL, EPS, LEVELS, START_POS, interval_score, vol_inputs, xgb_model,
)
from interval_midcap_check import cached_rows, fetch_extra  # noqa: E402
from pooled_experiment import build_panel, log, nw_t  # noqa: E402

REPORTS = ROOT / "reports"
METHODS = ["fhs_xgb", "fhs_xgb_grp", "fhs_ewma", "uncond", "normal_ewma"]
BANDS = {"r501_1000": (1, 500), "r1001_1500": (501, 1000)}      # rank after the Nifty 500, inclusive


def run(args):
    n200 = set(pd.read_csv(ROOT / "app/data/nifty200.csv")["Symbol"])
    csv500 = pd.read_csv(ROOT / "app/data/nifty500.csv")
    res = [json.loads(l) for l in (REPORTS / "benchmark_results.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    base = sorted({r["symbol"] for r in res if r["status"] == "ok"} & n200)

    mc = pd.read_csv(REPORTS / "market_cap_snapshot.csv")
    ranked = mc[mc["market_cap_inr"].notna()].sort_values("market_cap_inr", ascending=False).reset_index(drop=True)
    ranked["rank"] = ranked.index + 1
    cand = ranked[ranked["rank"] <= args.max_rank]
    if args.quick:
        cand = pd.concat([cand[cand["rank"] <= 20], cand[(cand["rank"] > 500) & (cand["rank"] <= 520)]])
    syms_all = list(cand["symbol"])
    failed = {} if args.skip_fetch else fetch_extra(syms_all, max_minutes=args.fetch_minutes)
    rows = cached_rows(syms_all)
    usable = [s for s in syms_all if rows[s] >= 400]
    rank_of = dict(zip(ranked["symbol"], ranked["rank"]))
    band_of = {s: next((b for b, (lo, hi) in BANDS.items() if lo <= rank_of[s] <= hi), None) for s in usable}
    usable = [s for s in usable if band_of[s]]
    n_no_cap = int((mc["market_cap_inr"].isna()).sum())
    log(f"base {len(base)} Nifty 200; ranked {len(ranked)} (no market cap: {n_no_cap}); candidates {len(syms_all)}; "
        f"usable (>=400 sessions) {len(usable)}; fetch failures {len(failed)}")

    sectors = dict(zip(csv500["Symbol"], csv500["Industry"]))
    panel = build_panel(base + usable, sectors, [1, 5], cross_sectional=False)
    xcols_needed = list(FEATURE_NAMES) + [c for c in panel.columns if c.startswith("nifty_") or c.startswith("vix")]
    panel = panel.dropna(subset=xcols_needed)
    n_per_date = panel.groupby(level="date").size()
    dates = pd.DatetimeIndex(n_per_date[n_per_date >= 30].index)
    panel = panel[panel.index.get_level_values("date").isin(dates)]
    N = len(dates)
    P = pd.Series(np.arange(N), index=dates).reindex(panel.index.get_level_values("date")).values
    syms = panel.index.get_level_values("symbol")
    is_base = syms.isin(base)
    group = np.array(["n200" if b else band_of.get(s) for s, b in zip(syms, is_base)])
    GROUPS = ["n200"] + [b for b in BANDS if (group == b).any()]
    years = panel.index.get_level_values("date").year.values
    log(f"panel {len(panel):,} rows; " + ", ".join(f"{g}: {int((group == g).sum()):,}" for g in GROUPS) + f"; {N} dates")
    V = vol_inputs(panel, [1, 5])
    X = panel[xcols_needed].astype("float32").copy()
    for nme in ("ewma", "roll60", "lr252"):
        X[f"log_{nme}"] = np.log(np.clip(V[nme], 1e-4, None)).astype("float32")
    stocks_per_group = {g: int(pd.Series(syms[group == g]).nunique()) for g in GROUPS}
    out = {"market_cap_snapshot_date": str(mc["as_of"].iloc[0]), "base_stocks": len(base),
           "ranked_stocks": int(len(ranked)), "stocks_without_market_cap": n_no_cap,
           "candidates": len(syms_all), "usable_stocks_by_group": stocks_per_group, "fetch_failures": failed,
           "rank_note": "rank counted among stocks outside the Nifty 500, by market cap on the snapshot date",
           "oos_start": str(dates[START_POS].date()), "oos_end": str(dates[-1].date()), "horizons": {}}

    for h in (1, 5):
        y = panel[f"y{h}"].values
        yfin = ~np.isnan(y)
        rv = V[f"rv{h}"]
        sq = np.sqrt(h)
        rec = {m: {f"{lv}_{s}": np.full(len(panel), np.nan) for lv in LEVELS for s in ("lo", "hi")} for m in METHODS}
        starts = list(range(START_POS, N, BLOCK))
        for t0 in starts:
            t1 = min(t0 + BLOCK, N)
            cal_hi = t0 - h - 1
            cal_lo = cal_hi - CAL + 1
            cal = (P >= cal_lo) & (P <= cal_hi) & yfin
            test = (P >= t0) & (P < t1)
            tr = (P <= cal_lo - h - 1) & yfin & ~np.isnan(rv) & is_base         # Nifty 200 ONLY
            xm = xgb_model(args.quick)
            xm.fit(X[tr], np.log(rv[tr] + EPS))
            need = cal | test
            sx = np.full(len(panel), np.nan)
            sx[need] = np.sqrt(np.exp(np.asarray(xm.predict(X[need]), dtype=float))) * sq
            se = V["ewma"] * sq
            sigma = {"fhs_xgb": sx, "fhs_xgb_grp": sx, "fhs_ewma": se, "normal_ewma": se}

            def zq(sig, mask):
                ok = mask & np.isfinite(sig) & (sig > 0)
                if ok.sum() < 200:
                    return None
                z = y[ok] / sig[ok]
                return {lv: (float(np.quantile(z, (1 - lv) / 2)), float(np.quantile(z, 1 - (1 - lv) / 2)))
                        for lv in LEVELS}

            base_z = {m: zq(sigma[m], cal & is_base) for m in ("fhs_xgb", "fhs_ewma")}
            grp_z = {g: (zq(sx, cal & (group == g)) or base_z["fhs_xgb"]) for g in GROUPS}
            for g in GROUPS:
                tm = test & (group == g)
                for m in METHODS:
                    for lv in LEVELS:
                        if m == "uncond":
                            lo, hi = V[f"uq{h}_{lv}_lo"][tm], V[f"uq{h}_{lv}_hi"][tm]
                        else:
                            if m == "normal_ewma":
                                zl, zh = norm.ppf((1 - lv) / 2), norm.ppf(1 - (1 - lv) / 2)
                            elif m == "fhs_xgb_grp":
                                zl, zh = grp_z[g][lv]
                            else:
                                zl, zh = base_z[m][lv]
                            lo, hi = zl * sigma[m][tm], zh * sigma[m][tm]
                        rec[m][f"{lv}_lo"][tm], rec[m][f"{lv}_hi"][tm] = lo, hi
            log(f"h={h} block {starts.index(t0) + 1}/{len(starts)} (train {int(tr.sum()):,} rows)")

        oos = (P >= START_POS) & yfin
        for m in METHODS:
            oos &= np.isfinite(rec[m]["0.8_lo"])
        ew = V["ewma"]
        t_lo, t_hi = np.nanquantile(ew[oos & is_base], [1 / 3, 2 / 3])
        date_idx = panel.index.get_level_values("date")
        hres = {"groups": {}}
        for g in GROUPS:
            gm = oos & (group == g)
            gres = {"n_rows": int(gm.sum()), "n_stocks": int(pd.Series(syms[gm]).nunique()), "methods": {}}
            for m in METHODS:
                mres = {}
                for lv in LEVELS:
                    lo, hi = rec[m][f"{lv}_lo"][gm], rec[m][f"{lv}_hi"][gm]
                    yy = y[gm]
                    hit = (yy >= lo) & (yy <= hi)
                    sc = interval_score(yy, lo, hi, lv)
                    cov_stock = pd.Series(hit).groupby(syms[gm].values).mean() * 100
                    mres[str(lv)] = {
                        "coverage_pct": float(hit.mean() * 100),
                        "miss_below_pct": float((yy < lo).mean() * 100), "miss_above_pct": float((yy > hi).mean() * 100),
                        "mean_width_pct": float((hi - lo).mean() * 100), "mean_interval_score": float(sc.mean()),
                        "coverage_by_year_pct": {str(yr): float(hit[years[gm] == yr].mean() * 100)
                                                 for yr in sorted(set(years[gm]))},
                        "coverage_by_vol_tercile_pct": {
                            nm: float(hit[msk].mean() * 100) for nm, msk in
                            (("low", ew[gm] <= t_lo), ("mid", (ew[gm] > t_lo) & (ew[gm] <= t_hi)), ("high", ew[gm] > t_hi))},
                        "share_stocks_within_10pp": float(((cov_stock - lv * 100).abs() <= 10).mean() * 100),
                        "worst_decile_stock_coverage_pct": float(np.quantile(cov_stock, 0.1)),
                    }
                    if lv == 0.8:
                        mres["_d"] = pd.Series(sc).groupby(date_idx[gm].values).mean()
                gres["methods"][m] = mres
            ub = gres["methods"]["uncond"]["_d"]
            for m in METHODS:
                d = gres["methods"][m].pop("_d")
                gres["methods"][m]["is80_improvement_vs_uncond_pct"] = float((1 - d.mean() / ub.mean()) * 100)
                gres["methods"][m]["is80_diff_nw_t_vs_uncond"] = nw_t((ub - d).values, max(5, 2 * h))
            hres["groups"][g] = gres
            for m in ("fhs_xgb", "fhs_xgb_grp", "fhs_ewma"):
                r8 = gres["methods"][m]["0.8"]
                log(f"h={h} {g:<11} {m:<12} cov50 {gres['methods'][m]['0.5']['coverage_pct']:.1f} cov80 {r8['coverage_pct']:.1f} "
                    f"cov95 {gres['methods'][m]['0.95']['coverage_pct']:.1f} width80 {r8['mean_width_pct']:.2f}% "
                    f"by vol-tercile {({k: round(v, 1) for k, v in r8['coverage_by_vol_tercile_pct'].items()})}")
        out["horizons"][str(h)] = hres

    elig, verdict = {g: {} for g in GROUPS}, {}
    for g in GROUPS:
        for m in METHODS:
            ok = True
            for h in ("1", "5"):
                r8 = out["horizons"][h]["groups"][g]["methods"][m]["0.8"]
                ok &= abs(r8["coverage_pct"] - 80) <= 3 and all(abs(c - 80) <= 5 for c in r8["coverage_by_year_pct"].values())
            elig[g][m] = bool(ok)
        if g != "n200":
            verdict[g] = ("fhs_xgb eligible: this band can be described as tested" if elig[g]["fhs_xgb"] else
                          "fhs_xgb NOT eligible; a band-specific calibration (fhs_xgb_grp) is eligible" if elig[g]["fhs_xgb_grp"] else
                          "neither shipped nor group-calibrated method eligible: keep the warning for this band")
    out["eligibility"], out["verdict"] = elig, verdict
    log("eligibility:", json.dumps(elig))
    log("VERDICT:", json.dumps(verdict))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--skip-fetch", action="store_true")
    ap.add_argument("--max-rank", type=int, default=1000)    # rank among stocks outside the Nifty 500 (overall 501-1500)
    ap.add_argument("--fetch-minutes", type=float, default=120.0)
    a = ap.parse_args()
    t = time.time()
    out = run(a)
    out["runtime_minutes"] = round((time.time() - t) / 60, 1)
    sfx = "_quick" if a.quick else ""
    (REPORTS / f"interval_tail_check{sfx}.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    log(f"wrote reports/interval_tail_check{sfx}.json ({out['runtime_minutes']} min)")


if __name__ == "__main__":
    main()
