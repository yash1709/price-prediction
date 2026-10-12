"""Out-of-sample coverage of the forecast ranges on stocks the method never saw (Nifty 201-500).

Question: do the shipped ranges keep their stated coverage on Nifty 500 stocks outside the Nifty 200?
Design: same walk-forward as scripts/interval_experiment.py (blocks of 126 sessions from START_POS,
252-session purged calibration window). Before every test block the volatility model AND the
calibration are fitted on Nifty 200 stocks ONLY; the resulting ranges are applied to (a) the Nifty 200
stocks (reference) and (b) the 301 unseen Nifty 201-500 stocks. Methods:
  fhs_xgb       shipped: Nifty-200 model and Nifty-200 calibration applied to every stock
  fhs_xgb_grp   same model, but calibration quantiles from the evaluated group's OWN calibration window
                (a legitimate fix: fitted on data BEFORE each block, never on the test block)
  fhs_ewma      shipped fallback, Nifty-200 calibration
  uncond / normal_ewma   baselines
Pre-registered rule (fixed before running), per group and method: eligible if overall 80 % coverage is
within +-3 pp at BOTH horizons and within +-5 pp in every calendar year.
Decision: fhs_xgb eligible on 201-500 -> the "never tested" note can become measured evidence;
otherwise, fhs_xgb_grp eligible -> a mid-cap-specific calibration is viable; otherwise the warnings stay.

Usage: python scripts/interval_midcap_check.py [--quick] [--skip-fetch]
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
from sqlalchemy import text  # noqa: E402

from app.core.errors import AppError, ProviderRateLimitError  # noqa: E402
from app.core.config import get_settings  # noqa: E402
from app.database.session import make_engine  # noqa: E402
from app.features.engineering import FEATURE_NAMES  # noqa: E402
from interval_experiment import (  # noqa: E402
    BLOCK, CAL, EPS, LEVELS, START_POS, interval_score, vol_inputs, xgb_model,
)
from pooled_experiment import build_panel, log, nw_t  # noqa: E402

REPORTS = ROOT / "reports"
METHODS = ["fhs_xgb", "fhs_xgb_grp", "fhs_ewma", "uncond", "normal_ewma"]
GROUPS = ["n200", "mid"]


def cached_rows(symbols: list[str]) -> dict[str, int]:
    engine = make_engine(get_settings().resolved_database_url)
    with engine.connect() as conn:
        rows = conn.execute(text("select symbol, count(*) from price_bars group by symbol")).fetchall()
    have = {r[0]: r[1] for r in rows}
    return {s: have.get(s, 0) for s in symbols}


def fetch_extra(symbols: list[str], max_minutes: float = 40.0) -> dict:
    """Make sure each symbol has cached history (resumable; rate-limit back-off; failures reported)."""
    from app.services.context import build_context
    from app.services.workflow import fetch_data
    ctx = build_context()
    have = cached_rows(symbols)
    todo = [s for s in symbols if have[s] < 1000]
    log(f"{len(symbols) - len(todo)} of {len(symbols)} already cached; fetching {len(todo)}")
    failed, t0 = {}, time.time()
    for i, sym in enumerate(todo, 1):
        if (time.time() - t0) / 60 > max_minutes:
            log("time budget reached; stopping fetch (re-run to resume)")
            break
        for attempt in range(3):
            try:
                fetch_data(ctx, sym)
                break
            except ProviderRateLimitError:
                time.sleep(30 * (attempt + 1))
            except AppError as exc:
                failed[sym] = f"{exc.code}: {exc.message}"[:120]
                break
            except Exception as exc:  # noqa: BLE001 - recorded, run continues
                failed[sym] = f"{type(exc).__name__}: {exc}"[:120]
                break
        time.sleep(0.3)
        if i % 25 == 0:
            log(f"  fetched {i}/{len(todo)} (failed so far: {len(failed)})")
    ctx.tasks.shutdown()
    return failed


def run(args):
    n200 = set(pd.read_csv(ROOT / "app/data/nifty200.csv")["Symbol"])
    csv500 = pd.read_csv(ROOT / "app/data/nifty500.csv")
    n500 = set(csv500["Symbol"])
    res = [json.loads(l) for l in (REPORTS / "benchmark_results.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    base = sorted({r["symbol"] for r in res if r["status"] == "ok"} & n200)
    extra = sorted(n500 - n200)
    if args.quick:
        extra = extra[:40]
    failed = {} if args.skip_fetch else fetch_extra(extra)
    rows = cached_rows(extra)
    usable_extra = [s for s in extra if rows[s] >= 400]
    log(f"base (Nifty 200) stocks: {len(base)}; Nifty 201-500 with >=400 sessions: {len(usable_extra)} of "
        f"{len(extra)}; fetch failures: {len(failed)}")

    sectors = dict(zip(csv500["Symbol"], csv500["Industry"]))
    panel = build_panel(base + usable_extra, sectors, [1, 5], cross_sectional=False)
    xcols_needed = list(FEATURE_NAMES) + [c for c in panel.columns if c.startswith("nifty_") or c.startswith("vix")]
    panel = panel.dropna(subset=xcols_needed)
    n_per_date = panel.groupby(level="date").size()
    dates = pd.DatetimeIndex(n_per_date[n_per_date >= 30].index)
    panel = panel[panel.index.get_level_values("date").isin(dates)]
    N = len(dates)
    P = pd.Series(np.arange(N), index=dates).reindex(panel.index.get_level_values("date")).values
    syms = panel.index.get_level_values("symbol")
    is_base = syms.isin(base)
    group = np.where(is_base, "n200", "mid")
    years = panel.index.get_level_values("date").year.values
    log(f"panel {len(panel):,} rows ({int(is_base.sum()):,} Nifty-200, {int((~is_base).sum()):,} mid), {N} dates")
    V = vol_inputs(panel, [1, 5])
    X = panel[xcols_needed].astype("float32").copy()
    for nme in ("ewma", "roll60", "lr252"):
        X[f"log_{nme}"] = np.log(np.clip(V[nme], 1e-4, None)).astype("float32")
    out = {"base_stocks": len(base), "mid_stocks_used": len(usable_extra), "mid_stocks_total": len(extra),
           "fetch_failures": failed, "oos_start": str(dates[START_POS].date()), "oos_end": str(dates[-1].date()),
           "horizons": {}}

    for h in (1, 5):
        y = panel[f"y{h}"].values
        yfin = ~np.isnan(y)
        rv = V[f"rv{h}"]
        sq = np.sqrt(h)
        rec = {m: {f"{lv}_{s}": np.full(len(panel), np.nan) for lv in LEVELS for s in ("lo", "hi")} for m in METHODS}
        zlog = []
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
                z = y[ok] / sig[ok]
                return {lv: (float(np.quantile(z, (1 - lv) / 2)), float(np.quantile(z, 1 - (1 - lv) / 2)))
                        for lv in LEVELS}

            base_z = {m: zq(sigma[m], cal & is_base) for m in ("fhs_xgb", "fhs_ewma")}
            mid_z = zq(sx, cal & ~is_base)
            zlog.append({"block_start": str(dates[t0].date()), "n200_z80": base_z["fhs_xgb"][0.8],
                         "mid_z80": mid_z[0.8], "n_cal_mid": int((cal & ~is_base).sum())})
            for g, gmask in (("n200", is_base), ("mid", ~is_base)):
                tm = test & gmask
                for m in METHODS:
                    for lv in LEVELS:
                        if m == "uncond":
                            lo, hi = V[f"uq{h}_{lv}_lo"][tm], V[f"uq{h}_{lv}_hi"][tm]
                        else:
                            if m == "normal_ewma":
                                zl, zh = norm.ppf((1 - lv) / 2), norm.ppf(1 - (1 - lv) / 2)
                            elif m == "fhs_xgb_grp":
                                zl, zh = (base_z["fhs_xgb"][lv] if g == "n200" else mid_z[lv])
                            else:
                                zl, zh = base_z[m][lv]
                            lo, hi = zl * sigma[m][tm], zh * sigma[m][tm]
                        rec[m][f"{lv}_lo"][tm], rec[m][f"{lv}_hi"][tm] = lo, hi
            log(f"h={h} block {starts.index(t0) + 1}/{len(starts)} (train {int(tr.sum()):,} rows, "
                f"mid z80 {mid_z[0.8][0]:+.2f}/{mid_z[0.8][1]:+.2f} vs n200 {base_z['fhs_xgb'][0.8][0]:+.2f}/"
                f"{base_z['fhs_xgb'][0.8][1]:+.2f})")

        oos = (P >= START_POS) & yfin
        for m in METHODS:
            oos &= np.isfinite(rec[m]["0.8_lo"])
        ew = V["ewma"]
        t_lo, t_hi = np.nanquantile(ew[oos & is_base], [1 / 3, 2 / 3])    # volatility terciles from Nifty 200
        date_idx = panel.index.get_level_values("date")
        hres = {"z_by_block": zlog, "groups": {}}
        for g in GROUPS:
            gm = oos & (group == g)
            gres = {"n_rows": int(gm.sum()), "n_stocks": int(pd.Series(syms[gm]).nunique()), "methods": {}}
            base_is = None
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
                log(f"h={h} {g:<4} {m:<12} cov50 {gres['methods'][m]['0.5']['coverage_pct']:.1f} cov80 {r8['coverage_pct']:.1f} "
                    f"cov95 {gres['methods'][m]['0.95']['coverage_pct']:.1f} width80 {r8['mean_width_pct']:.2f}% "
                    f"by vol-tercile {({k: round(v, 1) for k, v in r8['coverage_by_vol_tercile_pct'].items()})}")
        out["horizons"][str(h)] = hres

    elig = {g: {} for g in GROUPS}
    for g in GROUPS:
        for m in METHODS:
            ok = True
            for h in ("1", "5"):
                r8 = out["horizons"][h]["groups"][g]["methods"][m]["0.8"]
                ok &= abs(r8["coverage_pct"] - 80) <= 3 and all(abs(c - 80) <= 5 for c in r8["coverage_by_year_pct"].values())
            elig[g][m] = bool(ok)
    verdict = ("fhs_xgb eligible on Nifty 201-500: the 'never tested' note can be replaced with evidence"
               if elig["mid"]["fhs_xgb"] else
               "fhs_xgb NOT eligible on Nifty 201-500; mid-cap-specific calibration (fhs_xgb_grp) is eligible"
               if elig["mid"]["fhs_xgb_grp"] else
               "neither the shipped nor a group-calibrated method is eligible on Nifty 201-500: keep warnings")
    out["eligibility"], out["verdict"] = elig, verdict
    log("eligibility:", json.dumps(elig))
    log("VERDICT:", verdict)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--skip-fetch", action="store_true")
    a = ap.parse_args()
    t = time.time()
    out = run(a)
    out["runtime_minutes"] = round((time.time() - t) / 60, 1)
    sfx = "_quick" if a.quick else ""
    (REPORTS / f"interval_midcap_check{sfx}.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    log(f"wrote reports/interval_midcap_check{sfx}.json ({out['runtime_minutes']} min)")


if __name__ == "__main__":
    main()
