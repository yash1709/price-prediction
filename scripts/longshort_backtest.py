"""Cost-aware long-short backtest of the pooled cross-sectional signal.

Pre-registered (fixed before running):
  signal   : equal-weight average of cross-sectional ranks of pooled ridge / xgb-L2 / xgb-L1
  refit    : every 126 sessions on all prior data (purged by h+1); no selection, fixed hyperparameters
  strategy : top quintile long / bottom quintile short, equal weight, dollar-neutral
             daily (h=1 signal, rebalance every session) and weekly (h=5 signal, every 5 sessions)
  primary  : execution delayed one session (trade the close AFTER the signal), 10 bps per unit traded
  success  : net mean return > 0 with Newey-West t > 2.24 (Bonferroni for 2 strategies) AND positive in
             >= 60 % of calendar years
Also reported: costs 0..30 bps, break-even cost, long-only (shorting cash equities is not available to
Indian retail), per-year results, and a shuffled-signal placebo that validates the statistics.

Usage: python scripts/longshort_backtest.py [--quick]   (needs the benchmark price cache)
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

from pooled_experiment import build_panel, log, make_models, nw_t  # noqa: E402

REPORTS = ROOT / "reports"
BLOCK = 126
START_POS = 750          # >= ~3 years of training before the first out-of-sample date
COSTS_BPS = [0, 5, 10, 20, 30]
PRIMARY_COST = 10
T_THRESHOLD = 2.24
MIN_POSITIVE_YEARS = 0.6


# ------------------------------------------------------------------------------------------
def prepare(quick: bool):
    res = [json.loads(l) for l in (REPORTS / "benchmark_results.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    symbols = sorted({r["symbol"] for r in res if r["status"] == "ok"})
    names = {r["symbol"]: r.get("name") for r in res}
    if quick:
        symbols = symbols[:40]
    idx = pd.read_csv(ROOT / "data" / "universe_index.csv")
    sectors = dict(zip(idx["Symbol"], idx["Industry"]))
    log(f"{len(symbols)} stocks; building panel…")
    panel = build_panel(symbols, sectors, [1, 5])
    feat = [c for c in panel.columns if c not in ("adj", "symbol") and not c.startswith("y")]
    panel = panel.dropna(subset=feat)
    n = panel.groupby(level="date").size()
    dates = n[n >= 30].index
    panel = panel[panel.index.get_level_values("date").isin(dates)]
    log(f"panel: {len(panel):,} rows, {len(feat)} features, {len(dates)} dates "
        f"({dates[0].date()} → {dates[-1].date()})")
    return panel, feat, pd.DatetimeIndex(dates), names


def walk_forward_scores(panel, feat, dates, h, quick) -> pd.DataFrame:
    """Out-of-sample prediction per (date, symbol) for every model, refit each BLOCK sessions."""
    pos = pd.Series(np.arange(len(dates)), index=dates)
    P = pos.reindex(panel.index.get_level_values("date")).values
    N = len(dates)
    X = panel[feat].astype("float32")
    y = panel[f"y{h}"].values
    yfin = ~np.isnan(y)
    models = make_models(quick)
    out = {m: np.full(len(panel), np.nan) for m in models}
    t0, k = START_POS, 0
    while t0 < N:
        t1 = min(t0 + BLOCK, N)
        train = (P <= t0 - h - 1) & yfin
        test = (P >= t0) & (P < t1)
        lo, hi = np.quantile(y[train], [0.005, 0.995])
        for name, mk in models.items():
            m = mk()
            m.fit(X[train], np.clip(y[train], lo, hi))
            out[name][test] = np.asarray(m.predict(X[test]), dtype=float)
        k += 1
        log(f"h={h} refit {k}: train rows {train.sum():,}, test dates "
            f"{dates[t0].date()}..{dates[t1 - 1].date()}")
        t0 = t1
    df = pd.DataFrame(out, index=panel.index)
    return df


def to_scores(preds: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Wide (date x symbol) score frames: ensemble of per-date ranks, plus each single model."""
    ranks = preds.groupby(level="date").rank(pct=True)
    scores = {"ensemble": ranks.mean(axis=1).where(preds.notna().all(axis=1)).unstack("symbol")}
    for m in preds.columns:
        scores[m] = preds[m].unstack("symbol")
    return scores


# ------------------------------------------------------------------------------------------
def make_weights(row: pd.Series, kind: str) -> pd.Series | None:
    s = row.dropna()
    if len(s) < 30:
        return None
    r = s.rank(pct=True)
    top, bot = r[r > 0.8].index, r[r <= 0.2].index
    if len(top) < 5 or (kind == "ls" and len(bot) < 5):  # heavily tied scores: no trade that day
        return None
    w = pd.Series(1.0 / len(top), index=top)
    if kind == "ls":
        w = pd.concat([w, pd.Series(-1.0 / len(bot), index=bot)])
    return w


def backtest(S: pd.DataFrame, R: pd.DataFrame, dates, h: int, delay: int, every: int,
             kind: str) -> pd.DataFrame:
    """Hold from `delay` sessions after each signal date; realised return = h-period forward return."""
    N = len(dates)
    S = S.reindex(dates)
    R = R.reindex(dates)
    rows, prev = [], None
    for r in range(START_POS, N, every):
        d = r + delay
        if d + h > N - 1:
            break
        w = make_weights(S.iloc[r], kind)
        if w is None:
            continue
        ret = R.iloc[d]
        ok = ret.reindex(w.index).notna().values
        w = w[ok]
        if (w > 0).sum() < 5 or (kind == "ls" and (w < 0).sum() < 5):
            continue
        w[w > 0] /= w[w > 0].sum()
        if (w < 0).any():
            w[w < 0] /= -w[w < 0].sum()
        gross = float((w * ret.reindex(w.index)).sum())
        turn = float(w.sub(prev, fill_value=0).abs().sum()) if prev is not None else float(w.abs().sum())
        prev = w
        rows.append({"date": dates[d], "gross": gross, "turnover": turn, "universe": float(ret.mean())})
    return pd.DataFrame(rows)


def stats(df: pd.DataFrame, cost_bps: float, every: int, excess: bool = False) -> dict:
    net = df["gross"] - cost_bps / 1e4 * df["turnover"]
    if excess:
        net = net - df["universe"]
    ppy = 252 / every
    mean, sd = net.mean(), net.std()
    eq = (1 + net).cumprod()
    yearly = net.groupby(df["date"].dt.year).sum() * (ppy / net.groupby(df["date"].dt.year).size())
    return {
        "periods": int(len(net)), "ann_return_pct": float(mean * ppy * 100),
        "ann_vol_pct": float(sd * np.sqrt(ppy) * 100), "sharpe": float(mean / sd * np.sqrt(ppy)) if sd > 0 else None,
        "nw_t": nw_t(net.values, lag=5 if every == 1 else 2),
        "max_drawdown_pct": float((eq / eq.cummax() - 1).min() * 100),
        "avg_turnover_per_rebalance": float(df["turnover"].mean()),
        "gross_ann_return_pct": float(df["gross"].mean() * ppy * 100),
        "breakeven_cost_bps": float(df["gross"].mean() / df["turnover"].mean() * 1e4),
        "years_positive": f"{int((yearly > 0).sum())}/{len(yearly)}",
        "share_years_positive": float((yearly > 0).mean()),
        "yearly_ann_return_pct": {str(k): float(v * 100) for k, v in yearly.items()},
    }


def placebo(S: pd.DataFrame, R, dates, h, every, seeds=20) -> dict:
    """Shuffle the signal across stocks within each date: gross t-stats should look like N(0,1)."""
    ts, means = [], []
    for seed in range(seeds):
        rng = np.random.default_rng(seed)
        arr = S.reindex(dates).values.copy()
        for i in range(arr.shape[0]):
            m = ~np.isnan(arr[i])
            arr[i, m] = rng.permutation(arr[i, m])
        df = backtest(pd.DataFrame(arr, index=dates, columns=S.columns), R, dates, h, 1, every, "ls")
        st = stats(df, 0, every)
        ts.append(st["nw_t"])
        means.append(st["gross_ann_return_pct"])
    return {"seeds": seeds, "t_mean": float(np.mean(ts)), "t_std": float(np.std(ts)),
            "t_max_abs": float(np.max(np.abs(ts))), "gross_ann_return_pct_mean": float(np.mean(means))}


# ------------------------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--quick", action="store_true")
    a = ap.parse_args()
    t_start = time.time()
    panel, feat, dates, names = prepare(a.quick)
    out: dict = {"stocks": int(panel.index.get_level_values("symbol").nunique()),
                 "oos_start": str(dates[START_POS].date()), "oos_end": str(dates[-1].date()),
                 "cost_grid_bps": COSTS_BPS, "primary": {"cost_bps": PRIMARY_COST, "delay": 1,
                                                          "t_threshold": T_THRESHOLD},
                 "horizons": {}}
    equity = {}
    latest = []
    for h, every in ((1, 1), (5, 5)):
        preds = walk_forward_scores(panel, feat, dates, h, a.quick)
        scores = to_scores(preds)
        R = panel[f"y{h}"].unstack("symbol")
        hres: dict = {"variants": {}}
        for sname, S in scores.items():
            for delay in (0, 1):
                for kind in ("ls", "long"):
                    df = backtest(S, R, dates, h, delay, every, kind)
                    if df.empty:
                        continue
                    key = f"{sname}|delay{delay}|{kind}"
                    hres["variants"][key] = {
                        f"cost_{c}bps": stats(df, c, every, excess=(kind == "long")) for c in COSTS_BPS}
                    if sname == "ensemble" and delay == 1:
                        equity[f"h{h}_{kind}_net{PRIMARY_COST}bps"] = (
                            (df["gross"] - PRIMARY_COST / 1e4 * df["turnover"]).set_axis(df["date"]))
        prim = hres["variants"]["ensemble|delay1|ls"][f"cost_{PRIMARY_COST}bps"]
        passed = bool(prim["nw_t"] is not None and prim["nw_t"] > T_THRESHOLD and prim["ann_return_pct"] > 0
                      and prim["share_years_positive"] >= MIN_POSITIVE_YEARS)
        hres["primary_net_nw_t"] = prim["nw_t"]
        hres["criterion_passed"] = passed
        hres["placebo_gross_delay1"] = placebo(scores["ensemble"], R, dates, h, every)
        out["horizons"][str(h)] = hres
        log(f"h={h} PRIMARY (ensemble, delay 1, {PRIMARY_COST} bps): ann net {prim['ann_return_pct']:+.2f}% "
            f"Sharpe {prim['sharpe']:.2f} t={prim['nw_t']:.2f} years+ {prim['years_positive']} "
            f"turnover {prim['avg_turnover_per_rebalance']:.2f} breakeven {prim['breakeven_cost_bps']:.1f} bps "
            f"-> {'PASSED' if passed else 'NOT met'}")
        last = scores["ensemble"].iloc[-1].dropna().sort_values(ascending=False)
        latest += [{"horizon": h, "symbol": s, "name": names.get(s), "score_rank": float(v)}
                   for s, v in last.items()]
    out["runtime_minutes"] = round((time.time() - t_start) / 60, 1)
    suffix = "_quick" if a.quick else ""
    (REPORTS / f"longshort_backtest{suffix}.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    pd.DataFrame(equity).to_csv(REPORTS / f"longshort_net_returns{suffix}.csv")
    pd.DataFrame(latest).to_csv(REPORTS / f"latest_signal_ranking{suffix}.csv", index=False)
    log(f"wrote reports/longshort_backtest{suffix}.json ({out['runtime_minutes']} min)")


if __name__ == "__main__":
    main()
