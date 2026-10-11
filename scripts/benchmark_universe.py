"""Multi-stock benchmark: train/select on history, score frozen models on a held-out recent
window, then retrain on all data and store the production forecasts.

Honest-accuracy protocol (per stock, per horizon h):
  1. Hold out the most recent `--holdout` sessions (default 126 ~ 6 months).
  2. Run the normal pipeline (walk-forward evaluation + selection) on data BEFORE the holdout only,
     in a throw-away in-memory DB.
  3. Score every frozen model (and the selected one) on the holdout rows. Nothing in steps 1-2 has
     seen the holdout, so selection bias is excluded from the reported accuracy.
  4. Retrain on ALL data in the main DB and store genuine forecasts (`predictions` table).

Usage (from project root, venv active):
  python scripts/benchmark_universe.py --limit 200 --workers 4 --horizons 1 5
  python scripts/benchmark_universe.py --summarize-only      # rebuild the report from results
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import tempfile
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("MODEL_N_JOBS", "1")  # one thread per model: parallelism is across stocks

import numpy as np  # noqa: E402
import requests  # noqa: E402

REPORTS = ROOT / "reports"
RESULTS = REPORTS / "benchmark_results.jsonl"
INDEX_URL = "https://www.niftyindices.com/IndexConstituent/ind_nifty200list.csv"
UA = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}


# ------------------------------------------------------------------------------------------
# worker
# ------------------------------------------------------------------------------------------
def process_symbol(symbol: str, horizons: list[int], holdout: int) -> dict:
    from app.core.errors import AppError, ProviderError, ProviderRateLimitError
    from app.database.repository import Repository  # noqa: F401
    from app.features.engineering import build_dataset
    from app.models.evaluation import compute_metrics
    from app.models.training import load_artifact, train_symbol
    from app.services.context import build_context
    from app.services.workflow import fetch_data, run_prediction, run_training

    rec: dict = {"symbol": symbol, "status": "ok", "horizons": {}}
    t0 = time.time()
    time.sleep(random.uniform(0, 3))  # stagger first requests
    ctx = build_context()
    try:
        # ---- 1. data (with back-off on rate limits) ----
        for attempt in range(4):
            try:
                st = fetch_data(ctx, symbol)
                break
            except (ProviderRateLimitError, ProviderError) as exc:
                if attempt == 3:
                    raise
                time.sleep(15 * (attempt + 1))
        with ctx.repo() as repo:
            name = repo.get_stock(symbol).name
            df = repo.load_bars(symbol)
        rec.update(name=name, n_bars=len(df), first_date=str(df.index[0].date()),
                   last_date=str(df.index[-1].date()), data_warnings=st.warnings)
        s = ctx.settings
        need = holdout + 50 + s.min_train_rows + s.walk_forward_test_size + max(horizons) + 5
        if len(df) < need:
            rec.update(status="skipped", reason=f"insufficient history ({len(df)} < {need} sessions)")
            return rec

        # ---- 2/3. hold-out experiment in a throw-away DB ----
        cutoff = df.index[-holdout - 1]
        rec["holdout"] = {"cutoff": str(cutoff.date()), "sessions": holdout}
        with tempfile.TemporaryDirectory() as tmp:
            xs = s.model_copy(update={"database_url": "sqlite:///:memory:",
                                      "artifacts_dir": Path(tmp)})
            xctx = build_context(xs)
            try:
                with xctx.repo() as xrepo:
                    xrepo.ensure_stock(symbol)
                    xrepo.upsert_bars(symbol, df.iloc[:-holdout], "benchmark")
                    xrepo.s.commit()
                    run = train_symbol(xrepo, xs, symbol, horizons=horizons)
                    for h in horizons:
                        ds = build_dataset(df, h)  # features are causal; target known for all rows
                        mask = ds.X.index > cutoff
                        entry = {"holdout": {}, "validation": {}, "n_test": int(mask.sum())}
                        for mv in xrepo.list_model_versions(symbol, horizon=h, run_id=run.id):
                            art = load_artifact(mv.artifact_path)
                            Xt = ds.X.loc[mask, art["feature_names"]]
                            pred = ds.ref_price[mask].values * (
                                1 + np.asarray(art["estimator"].predict(Xt), dtype=float))
                            entry["holdout"][mv.algorithm] = compute_metrics(
                                ds.actual_price[mask], pred, ds.ref_price[mask])
                            entry["validation"][mv.algorithm] = mv.metrics
                            if mv.is_selected:
                                entry["selected"] = mv.algorithm
                        rec["horizons"][str(h)] = entry
            finally:
                xctx.tasks.shutdown()

        # ---- 4. production: retrain on all data and store forecasts ----
        run_training(ctx, symbol, None, horizons, None, None, refresh_data=False)
        fc = run_prediction(ctx, symbol, horizons, refresh_data=False)
        for p in fc["predictions"]:
            rec["horizons"][str(p["horizon"])]["forecast"] = {
                k: p[k] for k in ("as_of_date", "target_date_estimate", "reference_close",
                                  "predicted_price", "predicted_change_pct", "algorithm",
                                  "model_version")}
        return rec
    except AppError as exc:
        rec.update(status="error", reason=f"{exc.code}: {exc.message}"[:300])
        return rec
    except Exception as exc:  # noqa: BLE001 - recorded in the results file, never hidden
        rec.update(status="error", reason=f"{type(exc).__name__}: {exc}"[:300],
                   trace=traceback.format_exc()[-800:])
        return rec
    finally:
        rec["seconds"] = round(time.time() - t0, 1)
        ctx.tasks.shutdown()


# ------------------------------------------------------------------------------------------
# aggregation
# ------------------------------------------------------------------------------------------
def _sign_test(wins: int, losses: int) -> float | None:
    n = wins + losses
    if n == 0:
        return None
    from scipy.stats import binomtest
    return float(binomtest(wins, n, 0.5).pvalue)


def _agg_algo(rows: list[dict], algo: str, key: str) -> dict | None:
    """Aggregate one algorithm across stocks. rows = per-stock horizon entries."""
    pairs = [(e[key][algo], e[key]["naive"]) for e in rows if algo in e.get(key, {})]
    if not pairs:
        return None
    mape = [m["mape"] for m, _ in pairs if m["mape"] is not None]
    imp = [(n["mae"] - m["mae"]) / n["mae"] * 100 for m, n in pairs if n["mae"] > 0]
    wins = sum(m["mae"] < n["mae"] for m, n in pairs)
    losses = sum(m["mae"] > n["mae"] for m, n in pairs)
    n_dir = sum(m["n_directional"] for m, _ in pairs)
    correct = sum(round((m["directional_accuracy"] or 0) * m["n_directional"]) for m, _ in pairs)
    return {
        "n_stocks": len(pairs),
        "mean_mape_pct": float(np.mean(mape)), "median_mape_pct": float(np.median(mape)),
        "median_mae_vs_naive_pct": float(np.median(imp)), "mean_mae_vs_naive_pct": float(np.mean(imp)),
        "stocks_beating_naive": int(wins), "stocks_losing_to_naive": int(losses),
        "pct_stocks_beating_naive": wins / len(pairs) * 100,
        "sign_test_p": _sign_test(wins, losses),
        "directional_accuracy_pct": (correct / n_dir * 100) if n_dir else None,
        "directional_n": int(n_dir),
    }


def summarize(records: list[dict]) -> dict:
    ok = [r for r in records if r["status"] == "ok"]
    out: dict = {
        "n_requested": len(records), "n_ok": len(ok),
        "n_skipped": sum(r["status"] == "skipped" for r in records),
        "n_error": sum(r["status"] == "error" for r in records),
        "skipped_or_failed": [{"symbol": r["symbol"], "status": r["status"],
                               "reason": r.get("reason")} for r in records if r["status"] != "ok"],
        "holdout_sessions": ok[0]["holdout"]["sessions"] if ok else None,
        "horizons": {},
    }
    horizons = sorted({h for r in ok for h in r["horizons"]}, key=int)
    for h in horizons:
        entries = [r["horizons"][h] for r in ok if h in r["horizons"] and "selected" in r["horizons"][h]]
        algos = sorted({a for e in entries for a in e["holdout"]})
        def selected_vs_naive(es):  # holdout metrics of each stock's selected model, with naive
            return [{"holdout": {"sel": e["holdout"][e["selected"]], "naive": e["holdout"]["naive"]}}
                    for e in es]

        nonnaive = [e for e in entries if e["selected"] != "naive"]
        sys_all = _agg_algo(selected_vs_naive(entries), "sel", "holdout")
        sys_nn = _agg_algo(selected_vs_naive(nonnaive), "sel", "holdout") if nonnaive else None
        out["horizons"][h] = {
            "n_stocks": len(entries),
            "selected_counts": {a: sum(e["selected"] == a for e in entries) for a in algos},
            "selected_non_naive": len(nonnaive),
            "system_all_stocks_holdout": sys_all,
            "system_when_non_naive_selected_holdout": sys_nn,
            "per_algorithm_holdout": {a: _agg_algo(entries, a, "holdout") for a in algos},
            "per_algorithm_validation": {a: _agg_algo(entries, a, "validation") for a in algos},
            "median_holdout_test_rows": float(np.median([e["n_test"] for e in entries])),
        }
    return out


def write_reports(records: list[dict], summary: dict) -> None:
    REPORTS.mkdir(exist_ok=True)
    (REPORTS / "benchmark_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    # forecasts
    lines = ["symbol,name,horizon_days,as_of_date,target_date_estimate,reference_close,"
             "predicted_price,predicted_change_pct,model,holdout_selected_vs_naive_mae_pct"]
    for r in records:
        if r["status"] != "ok":
            continue
        for h, e in sorted(r["horizons"].items(), key=lambda kv: int(kv[0])):
            f = e.get("forecast")
            if not f:
                continue
            # holdout skill of the SAME algorithm that produced this forecast (trained pre-holdout)
            n, m = e["holdout"]["naive"], e["holdout"][f["algorithm"]]
            rel = (n["mae"] - m["mae"]) / n["mae"] * 100 if n["mae"] else 0.0
            nm = (r.get("name") or "").replace(",", " ")
            lines.append(f"{r['symbol']},{nm},{h},{f['as_of_date']},{f['target_date_estimate']},"
                         f"{f['reference_close']:.2f},{f['predicted_price']:.2f},"
                         f"{f['predicted_change_pct']:.3f},{f['algorithm']},{rel:.2f}")
    (REPORTS / "forecasts.csv").write_text("\n".join(lines) + "\n", encoding="utf-8")

    def f(v, d=2, suf=""):
        return "n/a" if v is None else f"{v:.{d}f}{suf}"

    md = ["# Nifty 200 benchmark report", "",
          f"Stocks requested: {summary['n_requested']} · evaluated: {summary['n_ok']} · "
          f"skipped: {summary['n_skipped']} · errors: {summary['n_error']}",
          f"Held-out window: last {summary['holdout_sessions']} sessions per stock (never seen by "
          "training or model selection).", ""]
    for h, s in summary["horizons"].items():
        sa = s["system_all_stocks_holdout"]
        md += [f"## Horizon: {h} trading day(s)", "",
               f"Stocks: {s['n_stocks']} · selected model counts: {s['selected_counts']}", "",
               "**Full system (the model selected for each stock), held-out window**", "",
               f"* Median MAPE: {f(sa['median_mape_pct'], 2, '%')} (mean {f(sa['mean_mape_pct'], 2, '%')})",
               f"* Median MAE improvement vs naive: {f(sa['median_mae_vs_naive_pct'], 2, '%')}",
               f"* Where the pipeline's pick differed from naive: it beat naive on MAE in "
               f"{sa['stocks_beating_naive']} stocks and lost in {sa['stocks_losing_to_naive']} "
               f"(two-sided sign test p = {f(sa['sign_test_p'], 3)}; more losses than wins means "
               f"the selection did not generalise)", ""]
        nn = s["system_when_non_naive_selected_holdout"]
        if nn:
            md += [f"Where a non-naive model was selected ({s['selected_non_naive']} stocks): "
                   f"median MAE vs naive {f(nn['median_mae_vs_naive_pct'], 2, '%')}, beat naive in "
                   f"{nn['stocks_beating_naive']}/{nn['n_stocks']} (p = {f(nn['sign_test_p'], 3)}), "
                   f"directional accuracy {f(nn['directional_accuracy_pct'], 1, '%')} "
                   f"(n={nn['directional_n']}).", ""]
        md += ["**Each algorithm used on every stock (held-out window)**", "",
               "Median MAE vs naive: negative = worse than naive. Sign test is two-sided on "
               "wins vs losses against naive; a small p with more losses than wins means the "
               "model is *significantly worse* than naive.", "",
               "| Algorithm | Median MAPE % | Median MAE vs naive % | Wins vs naive | Losses vs naive | Sign-test p (two-sided) | Directional acc. |",
               "|---|---|---|---|---|---|---|"]
        for a, m in s["per_algorithm_holdout"].items():
            md.append(f"| {a} | {f(m['median_mape_pct'])} | {f(m['median_mae_vs_naive_pct'])} | "
                      f"{m['stocks_beating_naive']} | {m['stocks_losing_to_naive']} | "
                      f"{f(m['sign_test_p'], 3)} | {f(m['directional_accuracy_pct'], 1, '%')} |")
        md += ["", "Validation (walk-forward, before the held-out window) for comparison:", "",
               "| Algorithm | Median MAE vs naive % | Stocks beating naive |", "|---|---|---|"]
        for a, m in s["per_algorithm_validation"].items():
            md.append(f"| {a} | {f(m['median_mae_vs_naive_pct'])} | "
                      f"{m['stocks_beating_naive']}/{m['n_stocks']} |")
        md.append("")
    (REPORTS / "benchmark_report.md").write_text("\n".join(md), encoding="utf-8")


# ------------------------------------------------------------------------------------------
# main
# ------------------------------------------------------------------------------------------
def load_records() -> list[dict]:
    if not RESULTS.exists():
        return []
    latest: dict[str, dict] = {}
    for line in RESULTS.read_text(encoding="utf-8").splitlines():
        if line.strip():
            r = json.loads(line)
            latest[r["symbol"]] = r
    return list(latest.values())


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--source", default=INDEX_URL, help="NSE index constituent CSV (URL or path)")
    ap.add_argument("--limit", type=int, default=200)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--horizons", type=int, nargs="+", default=[1, 5])
    ap.add_argument("--holdout", type=int, default=126)
    ap.add_argument("--retry-errors", action="store_true")
    ap.add_argument("--summarize-only", action="store_true")
    a = ap.parse_args()
    REPORTS.mkdir(exist_ok=True)

    if not a.summarize_only:
        from app.data.universe import parse_equity_list
        from app.services.context import build_context
        from app.services.workflow import import_universe_source, seed_universe_if_empty
        text = (requests.get(a.source, headers=UA, timeout=60).text
                if a.source.startswith("http") else Path(a.source).read_text(encoding="utf-8-sig"))
        local = ROOT / "data" / "universe_index.csv"
        local.write_text(text, encoding="utf-8")
        symbols = [r["symbol"] for r in parse_equity_list(text, "nse_import")][: a.limit]
        ctx = build_context()
        seed_universe_if_empty(ctx)
        print("universe import:", import_universe_source(ctx, str(local)), flush=True)
        ctx.tasks.shutdown()

        done = {r["symbol"]: r for r in load_records()}
        todo = [s for s in symbols if s not in done or (a.retry_errors and done[s]["status"] == "error")]
        print(f"{len(symbols)} symbols, {len(symbols) - len(todo)} already done, {len(todo)} to run "
              f"on {a.workers} workers", flush=True)
        t0, finished = time.time(), 0
        with ProcessPoolExecutor(max_workers=a.workers) as pool, \
                open(RESULTS, "a", encoding="utf-8") as fh:
            futs = {pool.submit(process_symbol, s, a.horizons, a.holdout): s for s in todo}
            for fut in as_completed(futs):
                sym = futs[fut]
                try:
                    rec = fut.result()
                except Exception as exc:  # worker crash
                    rec = {"symbol": sym, "status": "error", "reason": f"worker crash: {exc}"}
                fh.write(json.dumps(rec) + "\n")
                fh.flush()
                finished += 1
                el = time.time() - t0
                eta = el / finished * (len(todo) - finished)
                sel = {h: e.get("selected") for h, e in rec.get("horizons", {}).items()}
                print(f"[{finished}/{len(todo)}] {sym:<12} {rec['status']:<8} {sel or rec.get('reason', '')} "
                      f"({rec.get('seconds', 0)}s; elapsed {el / 60:.1f}m, eta {eta / 60:.1f}m)", flush=True)

    records = [r for r in load_records()]
    summary = summarize(records)
    write_reports(records, summary)
    print(f"\nWrote {REPORTS / 'benchmark_report.md'}, benchmark_summary.json, forecasts.csv", flush=True)


if __name__ == "__main__":
    main()
