"""Command-line entry point.

  python run.py init-db
  python run.py import-universe [--source PATH_OR_URL]
  python run.py fetch RELIANCE [--years 8]
  python run.py train RELIANCE [--horizons 1 5 10] [--algorithms ridge xgboost]
  python run.py predict RELIANCE [--horizons 1 5]
  python run.py update-outcomes RELIANCE
  python run.py api            # FastAPI on http://127.0.0.1:8000  (docs at /docs)
  python run.py dashboard      # Streamlit on http://localhost:8501
  python run.py all            # API + dashboard together
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from datetime import date, timedelta

from app.core.config import PROJECT_ROOT, get_settings
from app.core.errors import AppError
from app.core.logging import setup_logging


def _ctx():
    from app.services.context import build_context
    from app.services.workflow import seed_universe_if_empty
    ctx = build_context()
    seed_universe_if_empty(ctx)
    return ctx


def cmd_init_db(_: argparse.Namespace) -> None:
    ctx = _ctx()
    with ctx.repo() as repo:
        total, _rows = repo.search_stocks(None, 1, 0)
    print(f"Database ready at {get_settings().resolved_database_url} "
          f"({total} symbols in universe; seed list unless you ran import-universe).")


def cmd_import_universe(a: argparse.Namespace) -> None:
    from app.services.workflow import import_universe_source
    print(json.dumps(import_universe_source(_ctx(), a.source), indent=2))


def cmd_fetch(a: argparse.Namespace) -> None:
    from app.services.workflow import fetch_data
    start = date.today() - timedelta(days=int(365.25 * a.years)) if a.years else None
    st = fetch_data(_ctx(), a.symbol, start=start, force_refresh=a.refresh, full=a.full)
    print(json.dumps(st.to_dict(), indent=2))


def cmd_train(a: argparse.Namespace) -> None:
    from app.services.workflow import run_training
    res = run_training(_ctx(), a.symbol, a.algorithms, a.horizons, None, None,
                       progress=lambda f, m: print(f"[{f:5.0%}] {m}", flush=True))
    for h, info in res["summary"]["horizons"].items():
        print(f"\nHorizon {h}d  ->  selected: {info['selected']}  "
              f"(train rows {info['n_train_rows']}, OOS rows {info['n_oos_rows']})")
        print(f"  {'model':<14}{'MAE':>10}{'RMSE':>10}{'MAPE%':>8}{'DirAcc':>8}{'MAE vs naive':>14}")
        for name, m in info["metrics"].items():
            da = "n/a" if m["directional_accuracy"] is None else f"{m['directional_accuracy']:.1%}"
            imp = m["vs_naive"]["mae_improvement_pct"]
            print(f"  {name:<14}{m['mae']:>10.2f}{m['rmse']:>10.2f}{m['mape']:>8.2f}{da:>8}"
                  f"{imp:>+13.2f}%")


def cmd_predict(a: argparse.Namespace) -> None:
    from app.services.workflow import run_prediction
    res = run_prediction(_ctx(), a.symbol, a.horizons)
    for p in res["predictions"]:
        print(f"{p['symbol']} h={p['horizon']}d [{p['algorithm']}] as of {p['as_of_date']}: "
              f"ref ₹{p['reference_close']:.2f} -> forecast ₹{p['predicted_price']:.2f} "
              f"({p['predicted_change_pct']:+.2f}%) for ~{p['target_date_estimate']}")
    for w in res["warnings"]:
        print("WARNING:", w)
    print(res["disclaimer"])


def cmd_update_outcomes(a: argparse.Namespace) -> None:
    from app.services.outcomes import resolve_outcomes
    from app.services.workflow import fetch_data
    from app.core.symbols import normalize_symbol
    ctx, sym = _ctx(), normalize_symbol(a.symbol)
    fetch_data(ctx, sym, force_refresh=True)
    with ctx.repo() as repo:
        print(f"Recorded {resolve_outcomes(repo, sym)} new outcome(s); "
              f"{len(repo.unresolved_predictions(sym))} still pending.")


def cmd_api(a: argparse.Namespace) -> None:
    import uvicorn
    s = get_settings()
    uvicorn.run("app.api.main:app_factory", factory=True, host=a.host or s.api_host,
                port=a.port or s.api_port, reload=a.reload)


def _dashboard_cmd(port: int) -> list[str]:
    return [sys.executable, "-m", "streamlit", "run", str(PROJECT_ROOT / "frontend" / "dashboard.py"),
            "--server.port", str(port), "--server.headless", "true",
            "--browser.gatherUsageStats", "false"]


def cmd_dashboard(a: argparse.Namespace) -> None:
    sys.exit(subprocess.call(_dashboard_cmd(a.port), cwd=PROJECT_ROOT))


def cmd_all(a: argparse.Namespace) -> None:
    s = get_settings()
    api = subprocess.Popen([sys.executable, __file__, "api"], cwd=PROJECT_ROOT)
    time.sleep(2)
    dash = subprocess.Popen(_dashboard_cmd(a.port), cwd=PROJECT_ROOT)
    print(f"API: http://{s.api_host}:{s.api_port}/docs   Dashboard: http://localhost:{a.port}")
    try:
        api.wait()
    except KeyboardInterrupt:
        pass
    finally:
        for p in (api, dash):
            p.terminate()


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("init-db").set_defaults(fn=cmd_init_db)
    s = sub.add_parser("import-universe")
    s.add_argument("--source", help="URL or local path to NSE EQUITY_L.csv (default: UNIVERSE_URL)")
    s.set_defaults(fn=cmd_import_universe)
    s = sub.add_parser("fetch")
    s.add_argument("symbol")
    s.add_argument("--years", type=float, help="history length (default: DEFAULT_HISTORY_YEARS)")
    s.add_argument("--refresh", action="store_true", help="ignore cache freshness")
    s.add_argument("--full", action="store_true", help="re-download the full history")
    s.set_defaults(fn=cmd_fetch)
    s = sub.add_parser("train")
    s.add_argument("symbol")
    s.add_argument("--horizons", type=int, nargs="+", default=[1])
    s.add_argument("--algorithms", nargs="+")
    s.set_defaults(fn=cmd_train)
    s = sub.add_parser("predict")
    s.add_argument("symbol")
    s.add_argument("--horizons", type=int, nargs="+", default=[1])
    s.set_defaults(fn=cmd_predict)
    s = sub.add_parser("update-outcomes")
    s.add_argument("symbol")
    s.set_defaults(fn=cmd_update_outcomes)
    s = sub.add_parser("api")
    s.add_argument("--host")
    s.add_argument("--port", type=int)
    s.add_argument("--reload", action="store_true")
    s.set_defaults(fn=cmd_api)
    s = sub.add_parser("dashboard")
    s.add_argument("--port", type=int, default=8501)
    s.set_defaults(fn=cmd_dashboard)
    s = sub.add_parser("all")
    s.add_argument("--port", type=int, default=8501)
    s.set_defaults(fn=cmd_all)

    args = p.parse_args(argv)
    setup_logging(get_settings().log_level)
    try:
        args.fn(args)
    except AppError as exc:
        print(f"ERROR [{exc.code}]: {exc.message}", file=sys.stderr)
        sys.exit(2)


if __name__ == "__main__":
    main()
