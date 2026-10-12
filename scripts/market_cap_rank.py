"""Snapshot of market capitalisation for every NSE stock outside the Nifty 500, to rank them by size.

Source: Yahoo Finance `fast_info.market_cap` (shares x last price, in INR). It is a point-in-time snapshot
taken on the day the script runs; it is NOT a historical ranking and Yahoo's share counts can be stale or
missing for some small caps (those are reported as `missing`, never guessed).

Resumable: re-running only fetches symbols that are not in reports/market_cap_snapshot.csv yet.

Usage: python scripts/market_cap_rank.py [--workers 4]
"""
from __future__ import annotations

import argparse
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd  # noqa: E402
import yfinance as yf  # noqa: E402
from sqlalchemy import text  # noqa: E402

from app.core.config import get_settings  # noqa: E402
from app.database.session import make_engine  # noqa: E402

OUT = ROOT / "reports" / "market_cap_snapshot.csv"


def one(sym: str) -> dict:
    for attempt in range(3):
        try:
            fi = yf.Ticker(sym + ".NS").fast_info
            mc = fi.get("marketCap")
            return {"symbol": sym, "market_cap_inr": float(mc) if mc else None,
                    "last_price": float(fi.get("lastPrice")) if fi.get("lastPrice") else None, "status": "ok" if mc else "missing"}
        except Exception as exc:  # noqa: BLE001 - recorded per symbol
            if "429" in str(exc) or "Too Many" in str(exc):
                time.sleep(20 * (attempt + 1))
                continue
            return {"symbol": sym, "market_cap_inr": None, "last_price": None, "status": f"error:{type(exc).__name__}"}
    return {"symbol": sym, "market_cap_inr": None, "last_price": None, "status": "error:rate_limited"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()
    n500 = set(pd.read_csv(ROOT / "app/data/nifty500.csv")["Symbol"])
    engine = make_engine(get_settings().resolved_database_url)
    with engine.connect() as c:
        syms = [r[0] for r in c.execute(text("select symbol from stocks where active = 1 order by symbol"))]
    todo_all = [s for s in syms if s not in n500]
    done = pd.read_csv(OUT) if OUT.exists() else pd.DataFrame(columns=["symbol", "market_cap_inr", "last_price", "status", "as_of"])
    todo = [s for s in todo_all if s not in set(done["symbol"])]
    print(f"{len(todo_all)} stocks outside the Nifty 500; {len(todo)} to fetch", flush=True)
    rows, t0 = [], time.time()
    with ThreadPoolExecutor(args.workers) as ex:
        for i, r in enumerate(ex.map(one, todo), 1):
            r["as_of"] = str(date.today())
            rows.append(r)
            if i % 100 == 0 or i == len(todo):
                pd.concat([done, pd.DataFrame(rows)]).to_csv(OUT, index=False)
                print(f"  {i}/{len(todo)}  {time.time() - t0:.0f}s", flush=True)
    full = pd.concat([done, pd.DataFrame(rows)]).drop_duplicates("symbol", keep="last")
    full.to_csv(OUT, index=False)
    print(full["status"].value_counts().to_string(), flush=True)


if __name__ == "__main__":
    main()
