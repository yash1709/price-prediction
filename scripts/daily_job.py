"""Run the daily forward-testing job (see app/services/daily.py).

  python scripts/daily_job.py                      # all Nifty 200 stocks -> ./live
  python scripts/daily_job.py --limit 10 --out tmp_live
  python scripts/daily_job.py --max-minutes 35

Exit code: 0 = ok/degraded, 2 = failed (fewer than half of the stocks could be fetched).
Run it after the NSE close (>= 16:00 IST); the GitHub Actions workflow does this on weekdays.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd  # noqa: E402
import requests  # noqa: E402

from app.core.config import get_settings  # noqa: E402
from app.core.logging import setup_logging  # noqa: E402
from app.data.providers.yfinance_provider import YFinanceProvider  # noqa: E402
from app.models.intervals import IntervalService  # noqa: E402
from app.services.daily import DailyConfig, run_daily  # noqa: E402

INDEX_URL = "https://www.niftyindices.com/IndexConstituent/ind_nifty200list.csv"
FALLBACK = ROOT / "app" / "data" / "nifty200.csv"


def load_universe(limit: int | None) -> list[tuple[str, str | None]]:
    """Official Nifty 200 list; falls back to the committed copy if the download fails."""
    try:
        r = requests.get(INDEX_URL, headers={"User-Agent": "Mozilla/5.0"}, timeout=30)
        r.raise_for_status()
        from io import StringIO
        df = pd.read_csv(StringIO(r.text))
        print(f"universe: downloaded {len(df)} rows from NSE Indices")
    except Exception as exc:  # noqa: BLE001 - reported, then the committed copy is used
        print(f"universe: download failed ({exc}); using committed copy")
        df = pd.read_csv(FALLBACK)
    pairs = list(zip(df["Symbol"].astype(str), df["Company Name"].astype(str)))
    return pairs[:limit] if limit else pairs


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--out", default="live", help="output directory (default ./live)")
    ap.add_argument("--limit", type=int, help="only the first N stocks (testing)")
    ap.add_argument("--max-minutes", type=float, default=None, help="soft time budget")
    ap.add_argument("--horizons", type=int, nargs="+", default=[1, 5])
    a = ap.parse_args()
    settings = get_settings()
    setup_logging(settings.log_level)
    summary = run_daily(YFinanceProvider(retries=2), load_universe(a.limit), Path(a.out),
                        IntervalService(settings), settings,
                        cfg=DailyConfig(horizons=tuple(a.horizons), max_minutes=a.max_minutes))
    sys.exit(2 if summary["status"] == "failed" else 0)


if __name__ == "__main__":
    main()
