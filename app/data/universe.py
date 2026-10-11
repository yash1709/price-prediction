"""Stock-universe management. The bundled seed list is a SMALL STARTER set, not all NSE stocks.

Import the full maintained list with `python run.py import-universe` (NSE's EQUITY_L.csv, from a
URL or a local file you downloaded from nseindia.com).
"""
from __future__ import annotations

import io
import logging
from pathlib import Path

import pandas as pd
import requests

from app.core.errors import ProviderError
from app.core.symbols import normalize_symbol
from app.database.repository import Repository

log = logging.getLogger(__name__)
SEED_FILE = Path(__file__).with_name("seed_universe.csv")
_BROWSER_HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}


def parse_equity_list(csv_text: str, source: str) -> list[dict]:
    """Parse NSE EQUITY_L.csv (or any CSV with SYMBOL / NAME OF COMPANY columns)."""
    df = pd.read_csv(io.StringIO(csv_text))
    df.columns = [c.strip().upper() for c in df.columns]
    if "SYMBOL" not in df.columns:
        raise ProviderError("Universe file must contain a 'SYMBOL' column.")
    name_col = next((c for c in df.columns if c.startswith("NAME") or c == "COMPANY NAME"), None)
    series_col = "SERIES" if "SERIES" in df.columns else None
    isin_col = next((c for c in df.columns if "ISIN" in c), None)
    rows: dict[str, dict] = {}
    for _, r in df.iterrows():
        try:
            sym = normalize_symbol(str(r["SYMBOL"]))
        except Exception:  # noqa: BLE001 - skip malformed symbols but keep going
            log.warning("Skipping invalid symbol in universe file: %r", r["SYMBOL"])
            continue
        rows[sym] = {
            "symbol": sym,
            "name": str(r[name_col]).strip() if name_col and pd.notna(r[name_col]) else None,
            "series": str(r[series_col]).strip() if series_col and pd.notna(r[series_col]) else None,
            "isin": str(r[isin_col]).strip() if isin_col and pd.notna(r[isin_col]) else None,
            "source": source,
        }
    return list(rows.values())


def load_seed(repo: Repository) -> int:
    return repo.upsert_stocks(parse_equity_list(SEED_FILE.read_text(encoding="utf-8"), "seed"))


def import_universe(repo: Repository, source: str, timeout: float = 30.0) -> tuple[int, int]:
    """Import from an http(s) URL or local path. Returns (rows_in_file, newly_inserted)."""
    if source.lower().startswith(("http://", "https://")):
        try:
            resp = requests.get(source, headers=_BROWSER_HEADERS, timeout=timeout)
            resp.raise_for_status()
        except requests.RequestException as exc:
            raise ProviderError(
                f"Could not download the universe list ({exc}). Download EQUITY_L.csv from "
                f"nseindia.com manually and pass its local path instead.") from exc
        text = resp.text
    else:
        path = Path(source)
        if not path.exists():
            raise ProviderError(f"Universe file not found: {path}")
        text = path.read_text(encoding="utf-8-sig")
    rows = parse_equity_list(text, "nse_import")
    if not rows:
        raise ProviderError("Universe file contained no valid symbols.")
    return len(rows), repo.upsert_stocks(rows)
