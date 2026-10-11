"""Local-CSV provider: offline fallback and a template for custom/licensed feeds.

Place `<SYMBOL>.csv` in the CSV directory with columns:
Date, Open, High, Low, Close, Adj Close (optional), Volume
"""
from __future__ import annotations

from datetime import date, datetime, timezone
from pathlib import Path

import pandas as pd

from app.core.errors import ProviderError, SymbolNotFoundError
from app.core.symbols import normalize_symbol
from app.data.providers.base import CANONICAL_COLUMNS, DataProvider, ProviderResult


class CSVProvider(DataProvider):
    name = "csv"

    def __init__(self, directory: Path):
        self.directory = Path(directory)

    def fetch_daily(self, symbol: str, start: date, end: date | None = None) -> ProviderResult:
        symbol = normalize_symbol(symbol)
        path = self.directory / f"{symbol}.csv"
        if not path.exists():
            raise SymbolNotFoundError(f"No CSV file for '{symbol}' at {path}.")
        try:
            raw = pd.read_csv(path)
        except Exception as exc:  # noqa: BLE001
            raise ProviderError(f"Cannot read {path}: {exc}") from exc
        raw.columns = [c.strip().lower().replace(" ", "_") for c in raw.columns]
        if "date" not in raw.columns:
            raise ProviderError(f"{path} must have a 'Date' column.")
        if "adj_close" not in raw.columns:
            raw["adj_close"] = raw.get("close")
        missing = [c for c in CANONICAL_COLUMNS if c not in raw.columns]
        if missing:
            raise ProviderError(f"{path} lacks columns: {missing}")
        raw["date"] = pd.to_datetime(raw["date"], errors="coerce")
        df = raw.dropna(subset=["date"]).set_index("date")[CANONICAL_COLUMNS].sort_index()
        df.index = df.index.normalize()
        df.index.name = "date"
        df = df[df.index >= pd.Timestamp(start)]
        if end is not None:
            df = df[df.index <= pd.Timestamp(end)]
        if df.empty:
            raise SymbolNotFoundError(f"{path} has no rows in the requested range.")
        return ProviderResult(
            symbol=symbol, frame=df, source=f"csv:{path.name}",
            fetched_at=datetime.now(timezone.utc).replace(tzinfo=None), currency="INR",
            reported_symbol=symbol, notes=["Data read from a local CSV file."],
        )
