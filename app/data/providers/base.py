"""Provider interface. Implement `DataProvider` to plug in an official/licensed market-data API."""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import date, datetime

import pandas as pd

# Canonical columns every provider must return (index = tz-naive trading date, ascending).
CANONICAL_COLUMNS = ["open", "high", "low", "close", "adj_close", "volume"]


@dataclass
class ProviderResult:
    symbol: str
    frame: pd.DataFrame  # index: DatetimeIndex named "date"; columns: CANONICAL_COLUMNS
    source: str
    fetched_at: datetime
    currency: str | None = None
    exchange: str | None = None
    reported_symbol: str | None = None  # symbol echoed by the upstream service, when available
    notes: list[str] = field(default_factory=list)


class DataProvider(ABC):
    name: str = "base"

    @abstractmethod
    def fetch_daily(self, symbol: str, start: date, end: date | None = None) -> ProviderResult:
        """Return end-of-day OHLCV for an NSE symbol (without suffix) between start and end.

        Must raise `SymbolNotFoundError`, `ProviderRateLimitError` or `ProviderError`
        and must never fabricate or fill missing prices.
        """
