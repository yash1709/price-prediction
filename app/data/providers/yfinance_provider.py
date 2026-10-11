"""Yahoo Finance provider (end-of-day, delayed data; unofficial API, no credentials)."""
from __future__ import annotations

import logging
import time
from datetime import date, datetime, timedelta, timezone

import pandas as pd

from app.core.errors import ProviderError, ProviderRateLimitError, SymbolNotFoundError
from app.core.symbols import normalize_symbol, to_yahoo_ticker
from app.data.providers.base import CANONICAL_COLUMNS, DataProvider, ProviderResult

log = logging.getLogger(__name__)

_RENAME = {"Open": "open", "High": "high", "Low": "low", "Close": "close",
           "Adj Close": "adj_close", "Volume": "volume"}


class YFinanceProvider(DataProvider):
    name = "yfinance"

    def __init__(self, retries: int = 2, backoff_seconds: float = 1.5):
        self.retries = retries
        self.backoff = backoff_seconds

    def fetch_daily(self, symbol: str, start: date, end: date | None = None) -> ProviderResult:
        import yfinance as yf  # imported lazily so tests/other providers don't need it

        try:  # surface real errors (rate limits, delisting) instead of silently empty frames
            yf.config.debug.hide_exceptions = False
        except AttributeError:  # older yfinance: fall back to the per-call flag below
            pass

        symbol = normalize_symbol(symbol)
        ticker_name = to_yahoo_ticker(symbol)
        end_excl = (end or date.today()) + timedelta(days=1)  # yfinance `end` is exclusive
        last_exc: Exception | None = None
        for attempt in range(self.retries + 1):
            try:
                tk = yf.Ticker(ticker_name)
                raw = tk.history(start=start.isoformat(), end=end_excl.isoformat(),
                                 interval="1d", auto_adjust=False, actions=False)
                meta = {}
                try:
                    meta = tk.history_metadata or {}
                except Exception:  # metadata is best-effort; validation handles absence
                    log.debug("history_metadata unavailable for %s", ticker_name)
                return self._to_result(symbol, ticker_name, raw, meta)
            except SymbolNotFoundError:
                raise
            except Exception as exc:  # noqa: BLE001 - classified below
                last_exc = exc
                text = f"{type(exc).__name__}: {exc}"
                low = text.lower()
                if "ratelimit" in low or "too many requests" in low or "429" in low:
                    err: Exception = ProviderRateLimitError(
                        f"Yahoo Finance rate-limited the request for {ticker_name}. Retry later.")
                elif any(k in low for k in ("delisted", "no data found", "no price data",
                                            "not found", "symbol may be delisted", "404")):
                    raise SymbolNotFoundError(
                        f"No data for '{symbol}' ({ticker_name}); it may be invalid, delisted, "
                        f"or unavailable from this provider.") from exc
                else:
                    err = ProviderError(f"Failed to fetch {ticker_name}: {text}")
                last_exc = err
                if attempt < self.retries:
                    time.sleep(self.backoff * (attempt + 1))
        assert last_exc is not None
        raise last_exc

    @staticmethod
    def _to_result(symbol: str, ticker_name: str, raw: pd.DataFrame, meta: dict) -> ProviderResult:
        if raw is None or raw.empty:
            raise SymbolNotFoundError(
                f"No price data returned for '{symbol}' ({ticker_name}) in the requested range.")
        df = raw.rename(columns=_RENAME)
        missing = [c for c in CANONICAL_COLUMNS if c not in df.columns]
        if missing:
            raise ProviderError(f"Provider response for {ticker_name} lacks columns: {missing}")
        df = df[CANONICAL_COLUMNS].copy()
        idx = pd.DatetimeIndex(df.index)
        if idx.tz is not None:
            idx = idx.tz_localize(None)
        df.index = idx.normalize()
        df.index.name = "date"
        return ProviderResult(
            symbol=symbol, frame=df, source="yfinance",
            fetched_at=datetime.now(timezone.utc).replace(tzinfo=None),
            currency=meta.get("currency"), exchange=meta.get("exchangeName"),
            reported_symbol=meta.get("symbol"),
            notes=["End-of-day data from Yahoo Finance; may be delayed and is not real-time."],
        )
