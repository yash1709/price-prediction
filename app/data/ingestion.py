"""Cache-aware ingestion: decide whether to use cache, fetch increments, or refetch history."""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta

import pandas as pd

from app.core.config import Settings
from app.core.errors import AppError, DataUnavailableError
from app.core.symbols import normalize_symbol
from app.data.providers.base import DataProvider
from app.data.validation import clean_ohlcv, verify_symbol
from app.database.models import utcnow
from app.database.repository import Repository

log = logging.getLogger(__name__)

OVERLAP_DAYS = 7           # re-fetched days on incremental updates (catches revisions)
ADJ_MISMATCH_TOL = 1e-3    # relative adj_close change that signals a corporate action -> full refetch
PENDING_MAX_AGE_DAYS = 7   # a discarded newest bar older than this is not worth fast re-checks


def pending_bar(meta, today: date | None = None) -> date | None:
    """Date of a recent newest bar that the provider has not finalised yet (discarded at last fetch)."""
    raw = (meta.quality_report or {}).get("latest_bar_dropped") if meta else None
    if not raw:
        return None
    d = date.fromisoformat(raw)
    if meta.last_date and d <= meta.last_date:
        return None
    return d if ((today or utcnow().date()) - d) <= timedelta(days=PENDING_MAX_AGE_DAYS) else None


def _pending_warning(d: date) -> str:
    return (f"The newest bar ({d}) has no final price yet or its session is not complete, so it is "
            "ignored for now; the data will refresh automatically within minutes of it being published.")


@dataclass
class DataStatus:
    symbol: str
    provider: str
    first_date: date | None
    last_date: date | None
    n_rows: int
    fetched_at: datetime | None
    from_cache: bool
    stale: bool = False
    warnings: list[str] = field(default_factory=list)
    pending_bar_date: str | None = None   # newest bar not yet final at the provider (re-checked soon)

    def to_dict(self) -> dict:
        return {
            "pending_bar_date": self.pending_bar_date,
            "symbol": self.symbol, "source": self.provider,
            "first_date": self.first_date.isoformat() if self.first_date else None,
            "last_date": self.last_date.isoformat() if self.last_date else None,
            "n_rows": self.n_rows,
            "fetched_at": self.fetched_at.isoformat() + "Z" if self.fetched_at else None,
            "from_cache": self.from_cache, "stale": self.stale, "warnings": self.warnings,
            "data_note": "End-of-day (delayed) prices; not real-time.",
        }


def default_start(settings: Settings, today: date | None = None) -> date:
    return (today or date.today()) - timedelta(days=int(365.25 * settings.default_history_years))


def _status(repo: Repository, symbol: str, provider: DataProvider, from_cache: bool,
            stale: bool, warnings: list[str]) -> DataStatus:
    meta = repo.get_meta(symbol)
    pend = pending_bar(meta)
    if pend and from_cache and not stale and not any("newest bar" in w for w in warnings):
        warnings = [*warnings, _pending_warning(pend)]   # keep telling users on cached views too
    return DataStatus(symbol, meta.provider if meta else provider.name,
                      meta.first_date if meta else None, meta.last_date if meta else None,
                      meta.n_rows if meta else 0, meta.last_fetch_at if meta else None,
                      from_cache, stale, warnings, pend.isoformat() if pend else None)


def _fetch_and_store(repo: Repository, provider: DataProvider, settings: Settings, symbol: str,
                     start: date, end: date | None, now: datetime | None,
                     record_start: date, result=None) -> list[str]:
    """Fetch [start, end] (unless `result` is supplied), validate, upsert. `record_start` is the
    earliest date ever requested (stored so a stock listed after it isn't refetched each call)."""
    if result is None:
        result = provider.fetch_daily(symbol, start, end)
    verify_symbol(result, symbol)
    clean, report = clean_ohlcv(result.frame, now=now,
                                complete_hour_ist=settings.session_complete_hour_ist)
    if clean.empty:
        raise DataUnavailableError(f"No valid price rows remained for {symbol} after validation.",
                                   {"quality": report.to_dict()})
    repo.ensure_stock(symbol, source="manual")
    repo.upsert_bars(symbol, clean, source=result.source)
    qr = dict(report.to_dict(), notes=result.notes, exchange=result.exchange,
              requested_start=record_start.isoformat())
    repo.save_meta(symbol, result.source, last_fetch_at=utcnow(), last_attempt_at=utcnow(),
                   last_status="ok", last_error=None, quality_report=qr)
    repo.refresh_meta_range(symbol)
    return report.warnings


def ensure_data(repo: Repository, provider: DataProvider, settings: Settings, symbol: str,
                start: date | None = None, end: date | None = None, force_refresh: bool = False,
                full: bool = False, now: datetime | None = None) -> DataStatus:
    """Make sure the local cache holds daily bars for `symbol` covering [start, end]."""
    symbol = normalize_symbol(symbol)
    start = start or default_start(settings)
    meta = repo.get_meta(symbol)
    requested_start = None
    if meta and meta.quality_report and meta.quality_report.get("requested_start"):
        requested_start = date.fromisoformat(meta.quality_report["requested_start"])
    has_cache = bool(meta and meta.n_rows and meta.last_date)
    covers_start = has_cache and requested_start is not None and requested_start <= start
    # Normally trust the cache for CACHE_TTL_HOURS; while the newest bar is still pending at the
    # provider (no final price yet), re-check every PENDING_RETRY_MINUTES instead.
    ttl = (timedelta(minutes=settings.pending_retry_minutes) if pending_bar(meta)
           else timedelta(hours=settings.cache_ttl_hours))
    fresh = bool(meta and meta.last_fetch_at and (utcnow() - meta.last_fetch_at) < ttl)

    need_full = full or not covers_start
    need_incremental = (not need_full) and (force_refresh or not fresh)
    if not need_full and not need_incremental:
        return _status(repo, symbol, provider, True, False, [])

    try:
        result = None
        if need_incremental:
            overlap_from = meta.last_date - timedelta(days=OVERLAP_DAYS)
            before = repo.load_bars(symbol, start=overlap_from)
            result = provider.fetch_daily(symbol, overlap_from, end)
            verify_symbol(result, symbol)
            new, _ = clean_ohlcv(result.frame, now=now,
                                 complete_hour_ist=settings.session_complete_hour_ist)
            common = before.index.intersection(new.index)
            if len(common) and (
                (new.loc[common, "adj_close"] / before.loc[common, "adj_close"] - 1).abs().max()
                > ADJ_MISMATCH_TOL
            ):
                log.info("%s: adjusted prices changed (corporate action) -> full refetch", symbol)
                need_full, need_incremental = True, False
        if need_full:
            fetch_start = min(start, requested_start) if requested_start else start
            warnings = _fetch_and_store(repo, provider, settings, symbol, fetch_start, end, now,
                                        record_start=fetch_start)
        else:
            warnings = _fetch_and_store(repo, provider, settings, symbol, overlap_from, end, now,
                                        record_start=requested_start, result=result)
        return _status(repo, symbol, provider, False, False, warnings)
    except AppError as exc:
        if has_cache:
            log.warning("Refresh of %s failed (%s); serving cached data.", symbol, exc.message)
            repo.save_meta(symbol, meta.provider, last_attempt_at=utcnow(), last_status="error",
                           last_error=exc.message[:1000])
            return _status(repo, symbol, provider, True, True,
                           [f"Refresh failed ({exc.message}); showing cached data that may be stale."])
        raise


def load_prices(repo: Repository, symbol: str, start: date | None = None,
                end: date | None = None) -> pd.DataFrame:
    return repo.load_bars(normalize_symbol(symbol), start, end)
