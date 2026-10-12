"""Data-quality checks and cleaning for daily OHLCV frames. Never fills or invents prices."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta, timezone

import pandas as pd

from app.core.errors import SymbolMismatchError
from app.data.providers.base import CANONICAL_COLUMNS, ProviderResult

IST = timezone(timedelta(hours=5, minutes=30))
MAX_GAP_DAYS = 10          # longer calendar gaps are flagged (holiday clusters are normally < 6)
EXTREME_RETURN = 0.30      # |daily return| above this is flagged (possible unadjusted split)


@dataclass
class QualityReport:
    rows_in: int = 0
    rows_out: int = 0
    duplicates_removed: int = 0
    invalid_rows_removed: int = 0
    incomplete_session_dropped: bool = False
    # ISO date of the newest raw bar that was discarded (no price yet / session not complete);
    # while recent, ingestion re-checks the provider soon instead of trusting the long cache.
    latest_bar_dropped: str | None = None
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


def verify_symbol(result: ProviderResult, requested: str) -> None:
    """Reject responses that are demonstrably for another instrument."""
    rep = (result.reported_symbol or "").upper()
    if rep and rep.removesuffix(".NS").removesuffix(".BO") != requested.upper():
        raise SymbolMismatchError(
            f"Provider returned data for '{result.reported_symbol}' but '{requested}' was requested.")
    if result.currency and result.currency.upper() != "INR":
        raise SymbolMismatchError(
            f"Data for {requested} is quoted in {result.currency}, not INR; refusing to use it.")


def drop_incomplete_session(df: pd.DataFrame, now: datetime | None, complete_hour_ist: float
                            ) -> tuple[pd.DataFrame, bool]:
    """Drop a bar dated today (IST) if the NSE session may still be open / not yet final."""
    if df.empty:
        return df, False
    if now is None:
        now = datetime.now(timezone.utc)
    elif now.tzinfo is None:  # naive datetimes are interpreted as IST wall-clock time
        now = now.replace(tzinfo=IST)
    now_ist = now.astimezone(IST)
    hour = now_ist.hour + now_ist.minute / 60
    if df.index[-1] >= pd.Timestamp(now_ist.date()) and hour < complete_hour_ist:
        return df.iloc[:-1], True
    return df, False


def clean_ohlcv(df: pd.DataFrame, now: datetime | None = None,
                complete_hour_ist: float = 16.0) -> tuple[pd.DataFrame, QualityReport]:
    rep = QualityReport(rows_in=len(df))
    missing = [c for c in CANONICAL_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")
    out = df[CANONICAL_COLUMNS].copy()
    out.index = pd.DatetimeIndex(pd.to_datetime(out.index)).normalize()
    out.index.name = "date"
    out = out.apply(pd.to_numeric, errors="coerce")

    dup = out.index.duplicated(keep="last")
    rep.duplicates_removed = int(dup.sum())
    out = out[~dup].sort_index()
    raw_last = out.index[-1] if len(out) else None

    # adj_close may legitimately be absent; fall back to close ONLY for that column (documented).
    if out["adj_close"].isna().all():
        out["adj_close"] = out["close"]
        rep.warnings.append("adj_close unavailable from provider; close used as adj_close.")

    ohlc = ["open", "high", "low", "close"]
    bad = out[ohlc].isna().any(axis=1) | (out[ohlc] <= 0).any(axis=1)
    bad |= out["adj_close"].isna() | (out["adj_close"] <= 0)
    bad |= out["high"] < out["low"]
    bad |= out["high"] < out[["open", "close"]].max(axis=1) - 1e-9 * out["close"].abs()
    bad |= out["low"] > out[["open", "close"]].min(axis=1) + 1e-9 * out["close"].abs()
    bad |= out["volume"].lt(0)
    rep.invalid_rows_removed = int(bad.sum())
    if rep.invalid_rows_removed:
        rep.warnings.append(f"Removed {rep.invalid_rows_removed} row(s) with invalid OHLC/volume values.")
    out = out[~bad]

    out, dropped = drop_incomplete_session(out, now, complete_hour_ist)
    rep.incomplete_session_dropped = dropped
    if dropped:
        rep.warnings.append("Dropped today's bar: the NSE session may not be complete yet.")

    if len(out) > 1:
        gaps = out.index.to_series().diff().dt.days.dropna()
        n_gap = int((gaps > MAX_GAP_DAYS).sum())
        if n_gap:
            rep.warnings.append(f"{n_gap} gap(s) longer than {MAX_GAP_DAYS} calendar days "
                                f"(suspension or missing data); gaps are NOT filled.")
        ext = int((out["adj_close"].pct_change().abs() > EXTREME_RETURN).sum())
        if ext:
            rep.warnings.append(f"{ext} daily move(s) above {EXTREME_RETURN:.0%} "
                                f"(possible corporate action or bad tick); inspect before trusting.")
    if len(out) and (out["volume"].fillna(0) == 0).mean() > 0.2:
        rep.warnings.append("More than 20% of sessions report zero/missing volume.")
    if raw_last is not None and (len(out) == 0 or out.index[-1] < raw_last):
        rep.latest_bar_dropped = str(raw_last.date())
        rep.warnings.append(
            f"The newest bar ({raw_last.date()}) has no final price yet or its session is not complete, "
            "so it was ignored. It is picked up automatically on a later refresh.")
    rep.rows_out = len(out)
    return out, rep
