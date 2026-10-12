"""A pending newest bar (no final price yet) is ignored, flagged, and re-checked quickly."""
from datetime import date, datetime, timedelta, timezone

import numpy as np
import pandas as pd
import pytest

from app.data.ingestion import ensure_data, pending_bar
from app.data.validation import IST, clean_ohlcv
from app.database.models import utcnow
from tests.conftest import make_prices

PRICE_COLS = ["open", "high", "low", "close", "adj_close"]


def recent_frame(days_ago: int = 2, pending: bool = False) -> pd.DataFrame:
    df = make_prices(300, end=str(date.today() - timedelta(days=days_ago)))
    if pending:                                   # Yahoo-style: row exists, volume present, price missing
        df.iloc[-1, [df.columns.get_loc(c) for c in PRICE_COLS]] = np.nan
    return df


def test_cleaner_flags_a_missing_price_on_the_newest_bar():
    df = recent_frame(pending=True)
    out, rep = clean_ohlcv(df, now=datetime(2030, 1, 1, tzinfo=timezone.utc))
    assert rep.latest_bar_dropped == str(df.index[-1].date()) and out.index[-1] == df.index[-2]
    assert any("newest bar" in w for w in rep.warnings)
    ok, rep2 = clean_ohlcv(recent_frame(), now=datetime(2030, 1, 1, tzinfo=timezone.utc))
    assert rep2.latest_bar_dropped is None and not any("newest bar" in w for w in rep2.warnings)


def test_cleaner_flags_an_incomplete_session_too():
    df = make_prices(50, end="2024-06-28")
    _, rep = clean_ohlcv(df, now=datetime(2024, 6, 28, 12, 0, tzinfo=IST))      # market still open
    assert rep.incomplete_session_dropped and rep.latest_bar_dropped == "2024-06-28"


def test_a_pending_bar_is_ignored_then_picked_up_within_the_retry_window(ctx, provider):
    pending, complete = recent_frame(pending=True), recent_frame()
    provider.frames["PEND"] = pending
    start = date.today() - timedelta(days=800)
    with ctx.repo() as repo:
        st = ensure_data(repo, provider, ctx.settings, "PEND", start=start)
        assert st.last_date == pending.index[-2].date()                           # bar with no price ignored
        assert st.pending_bar_date == str(pending.index[-1].date()) and any("newest bar" in w for w in st.warnings)
        calls = len(provider.calls)

        st = ensure_data(repo, provider, ctx.settings, "PEND", start=start)       # right away: cached...
        assert st.from_cache and len(provider.calls) == calls
        assert any("newest bar" in w for w in st.warnings)                        # ...but still flagged

        provider.frames["PEND"] = complete                                        # provider finalises the bar
        repo.save_meta("PEND", "fake", last_fetch_at=utcnow() - timedelta(minutes=5))
        ensure_data(repo, provider, ctx.settings, "PEND", start=start)
        assert len(provider.calls) == calls                                       # < 15 min: still cached

        repo.save_meta("PEND", "fake", last_fetch_at=utcnow() - timedelta(minutes=20))
        st = ensure_data(repo, provider, ctx.settings, "PEND", start=start)
        assert len(provider.calls) == calls + 1 and not st.from_cache             # > 15 min: re-checked
        assert st.last_date == complete.index[-1].date()                          # newest bar now present
        assert st.pending_bar_date is None and not any("newest bar" in w for w in st.warnings)


def test_normal_cache_keeps_its_long_ttl(ctx, provider):
    provider.frames["OKAY"] = recent_frame()
    start = date.today() - timedelta(days=800)
    with ctx.repo() as repo:
        ensure_data(repo, provider, ctx.settings, "OKAY", start=start)
        calls = len(provider.calls)
        repo.save_meta("OKAY", "fake", last_fetch_at=utcnow() - timedelta(minutes=20))
        st = ensure_data(repo, provider, ctx.settings, "OKAY", start=start)
        assert st.from_cache and len(provider.calls) == calls and st.pending_bar_date is None
        repo.save_meta("OKAY", "fake", last_fetch_at=utcnow() - timedelta(hours=7))   # past the 6 h TTL
        st = ensure_data(repo, provider, ctx.settings, "OKAY", start=start)
        assert not st.from_cache and len(provider.calls) == calls + 1


def test_an_old_dropped_bar_does_not_cause_endless_refetching(ctx, provider):
    provider.frames["OLD"] = recent_frame(days_ago=30, pending=True)             # dropped bar is a month old
    start = date.today() - timedelta(days=800)
    with ctx.repo() as repo:
        st = ensure_data(repo, provider, ctx.settings, "OLD", start=start)
        assert st.pending_bar_date is None and pending_bar(repo.get_meta("OLD")) is None
        calls = len(provider.calls)
        repo.save_meta("OLD", "fake", last_fetch_at=utcnow() - timedelta(minutes=20))
        assert ensure_data(repo, provider, ctx.settings, "OLD", start=start).from_cache
        assert len(provider.calls) == calls


def test_pending_bar_helper_ignores_resolved_and_missing():
    class M:  # minimal stand-in for DataCacheMeta
        def __init__(self, dropped, last):
            self.quality_report, self.last_date = ({"latest_bar_dropped": dropped} if dropped else {}), last
    today = date(2026, 10, 13)
    assert pending_bar(None) is None and pending_bar(M(None, date(2026, 10, 9)), today) is None
    assert pending_bar(M("2026-10-12", date(2026, 10, 9)), today) == date(2026, 10, 12)
    assert pending_bar(M("2026-10-09", date(2026, 10, 9)), today) is None         # already have that bar
    assert pending_bar(M("2026-09-01", date(2026, 8, 30)), today) is None          # too old to matter
