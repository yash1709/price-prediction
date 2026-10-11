from datetime import date, datetime, timedelta, timezone

import numpy as np
import pandas as pd
import pytest

from app.core.errors import (
    InvalidSymbolError, ProviderError, SymbolMismatchError, SymbolNotFoundError,
)
from app.core.symbols import normalize_symbol, to_yahoo_ticker
from app.data.ingestion import ensure_data
from app.data.providers.base import ProviderResult
from app.data.providers.csv_provider import CSVProvider
from app.data.providers.yfinance_provider import YFinanceProvider
from app.data.universe import SEED_FILE, parse_equity_list
from app.data.validation import IST, clean_ohlcv, drop_incomplete_session, verify_symbol
from tests.conftest import FakeProvider, make_prices


def test_symbol_normalisation():
    assert normalize_symbol(" reliance.ns ") == "RELIANCE"
    assert normalize_symbol("m&m") == "M&M"
    assert to_yahoo_ticker("bajaj-auto") == "BAJAJ-AUTO.NS"
    for bad in ("", "AB CD", "../etc", "X" * 30, "A;B"):
        with pytest.raises(InvalidSymbolError):
            normalize_symbol(bad)


def test_clean_removes_duplicates_invalid_and_never_fills():
    df = make_prices(30)
    dup = pd.concat([df, df.iloc[[5]]]).sort_index()
    dup.iloc[10, dup.columns.get_loc("close")] = np.nan          # missing price
    dup.iloc[12, dup.columns.get_loc("high")] = dup.iloc[12]["low"] - 1   # high < low
    dup.iloc[14, dup.columns.get_loc("volume")] = -5             # negative volume
    out, rep = clean_ohlcv(dup, now=datetime(2030, 1, 1, tzinfo=timezone.utc))
    assert rep.duplicates_removed == 1
    assert rep.invalid_rows_removed == 3
    assert out.index.is_unique and out.index.is_monotonic_increasing
    assert len(out) == 30 - 3                       # rows dropped, NOT interpolated
    assert not out.isna().any().any()


def test_gap_and_extreme_move_are_flagged_not_fixed():
    df = make_prices(60)
    df = df.drop(df.index[20:35])                   # ~3 week gap
    df.iloc[40, df.columns.get_loc("adj_close")] *= 1.6
    out, rep = clean_ohlcv(df, now=datetime(2030, 1, 1, tzinfo=timezone.utc))
    assert any("gap" in w for w in rep.warnings)
    assert any("daily move" in w for w in rep.warnings)
    assert len(out) == len(df)


def test_incomplete_session_dropped_before_close_only():
    df = make_prices(10, end="2024-06-28")
    before_close = datetime(2024, 6, 28, 12, 0, tzinfo=IST)
    after_close = datetime(2024, 6, 28, 17, 0, tzinfo=IST)
    out, dropped = drop_incomplete_session(df, before_close, 16.0)
    assert dropped and len(out) == 9
    out, dropped = drop_incomplete_session(df, after_close, 16.0)
    assert not dropped and len(out) == 10
    out, dropped = drop_incomplete_session(df, datetime(2024, 6, 29, 10, 0, tzinfo=IST), 16.0)
    assert not dropped


def _result(**kw):
    base = dict(symbol="TCS", frame=make_prices(5), source="x", fetched_at=datetime(2024, 1, 1),
                currency="INR", reported_symbol="TCS.NS")
    base.update(kw)
    return ProviderResult(**base)


def test_symbol_validation():
    verify_symbol(_result(), "TCS")
    with pytest.raises(SymbolMismatchError):
        verify_symbol(_result(reported_symbol="INFY.NS"), "TCS")
    with pytest.raises(SymbolMismatchError):
        verify_symbol(_result(currency="USD"), "TCS")


def test_yfinance_empty_response_is_symbol_not_found():
    with pytest.raises(SymbolNotFoundError):
        YFinanceProvider._to_result("XYZ", "XYZ.NS", pd.DataFrame(), {})


def test_yfinance_response_is_normalised():
    raw = make_prices(5).rename(columns={"open": "Open", "high": "High", "low": "Low",
                                         "close": "Close", "adj_close": "Adj Close",
                                         "volume": "Volume"})
    raw.index = raw.index.tz_localize("Asia/Kolkata")
    res = YFinanceProvider._to_result("TCS", "TCS.NS", raw, {"symbol": "TCS.NS", "currency": "INR"})
    assert list(res.frame.columns) == ["open", "high", "low", "close", "adj_close", "volume"]
    assert res.frame.index.tz is None and res.reported_symbol == "TCS.NS"


def test_csv_provider(tmp_path):
    p = make_prices(20)
    p.rename(columns=str.title).rename(columns={"Adj_Close": "Adj Close"}).to_csv(tmp_path / "ABC.csv")
    res = CSVProvider(tmp_path).fetch_daily("ABC", date(2000, 1, 1))
    assert len(res.frame) == 20
    with pytest.raises(SymbolNotFoundError):
        CSVProvider(tmp_path).fetch_daily("NOPE", date(2000, 1, 1))


def test_universe_parsing_and_seed_is_flagged_small():
    rows = parse_equity_list(
        "SYMBOL, NAME OF COMPANY, SERIES, ISIN NUMBER\nABC,Abc Ltd,EQ,INE1\nbad sym,x,EQ,y\n", "nse_import")
    assert [r["symbol"] for r in rows] == ["ABC"] and rows[0]["isin"] == "INE1"
    seed = parse_equity_list(SEED_FILE.read_text(encoding="utf-8"), "seed")
    assert 10 < len(seed) < 100  # a starter list, documented as not-the-full-NSE


def test_ensure_data_uses_cache_then_increments(ctx, provider, prices):
    with ctx.repo() as repo:
        st = ensure_data(repo, provider, ctx.settings, "TESTCO", start=date(2022, 1, 1))
        assert not st.from_cache and st.n_rows > 0 and st.last_date == prices.index[-1].date()
        n_calls = len(provider.calls)
        st2 = ensure_data(repo, provider, ctx.settings, "TESTCO", start=date(2022, 1, 1))
        assert st2.from_cache and len(provider.calls) == n_calls       # no new request
        # a later request with a longer history triggers a refetch of the earlier range
        ensure_data(repo, provider, ctx.settings, "TESTCO", start=date(2021, 1, 1))
        assert len(provider.calls) == n_calls + 1


def test_ensure_data_incremental_refresh_adds_new_bars(ctx, provider, prices):
    longer = make_prices(705)
    provider.frames["TESTCO"] = longer.iloc[:-5]
    with ctx.repo() as repo:
        ensure_data(repo, provider, ctx.settings, "TESTCO", start=date(2022, 1, 1))
        before = repo.get_meta("TESTCO").n_rows
        provider.frames["TESTCO"] = longer
        st = ensure_data(repo, provider, ctx.settings, "TESTCO", start=date(2022, 1, 1),
                         force_refresh=True)
        assert st.n_rows == before + 5
        assert provider.calls[-1][1] > date(2022, 1, 1)                # incremental window


def test_ensure_data_serves_stale_cache_on_failure(ctx, provider):
    with ctx.repo() as repo:
        ensure_data(repo, provider, ctx.settings, "TESTCO", start=date(2022, 1, 1))
        provider.fail_with = ProviderError("network down")
        st = ensure_data(repo, provider, ctx.settings, "TESTCO", start=date(2022, 1, 1),
                         force_refresh=True)
        assert st.stale and st.from_cache and any("network down" in w for w in st.warnings)
        assert repo.get_meta("TESTCO").last_status == "error"


def test_unknown_symbol_without_cache_raises_and_is_not_registered(ctx, provider):
    with ctx.repo() as repo:
        with pytest.raises(SymbolNotFoundError):
            ensure_data(repo, provider, ctx.settings, "NOSUCH", start=date(2022, 1, 1))
        assert repo.get_stock("NOSUCH") is None


def test_mismatched_symbol_is_rejected(ctx, provider):
    provider.reported_symbol_override = "OTHER.NS"
    with ctx.repo() as repo:
        with pytest.raises(SymbolMismatchError):
            ensure_data(repo, provider, ctx.settings, "TESTCO", start=date(2022, 1, 1))
        assert repo.load_bars("TESTCO").empty if repo.get_stock("TESTCO") else True
