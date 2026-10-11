"""Shared fixtures. Tests never touch the network: they use a fake in-memory provider and
synthetic random-walk prices (test data only; the application never fabricates prices)."""
from __future__ import annotations

import sys
from datetime import date, datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.api.main import create_app  # noqa: E402
from app.core.config import Settings  # noqa: E402
from app.core.errors import ProviderError, SymbolNotFoundError  # noqa: E402
from app.data.providers.base import DataProvider, ProviderResult  # noqa: E402
from app.services.context import build_context  # noqa: E402


def make_prices(n: int = 700, end: str = "2024-06-28", seed: int = 0, start_price: float = 1000.0
                ) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range(end=end, periods=n, name="date")
    ret = rng.normal(0.0004, 0.012, n)
    close = start_price * np.exp(np.cumsum(ret))
    open_ = close * (1 + rng.normal(0, 0.003, n))
    high = np.maximum(open_, close) * (1 + np.abs(rng.normal(0, 0.004, n)))
    low = np.minimum(open_, close) * (1 - np.abs(rng.normal(0, 0.004, n)))
    vol = rng.integers(1_000_000, 5_000_000, n).astype(float)
    return pd.DataFrame({"open": open_, "high": high, "low": low, "close": close,
                         "adj_close": close, "volume": vol}, index=idx)


class FakeProvider(DataProvider):
    name = "fake"

    def __init__(self, frames: dict[str, pd.DataFrame] | None = None):
        self.frames = frames or {}
        self.calls: list[tuple[str, date, date | None]] = []
        self.fail_with: Exception | None = None
        self.reported_symbol_override: str | None = None

    def fetch_daily(self, symbol, start, end=None) -> ProviderResult:
        self.calls.append((symbol, start, end))
        if self.fail_with:
            raise self.fail_with
        if symbol not in self.frames:
            raise SymbolNotFoundError(f"No data for {symbol}")
        df = self.frames[symbol]
        df = df[df.index >= pd.Timestamp(start)]
        if end is not None:
            df = df[df.index <= pd.Timestamp(end)]
        if df.empty:
            raise SymbolNotFoundError(f"No rows for {symbol}")
        return ProviderResult(symbol=symbol, frame=df.copy(), source="fake",
                              fetched_at=datetime(2024, 7, 1), currency="INR", exchange="NSI",
                              reported_symbol=self.reported_symbol_override or f"{symbol}.NS")


@pytest.fixture
def prices() -> pd.DataFrame:
    return make_prices()


@pytest.fixture
def settings(tmp_path) -> Settings:
    return Settings(
        database_url=f"sqlite:///{(tmp_path / 'test.db').as_posix()}",
        artifacts_dir=tmp_path / "artifacts", data_dir=tmp_path / "data",
        min_train_rows=150, walk_forward_folds=3, walk_forward_test_size=21,
        cache_ttl_hours=6, _env_file=None)


@pytest.fixture
def provider(prices) -> FakeProvider:
    return FakeProvider({"TESTCO": prices})


def make_market(index: pd.DatetimeIndex, seed: int = 1) -> pd.DataFrame:
    """Synthetic Nifty / India VIX closes on the given dates (test data only)."""
    rng = np.random.default_rng(seed)
    nifty = 18000 * np.exp(np.cumsum(rng.normal(0.0004, 0.009, len(index))))
    vix = np.clip(15 + np.cumsum(rng.normal(0, 0.4, len(index))), 9, 40)
    return pd.DataFrame({"nifty": nifty, "vix": vix}, index=index)


@pytest.fixture
def market(prices) -> pd.DataFrame:
    return make_market(prices.index)


@pytest.fixture
def ctx(settings, provider, market):
    from app.models.intervals import IntervalService
    c = build_context(settings, provider, IntervalService(settings, lambda: market))
    yield c
    c.tasks.shutdown()


@pytest.fixture
def client(ctx, settings):
    from fastapi.testclient import TestClient
    app = create_app(settings, ctx=ctx)
    with TestClient(app) as c:
        yield c
