"""High-level use-cases shared by the API and the CLI."""
from __future__ import annotations

import logging
from datetime import date

from app.core.errors import NotFoundError
from app.core.symbols import normalize_symbol
from app.data.ingestion import DataStatus, ensure_data
from app.data.universe import import_universe, load_seed
from app.models.prediction import DISCLAIMER, predict_for_symbol
from app.models.training import train_symbol
from app.services.context import AppContext
from app.services.outcomes import resolve_outcomes

log = logging.getLogger(__name__)


def seed_universe_if_empty(ctx: AppContext) -> None:
    with ctx.repo() as repo:
        total, _ = repo.search_stocks(None, 1, 0)
        if total == 0:
            load_seed(repo)


def import_universe_source(ctx: AppContext, source: str | None = None) -> dict:
    with ctx.repo() as repo:
        n_file, n_new = import_universe(repo, source or ctx.settings.universe_url)
        total, _ = repo.search_stocks(None, 1, 0)
    return {"rows_in_file": n_file, "newly_inserted": n_new, "universe_size": total}


def fetch_data(ctx: AppContext, symbol: str, start: date | None = None, end: date | None = None,
               force_refresh: bool = False, full: bool = False) -> DataStatus:
    symbol = normalize_symbol(symbol)
    with ctx.repo() as repo:
        return ensure_data(repo, ctx.provider, ctx.settings, symbol, start, end,
                           force_refresh=force_refresh, full=full)


def run_training(ctx: AppContext, symbol: str, algorithms: list[str] | None,
                 horizons: list[int] | None, start: date | None, end: date | None,
                 progress=None, refresh_data: bool = True) -> dict:
    symbol = normalize_symbol(symbol)
    if progress:
        progress(0.0, "checking data")
    if refresh_data:
        fetch_data(ctx, symbol, start=start, end=end)
    with ctx.repo() as repo:
        if progress:
            progress(0.02, "training")
        run = train_symbol(repo, ctx.settings, symbol, algorithms, horizons, start, end,
                           progress=progress)
        return {"run_id": run.id, "status": run.status, "summary": run.summary}


def run_prediction(ctx: AppContext, symbol: str, horizons: list[int] | None,
                   model_version_id: int | None = None, refresh_data: bool = True) -> dict:
    symbol = normalize_symbol(symbol)
    warnings: list[str] = []
    if refresh_data:
        status = fetch_data(ctx, symbol)
        warnings += status.warnings
    with ctx.repo() as repo:
        if repo.get_stock(symbol) is None:
            raise NotFoundError(f"Unknown symbol {symbol}; fetch its data first.")
        resolve_outcomes(repo, symbol)  # settle earlier forecasts whose target day has now passed
        preds, w = predict_for_symbol(repo, symbol, horizons, model_version_id)
        warnings += w
        meta = repo.get_meta(symbol)
        data_info = {"last_bar_date": meta.last_date.isoformat() if meta and meta.last_date else None,
                     "source": meta.provider if meta else None}
    return {"symbol": symbol, "kind": "forecast", "predictions": preds, "warnings": warnings,
            "data": data_info, "disclaimer": DISCLAIMER}
