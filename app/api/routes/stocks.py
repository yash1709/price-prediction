from __future__ import annotations

from datetime import date

from fastapi import APIRouter, Depends, Query

from app.api.deps import get_ctx, get_repo, symbol_param
from app.api.schemas import PredictRequest, TaskAccepted, TrainRequest
from app.core.errors import NotFoundError
from app.data.ingestion import default_start
from app.database.repository import Repository, utc_iso
from app.features.engineering import FEATURE_NAMES, LOOKBACK
from app.models.prediction import prediction_to_dict
from app.services.context import AppContext
from app.services.outcomes import performance_summary, resolve_outcomes
from app.services.workflow import fetch_data, run_prediction, run_training

router = APIRouter(prefix="/stocks", tags=["stocks"])

PRICE_NOTE = "Prices in INR (₹), end-of-day and delayed; not real-time."


def _model_dict(mv) -> dict:
    return {
        "id": mv.id, "run_id": mv.run_id, "symbol": mv.symbol, "algorithm": mv.algorithm,
        "horizon": mv.horizon, "version": mv.version, "train_start": mv.train_start.isoformat(),
        "train_end": mv.train_end.isoformat(), "n_train": mv.n_train, "metrics": mv.metrics,
        "eligible": mv.eligible, "selection_note": mv.selection_note,
        "is_selected": mv.is_selected, "created_at": utc_iso(mv.created_at),
        "n_features": len(mv.feature_names),
    }


@router.get("")
def list_stocks(q: str | None = Query(None, description="Search symbol or company name"),
                limit: int = Query(100, ge=1, le=5000), offset: int = Query(0, ge=0),
                repo: Repository = Depends(get_repo)) -> dict:
    total, rows = repo.search_stocks(q, limit, offset)
    return {"total": total, "limit": limit, "offset": offset,
            "stocks": [{"symbol": s.symbol, "name": s.name, "series": s.series,
                        "source": s.source} for s in rows],
            "note": "The bundled universe is a small seed list; run `python run.py "
                    "import-universe` for the full NSE list. Any valid NSE symbol can also be "
                    "requested directly."}


@router.get("/{symbol}/history")
def history(symbol: str = Depends(symbol_param), start: date | None = None, end: date | None = None,
            refresh: bool = False, ctx: AppContext = Depends(get_ctx)) -> dict:
    s = start or default_start(ctx.settings)
    status = fetch_data(ctx, symbol, start=s, end=end, force_refresh=refresh)
    with ctx.repo() as repo:
        df = repo.load_bars(symbol, s, end)
        stock = repo.get_stock(symbol)
    rows = [{"date": ts.date().isoformat(), **{c: (None if v != v else float(v))
             for c, v in r.items()}} for ts, r in df.iterrows()]
    return {"symbol": symbol, "name": stock.name if stock else None, "currency": "INR",
            "note": PRICE_NOTE, "data": status.to_dict(), "count": len(rows), "rows": rows}


@router.post("/{symbol}/train", status_code=202, response_model=TaskAccepted)
def train(body: TrainRequest, symbol: str = Depends(symbol_param),
          ctx: AppContext = Depends(get_ctx)) -> TaskAccepted:
    """Start training + walk-forward evaluation in the background; poll `/tasks/{task_id}`."""
    task = ctx.tasks.submit("train", symbol, lambda progress: run_training(
        ctx, symbol, body.algorithms, body.horizons, body.start, body.end, progress,
        refresh_data=body.refresh_data))
    return TaskAccepted(task_id=task.id, status=task.status, status_url=f"/tasks/{task.id}")


@router.get("/{symbol}/models")
def models(symbol: str = Depends(symbol_param), horizon: int | None = Query(None, ge=1),
           selected_only: bool = False, repo: Repository = Depends(get_repo)) -> dict:
    mvs = repo.list_model_versions(symbol, horizon, selected_only)
    return {"symbol": symbol, "count": len(mvs), "models": [_model_dict(m) for m in mvs],
            "feature_names": FEATURE_NAMES, "lookback_sessions": LOOKBACK,
            "target": "return over `horizon` trading days: adj_close[t+h]/adj_close[t]-1",
            "evaluation": "Expanding-window walk-forward, chronological, purge of h-1 rows. "
                          "Metrics are out-of-sample backtests, not live forecasts."}


@router.get("/{symbol}/models/{model_id}/backtest")
def backtest(model_id: int, symbol: str = Depends(symbol_param),
             repo: Repository = Depends(get_repo)) -> dict:
    mv = repo.get_model_version(model_id)
    if mv is None or mv.symbol != symbol:
        raise NotFoundError(f"Model version {model_id} not found for {symbol}.")
    rows = repo.backtest_rows(model_id)
    return {"symbol": symbol, "kind": "backtest", "model": _model_dict(mv),
            "note": "Historical out-of-sample predictions; NOT future forecasts. "
                    "Prices are split/dividend adjusted.",
            "rows": [{"as_of_date": r.as_of_date.isoformat(),
                      "target_date": r.target_date.isoformat(), "fold": r.fold,
                      "reference_price": r.reference_price, "predicted_price": r.predicted_price,
                      "actual_price": r.actual_price} for r in rows]}


@router.post("/{symbol}/predict")
def predict(body: PredictRequest, symbol: str = Depends(symbol_param),
            ctx: AppContext = Depends(get_ctx)) -> dict:
    return run_prediction(ctx, symbol, body.horizons, body.model_version_id, body.refresh_data)


@router.get("/{symbol}/predictions")
def predictions(symbol: str = Depends(symbol_param), limit: int = Query(200, ge=1, le=5000),
                horizon: int | None = Query(None, ge=1),
                repo: Repository = Depends(get_repo)) -> dict:
    rows = repo.list_predictions(symbol, limit, horizon)
    return {"symbol": symbol, "count": len(rows), "kind": "forecast",
            "predictions": [prediction_to_dict(p) for p in rows]}


@router.get("/{symbol}/performance")
def performance(symbol: str = Depends(symbol_param), repo: Repository = Depends(get_repo)) -> dict:
    rows = repo.list_predictions(symbol, 10_000)
    resolved = [prediction_to_dict(p) for p in rows if p.outcome is not None]
    return {"symbol": symbol, "pending": sum(1 for p in rows if p.outcome is None),
            "resolved": len(resolved), "by_model": performance_summary(repo, symbol),
            "outcomes": resolved,
            "note": "Realised accuracy of stored forecasts versus actual closes. "
                    "Call POST /stocks/{symbol}/outcomes/update to record newly available closes."}


@router.post("/{symbol}/outcomes/update")
def update_outcomes(symbol: str = Depends(symbol_param), refresh: bool = True,
                    ctx: AppContext = Depends(get_ctx)) -> dict:
    """Fetch the latest data and record actual closes / errors for matured forecasts."""
    warnings: list[str] = []
    if refresh:
        warnings = fetch_data(ctx, symbol, force_refresh=True).warnings
    with ctx.repo() as repo:
        n = resolve_outcomes(repo, symbol)
        pending = len(repo.unresolved_predictions(symbol))
    return {"symbol": symbol, "newly_resolved": n, "still_pending": pending, "warnings": warnings}
