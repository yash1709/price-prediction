from __future__ import annotations

from fastapi import APIRouter, Depends

from app.api.deps import get_ctx
from app.api.schemas import RefreshRequest, TaskAccepted, UniverseImportRequest
from app.core.symbols import normalize_symbol
from app.services.context import AppContext
from app.services.workflow import fetch_data, import_universe_source

router = APIRouter(prefix="/data", tags=["data"])


@router.post("/refresh", status_code=202, response_model=TaskAccepted)
def refresh(body: RefreshRequest, ctx: AppContext = Depends(get_ctx)) -> TaskAccepted:
    """Refresh cached market data for the given symbols (background task)."""
    symbols = [normalize_symbol(s) for s in body.symbols]

    def work(progress):
        results = {}
        for i, sym in enumerate(symbols):
            progress(i / len(symbols), f"refreshing {sym}")
            try:
                results[sym] = fetch_data(ctx, sym, force_refresh=True, full=body.full).to_dict()
            except Exception as exc:  # noqa: BLE001 - per-symbol failure is reported, not hidden
                results[sym] = {"error": str(exc)}
        return results

    task = ctx.tasks.submit("refresh", ",".join(sorted(symbols)), work)
    return TaskAccepted(task_id=task.id, status=task.status, status_url=f"/tasks/{task.id}")


@router.post("/universe/import")
def universe_import(body: UniverseImportRequest, ctx: AppContext = Depends(get_ctx)) -> dict:
    """Import the NSE equity list (URL or local file) into the stock universe."""
    return import_universe_source(ctx, body.source)
