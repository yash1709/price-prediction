from __future__ import annotations

from datetime import datetime, timezone

from fastapi import APIRouter, Depends
from sqlalchemy import text

from app import __version__
from app.api.deps import get_ctx
from app.models.registry import ALGORITHMS
from app.services.context import AppContext

router = APIRouter(tags=["health"])


@router.get("/health")
def health(ctx: AppContext = Depends(get_ctx)) -> dict:
    db_ok = True
    try:
        with ctx.repo() as repo:
            repo.s.execute(text("SELECT 1"))
    except Exception:  # noqa: BLE001 - reported in the payload, not swallowed
        db_ok = False
    return {"status": "ok" if db_ok else "degraded", "version": __version__, "database": db_ok,
            "data_provider": ctx.provider.name,
            "algorithms": {k: v.description for k, v in ALGORITHMS.items()},
            "time_utc": datetime.now(timezone.utc).isoformat()}
