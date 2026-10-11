from __future__ import annotations

from fastapi import APIRouter, Depends

from app.api.deps import get_ctx
from app.core.errors import NotFoundError
from app.services.context import AppContext

router = APIRouter(tags=["tasks"])


@router.get("/tasks/{task_id}")
def task_status(task_id: str, ctx: AppContext = Depends(get_ctx)) -> dict:
    task = ctx.tasks.get(task_id)
    if task is None:
        raise NotFoundError(f"Task {task_id} not found (task state is lost on API restart).")
    return task.to_dict()
