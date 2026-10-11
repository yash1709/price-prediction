"""Minimal in-process background task runner (swap for Celery/RQ if you outgrow it).

Task state lives in memory: it is lost on API restart. Durable results (training runs, model
versions) are written to the database by the work itself.
"""
from __future__ import annotations

import logging
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable

from app.core.errors import AppError, ConflictError

log = logging.getLogger(__name__)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class TaskInfo:
    id: str
    kind: str
    key: str
    status: str = "queued"  # queued | running | succeeded | failed
    progress: float = 0.0
    message: str = ""
    result: Any = None
    error: dict | None = None
    created_at: str = field(default_factory=_now)
    finished_at: str | None = None

    def to_dict(self) -> dict:
        return dict(self.__dict__)


class TaskManager:
    def __init__(self, max_workers: int = 2):
        self._pool = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="task")
        self._tasks: dict[str, TaskInfo] = {}
        self._lock = threading.Lock()

    def submit(self, kind: str, key: str, fn: Callable[[Callable[[float, str], None]], Any]
               ) -> TaskInfo:
        """Run fn(progress_cb) in the background. One active task per (kind, key)."""
        with self._lock:
            for t in self._tasks.values():
                if t.kind == kind and t.key == key and t.status in ("queued", "running"):
                    raise ConflictError(f"A '{kind}' task for '{key}' is already {t.status}.",
                                        {"task_id": t.id})
            info = TaskInfo(id=str(uuid.uuid4()), kind=kind, key=key)
            self._tasks[info.id] = info

        def progress(frac: float, msg: str) -> None:
            info.progress, info.message = round(min(max(frac, 0.0), 1.0), 3), msg

        def runner() -> None:
            info.status = "running"
            try:
                info.result = fn(progress)
                info.progress, info.status = 1.0, "succeeded"
            except AppError as exc:
                info.status, info.error = "failed", {"code": exc.code, "message": exc.message,
                                                     "details": exc.details}
            except Exception as exc:  # noqa: BLE001 - recorded, never swallowed silently
                log.exception("Task %s (%s) crashed", info.id, kind)
                info.status, info.error = "failed", {"code": "internal_error", "message": str(exc),
                                                     "details": {}}
            finally:
                info.finished_at = _now()

        self._pool.submit(runner)
        return info

    def get(self, task_id: str) -> TaskInfo | None:
        return self._tasks.get(task_id)

    def shutdown(self) -> None:
        self._pool.shutdown(wait=False, cancel_futures=True)
