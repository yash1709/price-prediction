"""FastAPI application factory."""
from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from app import __version__
from app.api.routes import data, health, risk, stocks, tasks
from app.core.config import Settings, get_settings
from app.core.errors import AppError
from app.core.logging import setup_logging
from app.data.providers import DataProvider
from app.services.context import AppContext, build_context
from app.services.workflow import seed_universe_if_empty

log = logging.getLogger("app.api")

DESCRIPTION = (
    "NSE stock price forecasting research API. Prices are end-of-day, delayed data in INR. "
    "Forecasts are research output, not investment advice."
)


def _error(status: int, code: str, message: str, details: dict | None = None) -> JSONResponse:
    return JSONResponse(status_code=status,
                        content={"error": {"code": code, "message": message,
                                           "details": details or {}}})


def create_app(settings: Settings | None = None, provider: DataProvider | None = None,
               ctx: AppContext | None = None) -> FastAPI:
    settings = settings or get_settings()
    setup_logging(settings.log_level)
    ctx = ctx or build_context(settings, provider)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        seed_universe_if_empty(ctx)
        yield
        ctx.tasks.shutdown()

    app = FastAPI(title="NSE Stock Prediction API", version=__version__, description=DESCRIPTION,
                  lifespan=lifespan)
    app.state.ctx = ctx

    @app.exception_handler(AppError)
    async def _app_error(_: Request, exc: AppError):
        level = logging.WARNING if exc.status_code < 500 else logging.ERROR
        log.log(level, "%s: %s", exc.code, exc.message)
        return _error(exc.status_code, exc.code, exc.message, exc.details)

    @app.exception_handler(RequestValidationError)
    async def _validation(_: Request, exc: RequestValidationError):
        errs = [{"loc": [str(x) for x in e["loc"]], "msg": e["msg"]} for e in exc.errors()]
        return _error(422, "validation_error", "Request validation failed.", {"errors": errs})

    @app.exception_handler(Exception)
    async def _unhandled(_: Request, exc: Exception):
        log.exception("Unhandled error")
        return _error(500, "internal_error", "Unexpected server error; see server logs.")

    for module in (health, stocks, data, tasks, risk):
        app.include_router(module.router)
    return app


def app_factory() -> FastAPI:  # for `uvicorn app.api.main:app_factory --factory`
    return create_app()
