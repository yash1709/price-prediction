from __future__ import annotations

from typing import Iterator

from fastapi import Depends, Path, Request

from app.core.symbols import normalize_symbol
from app.database.repository import Repository
from app.services.context import AppContext


def get_ctx(request: Request) -> AppContext:
    return request.app.state.ctx


def get_repo(ctx: AppContext = Depends(get_ctx)) -> Iterator[Repository]:
    with ctx.repo() as repo:
        yield repo


def symbol_param(symbol: str = Path(..., description="NSE symbol, e.g. RELIANCE or M&M")) -> str:
    return normalize_symbol(symbol)
