from __future__ import annotations

from fastapi import APIRouter, Depends, Query
from pydantic import BaseModel, Field

from app.api.deps import get_ctx, symbol_param
from app.services.context import AppContext
from app.services.risk import MAX_HOLDINGS, Holding, portfolio_risk, position_size, stock_risk

router = APIRouter(tags=["risk"])


class PositionSizeRequest(BaseModel):
    symbol: str
    portfolio_value: float = Field(gt=0, description="Total portfolio value in INR")
    risk_pct: float = Field(gt=0, le=100, description="Max loss on a bad move, % of the portfolio")
    horizon: int = Field(1, description="1 or 5 trading days")
    tail: str = Field("1_in_10", description="'1_in_10' (80% range bound) or '1_in_40' (95% bound)")


class HoldingIn(BaseModel):
    symbol: str
    amount: float = Field(gt=0, description="Invested amount in INR")


class PortfolioRequest(BaseModel):
    holdings: list[HoldingIn] = Field(min_length=1, max_length=MAX_HOLDINGS)


@router.get("/stocks/{symbol}/risk")
def risk_card(symbol: str = Depends(symbol_param), refresh: bool = Query(True),
              ctx: AppContext = Depends(get_ctx)) -> dict:
    """Volatility, calibrated expected-move ranges, bad-day thresholds, drawdown, beta, liquidity."""
    return stock_risk(ctx, symbol, refresh)


@router.post("/risk/position-size")
def risk_position_size(body: PositionSizeRequest, ctx: AppContext = Depends(get_ctx)) -> dict:
    """Largest position whose loss on a bad move stays within the risk budget."""
    return position_size(stock_risk(ctx, body.symbol), body.portfolio_value, body.risk_pct,
                         body.horizon, body.tail)


@router.post("/risk/portfolio")
def risk_portfolio(body: PortfolioRequest, ctx: AppContext = Depends(get_ctx)) -> dict:
    """Portfolio volatility, risk contributions, diversification and correlations (no price forecast)."""
    return portfolio_risk(ctx, [Holding(h.symbol, h.amount) for h in body.holdings])
