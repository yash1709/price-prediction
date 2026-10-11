"""Pydantic request/response schemas."""
from __future__ import annotations

from datetime import date
from typing import Any

from pydantic import BaseModel, Field, field_validator, model_validator

from app.models.registry import ALGORITHMS

MAX_HORIZON = 20


def _check_horizons(v: list[int]) -> list[int]:
    if not v:
        raise ValueError("horizons must not be empty")
    if any(h < 1 or h > MAX_HORIZON for h in v):
        raise ValueError(f"each horizon must be between 1 and {MAX_HORIZON} trading days")
    return sorted(set(v))


class DateRange(BaseModel):
    start: date | None = None
    end: date | None = None

    @model_validator(mode="after")
    def _order(self):
        if self.start and self.end and self.start >= self.end:
            raise ValueError("start must be before end")
        if self.end and self.end > date.today():
            raise ValueError("end cannot be in the future")
        return self


class TrainRequest(DateRange):
    algorithms: list[str] | None = Field(None, description="Subset of algorithms; default: all")
    horizons: list[int] = Field(default_factory=lambda: [1])
    refresh_data: bool = True

    @field_validator("horizons")
    @classmethod
    def _h(cls, v):
        return _check_horizons(v)

    @field_validator("algorithms")
    @classmethod
    def _a(cls, v):
        if v is not None:
            bad = [a for a in v if a not in ALGORITHMS]
            if bad:
                raise ValueError(f"unknown algorithms {bad}; available: {sorted(ALGORITHMS)}")
        return v


class PredictRequest(BaseModel):
    horizons: list[int] = Field(default_factory=lambda: [1])
    model_version_id: int | None = None
    refresh_data: bool = True

    @field_validator("horizons")
    @classmethod
    def _h(cls, v):
        return _check_horizons(v)


class RefreshRequest(BaseModel):
    symbols: list[str] = Field(min_length=1, max_length=50)
    full: bool = False


class UniverseImportRequest(BaseModel):
    source: str | None = Field(None, description="URL or local path to NSE EQUITY_L.csv")


class ErrorBody(BaseModel):
    code: str
    message: str
    details: dict[str, Any] = {}


class ErrorResponse(BaseModel):
    error: ErrorBody


class TaskAccepted(BaseModel):
    task_id: str
    status: str
    status_url: str
