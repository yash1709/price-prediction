"""SQLAlchemy ORM models (SQLite now; the URL can point at PostgreSQL later)."""
from __future__ import annotations

from datetime import date, datetime, timezone

from sqlalchemy import (
    JSON, Boolean, Date, DateTime, Float, ForeignKey, Integer, String, Text, UniqueConstraint,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


def utcnow() -> datetime:
    return datetime.now(timezone.utc).replace(tzinfo=None)  # stored as naive UTC


class Base(DeclarativeBase):
    pass


class Stock(Base):
    __tablename__ = "stocks"
    symbol: Mapped[str] = mapped_column(String(20), primary_key=True)
    name: Mapped[str | None] = mapped_column(String(200))
    series: Mapped[str | None] = mapped_column(String(10))
    isin: Mapped[str | None] = mapped_column(String(20))
    source: Mapped[str] = mapped_column(String(50), default="manual")  # seed | nse_import | manual
    active: Mapped[bool] = mapped_column(Boolean, default=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)


class PriceBar(Base):
    """Daily OHLCV cache. `close` is the unadjusted close, `adj_close` is split/dividend adjusted."""
    __tablename__ = "price_bars"
    __table_args__ = (UniqueConstraint("symbol", "date", name="uq_bar_symbol_date"),)
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    symbol: Mapped[str] = mapped_column(ForeignKey("stocks.symbol"), index=True)
    date: Mapped[date] = mapped_column(Date, index=True)
    open: Mapped[float] = mapped_column(Float)
    high: Mapped[float] = mapped_column(Float)
    low: Mapped[float] = mapped_column(Float)
    close: Mapped[float] = mapped_column(Float)
    adj_close: Mapped[float | None] = mapped_column(Float)
    volume: Mapped[float | None] = mapped_column(Float)
    source: Mapped[str] = mapped_column(String(50))
    fetched_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)


class DataCacheMeta(Base):
    __tablename__ = "data_cache_meta"
    symbol: Mapped[str] = mapped_column(ForeignKey("stocks.symbol"), primary_key=True)
    provider: Mapped[str] = mapped_column(String(50))
    last_fetch_at: Mapped[datetime | None] = mapped_column(DateTime)
    last_attempt_at: Mapped[datetime | None] = mapped_column(DateTime)
    first_date: Mapped[date | None] = mapped_column(Date)
    last_date: Mapped[date | None] = mapped_column(Date)
    n_rows: Mapped[int] = mapped_column(Integer, default=0)
    last_status: Mapped[str] = mapped_column(String(20), default="never")  # ok | error | never
    last_error: Mapped[str | None] = mapped_column(Text)
    quality_report: Mapped[dict | None] = mapped_column(JSON)


class TrainingRun(Base):
    __tablename__ = "training_runs"
    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    symbol: Mapped[str] = mapped_column(ForeignKey("stocks.symbol"), index=True)
    status: Mapped[str] = mapped_column(String(20), default="running")  # running | succeeded | failed
    algorithms: Mapped[list] = mapped_column(JSON)
    horizons: Mapped[list] = mapped_column(JSON)
    params: Mapped[dict] = mapped_column(JSON, default=dict)
    train_start: Mapped[date | None] = mapped_column(Date)
    train_end: Mapped[date | None] = mapped_column(Date)
    started_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)
    finished_at: Mapped[datetime | None] = mapped_column(DateTime)
    error: Mapped[str | None] = mapped_column(Text)
    summary: Mapped[dict | None] = mapped_column(JSON)


class ModelVersion(Base):
    """One fitted model for (symbol, algorithm, horizon) plus its walk-forward evaluation."""
    __tablename__ = "model_versions"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    run_id: Mapped[str] = mapped_column(ForeignKey("training_runs.id"), index=True)
    symbol: Mapped[str] = mapped_column(ForeignKey("stocks.symbol"), index=True)
    algorithm: Mapped[str] = mapped_column(String(40))
    horizon: Mapped[int] = mapped_column(Integer)
    version: Mapped[str] = mapped_column(String(120), unique=True)
    artifact_path: Mapped[str] = mapped_column(String(500))
    feature_names: Mapped[list] = mapped_column(JSON)
    train_start: Mapped[date] = mapped_column(Date)
    train_end: Mapped[date] = mapped_column(Date)
    n_train: Mapped[int] = mapped_column(Integer)
    metrics: Mapped[dict] = mapped_column(JSON)  # walk-forward out-of-sample metrics
    eligible: Mapped[bool] = mapped_column(Boolean, default=False)
    selection_note: Mapped[str | None] = mapped_column(Text)
    is_selected: Mapped[bool] = mapped_column(Boolean, default=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)

    backtest: Mapped[list["BacktestPrediction"]] = relationship(
        back_populates="model_version", cascade="all, delete-orphan"
    )


class BacktestPrediction(Base):
    """Historical out-of-sample (walk-forward) predictions. These are NOT forecasts."""
    __tablename__ = "backtest_predictions"
    __table_args__ = (UniqueConstraint("model_version_id", "as_of_date", name="uq_bt_mv_date"),)
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    model_version_id: Mapped[int] = mapped_column(ForeignKey("model_versions.id"), index=True)
    as_of_date: Mapped[date] = mapped_column(Date)
    target_date: Mapped[date] = mapped_column(Date)
    reference_price: Mapped[float] = mapped_column(Float)
    predicted_price: Mapped[float] = mapped_column(Float)
    actual_price: Mapped[float] = mapped_column(Float)
    fold: Mapped[int] = mapped_column(Integer)

    model_version: Mapped[ModelVersion] = relationship(back_populates="backtest")


class Prediction(Base):
    """A genuine forecast made at `created_at` using data through `as_of_date`."""
    __tablename__ = "predictions"
    __table_args__ = (
        UniqueConstraint("model_version_id", "as_of_date", "horizon", name="uq_pred_mv_asof_h"),
    )
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    symbol: Mapped[str] = mapped_column(ForeignKey("stocks.symbol"), index=True)
    model_version_id: Mapped[int] = mapped_column(ForeignKey("model_versions.id"), index=True)
    horizon: Mapped[int] = mapped_column(Integer)
    as_of_date: Mapped[date] = mapped_column(Date)
    target_date_est: Mapped[date] = mapped_column(Date)  # weekday estimate; NSE holidays ignored
    reference_close: Mapped[float] = mapped_column(Float)
    predicted_price: Mapped[float] = mapped_column(Float)
    predicted_change: Mapped[float] = mapped_column(Float)
    predicted_change_pct: Mapped[float] = mapped_column(Float)
    method: Mapped[str] = mapped_column(String(20), default="direct")
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)

    model_version: Mapped[ModelVersion] = relationship()
    outcome: Mapped["Outcome | None"] = relationship(
        back_populates="prediction", uselist=False, cascade="all, delete-orphan"
    )
    interval: Mapped["PredictionInterval | None"] = relationship(
        back_populates="prediction", uselist=False, cascade="all, delete-orphan"
    )


class PredictionInterval(Base):
    """Calibrated forecast ranges (in rupees) around the reference close for one prediction."""
    __tablename__ = "prediction_intervals"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    prediction_id: Mapped[int] = mapped_column(ForeignKey("predictions.id"), unique=True)
    method: Mapped[str] = mapped_column(String(20))  # fhs_xgb | fhs_ewma
    sigma_daily: Mapped[float] = mapped_column(Float)
    lo50: Mapped[float] = mapped_column(Float)
    hi50: Mapped[float] = mapped_column(Float)
    lo80: Mapped[float] = mapped_column(Float)
    hi80: Mapped[float] = mapped_column(Float)
    lo95: Mapped[float] = mapped_column(Float)
    hi95: Mapped[float] = mapped_column(Float)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)

    prediction: Mapped[Prediction] = relationship(back_populates="interval")


class Outcome(Base):
    __tablename__ = "outcomes"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    prediction_id: Mapped[int] = mapped_column(ForeignKey("predictions.id"), unique=True)
    actual_date: Mapped[date] = mapped_column(Date)
    actual_close: Mapped[float] = mapped_column(Float)
    error: Mapped[float] = mapped_column(Float)  # predicted - actual (rupees)
    abs_error: Mapped[float] = mapped_column(Float)
    pct_error: Mapped[float] = mapped_column(Float)  # (predicted - actual) / actual * 100
    direction_correct: Mapped[bool | None] = mapped_column(Boolean)  # None when undefined
    recorded_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)

    prediction: Mapped[Prediction] = relationship(back_populates="outcome")
