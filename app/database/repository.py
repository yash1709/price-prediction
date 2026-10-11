"""Data-access helpers. All SQL lives here so services stay storage-agnostic."""
from __future__ import annotations

from datetime import date, datetime

import pandas as pd
from sqlalchemy import delete, func, select, update
from sqlalchemy.orm import Session

from app.database.models import (
    BacktestPrediction, DataCacheMeta, ModelVersion, Outcome, Prediction, PredictionInterval,
    PriceBar, Stock, TrainingRun, utcnow,
)

OHLCV_COLS = ["open", "high", "low", "close", "adj_close", "volume"]


class Repository:
    def __init__(self, session: Session):
        self.s = session

    # ---- stocks -----------------------------------------------------------------------------
    def get_stock(self, symbol: str) -> Stock | None:
        return self.s.get(Stock, symbol)

    def ensure_stock(self, symbol: str, name: str | None = None, source: str = "manual",
                     **extra) -> Stock:
        stock = self.get_stock(symbol)
        if stock is None:
            stock = Stock(symbol=symbol, name=name, source=source, **extra)
            self.s.add(stock)
            self.s.flush()
        return stock

    def upsert_stocks(self, rows: list[dict]) -> int:
        """Insert unknown symbols, refresh name/series/isin of known ones. Returns #inserted."""
        existing = {st.symbol: st for st in self.s.scalars(select(Stock))}
        inserted = 0
        for r in rows:
            st = existing.get(r["symbol"])
            if st is None:
                self.s.add(Stock(**r))
                inserted += 1
            else:
                for k in ("name", "series", "isin"):
                    if r.get(k):
                        setattr(st, k, r[k])
                if r.get("source") == "nse_import":
                    st.source = "nse_import"
        self.s.flush()
        return inserted

    def search_stocks(self, q: str | None, limit: int, offset: int) -> tuple[int, list[Stock]]:
        stmt = select(Stock).where(Stock.active.is_(True))
        if q:
            like = f"%{q.strip().upper()}%"
            stmt = stmt.where(Stock.symbol.like(like) | func.upper(Stock.name).like(like))
        total = self.s.scalar(select(func.count()).select_from(stmt.subquery())) or 0
        rows = list(self.s.scalars(stmt.order_by(Stock.symbol).limit(limit).offset(offset)))
        return total, rows

    # ---- price bars -------------------------------------------------------------------------
    def load_bars(self, symbol: str, start: date | None = None, end: date | None = None
                  ) -> pd.DataFrame:
        stmt = select(PriceBar).where(PriceBar.symbol == symbol)
        if start:
            stmt = stmt.where(PriceBar.date >= start)
        if end:
            stmt = stmt.where(PriceBar.date <= end)
        rows = self.s.scalars(stmt.order_by(PriceBar.date)).all()
        if not rows:
            return pd.DataFrame(columns=OHLCV_COLS, index=pd.DatetimeIndex([], name="date"))
        df = pd.DataFrame(
            [{"date": r.date, **{c: getattr(r, c) for c in OHLCV_COLS}} for r in rows]
        )
        df["date"] = pd.to_datetime(df["date"])
        return df.set_index("date")

    def upsert_bars(self, symbol: str, df: pd.DataFrame, source: str) -> int:
        """Insert/update bars keyed on (symbol, date). Returns number of rows written."""
        if df.empty:
            return 0
        dates = [d.date() for d in df.index]
        existing = {
            b.date: b for b in self.s.scalars(
                select(PriceBar).where(PriceBar.symbol == symbol, PriceBar.date.in_(dates))
            )
        }
        now = utcnow()
        n = 0
        for ts, row in df.iterrows():
            d = ts.date()
            vals = {c: (None if pd.isna(row.get(c)) else float(row[c])) for c in OHLCV_COLS}
            bar = existing.get(d)
            if bar is None:
                self.s.add(PriceBar(symbol=symbol, date=d, source=source, fetched_at=now, **vals))
            else:
                for k, v in vals.items():
                    setattr(bar, k, v)
                bar.source, bar.fetched_at = source, now
            n += 1
        self.s.flush()
        return n

    def delete_bars(self, symbol: str) -> None:
        self.s.execute(delete(PriceBar).where(PriceBar.symbol == symbol))

    def get_meta(self, symbol: str) -> DataCacheMeta | None:
        return self.s.get(DataCacheMeta, symbol)

    def save_meta(self, symbol: str, provider: str, **fields) -> DataCacheMeta:
        meta = self.get_meta(symbol)
        if meta is None:
            meta = DataCacheMeta(symbol=symbol, provider=provider)
            self.s.add(meta)
        meta.provider = provider
        for k, v in fields.items():
            setattr(meta, k, v)
        self.s.flush()
        return meta

    def refresh_meta_range(self, symbol: str) -> None:
        first, last, n = self.s.execute(
            select(func.min(PriceBar.date), func.max(PriceBar.date), func.count())
            .where(PriceBar.symbol == symbol)
        ).one()
        self.save_meta(symbol, self.get_meta(symbol).provider, first_date=first, last_date=last,
                       n_rows=n)

    # ---- training / models ------------------------------------------------------------------
    def add_run(self, run: TrainingRun) -> TrainingRun:
        self.s.add(run)
        self.s.flush()
        return run

    def get_run(self, run_id: str) -> TrainingRun | None:
        return self.s.get(TrainingRun, run_id)

    def add_model_version(self, mv: ModelVersion, backtest: list[BacktestPrediction]) -> ModelVersion:
        mv.backtest = backtest
        self.s.add(mv)
        self.s.flush()
        return mv

    def get_model_version(self, mv_id: int) -> ModelVersion | None:
        return self.s.get(ModelVersion, mv_id)

    def list_model_versions(self, symbol: str, horizon: int | None = None,
                            selected_only: bool = False, run_id: str | None = None
                            ) -> list[ModelVersion]:
        stmt = select(ModelVersion).where(ModelVersion.symbol == symbol)
        if horizon is not None:
            stmt = stmt.where(ModelVersion.horizon == horizon)
        if selected_only:
            stmt = stmt.where(ModelVersion.is_selected.is_(True))
        if run_id:
            stmt = stmt.where(ModelVersion.run_id == run_id)
        return list(self.s.scalars(stmt.order_by(ModelVersion.created_at.desc(), ModelVersion.id)))

    def clear_selected(self, symbol: str, horizon: int) -> None:
        self.s.execute(
            update(ModelVersion)
            .where(ModelVersion.symbol == symbol, ModelVersion.horizon == horizon)
            .values(is_selected=False)
        )

    def selected_model(self, symbol: str, horizon: int) -> ModelVersion | None:
        return self.s.scalars(
            select(ModelVersion)
            .where(ModelVersion.symbol == symbol, ModelVersion.horizon == horizon,
                   ModelVersion.is_selected.is_(True))
            .order_by(ModelVersion.created_at.desc()).limit(1)
        ).first()

    def backtest_rows(self, mv_id: int) -> list[BacktestPrediction]:
        return list(self.s.scalars(
            select(BacktestPrediction).where(BacktestPrediction.model_version_id == mv_id)
            .order_by(BacktestPrediction.as_of_date)
        ))

    # ---- predictions / outcomes -------------------------------------------------------------
    def find_prediction(self, mv_id: int, as_of: date, horizon: int) -> Prediction | None:
        return self.s.scalars(select(Prediction).where(
            Prediction.model_version_id == mv_id, Prediction.as_of_date == as_of,
            Prediction.horizon == horizon)).first()

    def add_prediction(self, p: Prediction) -> Prediction:
        self.s.add(p)
        self.s.flush()
        return p

    def list_predictions(self, symbol: str, limit: int = 200, horizon: int | None = None
                         ) -> list[Prediction]:
        stmt = select(Prediction).where(Prediction.symbol == symbol)
        if horizon is not None:
            stmt = stmt.where(Prediction.horizon == horizon)
        return list(self.s.scalars(
            stmt.order_by(Prediction.as_of_date.desc(), Prediction.id.desc()).limit(limit)))

    def unresolved_predictions(self, symbol: str) -> list[Prediction]:
        return list(self.s.scalars(
            select(Prediction).outerjoin(Outcome).where(
                Prediction.symbol == symbol, Outcome.id.is_(None))
            .order_by(Prediction.as_of_date)))

    def add_interval(self, iv: PredictionInterval) -> PredictionInterval:
        self.s.add(iv)
        self.s.flush()
        return iv

    def add_outcome(self, o: Outcome) -> Outcome:
        self.s.add(o)
        self.s.flush()
        return o


def utc_iso(dt: datetime | None) -> str | None:
    return dt.isoformat() + "Z" if dt else None
