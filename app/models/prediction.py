"""Generate and persist genuine forecasts from the selected (or a specific) model version."""
from __future__ import annotations

import logging
from datetime import date, datetime

import numpy as np
import pandas as pd

from app.core.errors import AppError, NoModelError, NotFoundError
from app.database.models import ModelVersion, Prediction, PredictionInterval
from app.database.repository import Repository
from app.features.engineering import latest_features
from app.models.intervals import Z80
from app.models.training import load_artifact

log = logging.getLogger(__name__)
STALE_DATA_DAYS = 5
DISCLAIMER = ("Research forecast, not investment advice. Backtest results do not guarantee future "
              "accuracy; the naive baseline is often hard to beat.")


class InvalidPredictionError(AppError):
    code, status_code = "invalid_prediction", 500


def estimate_target_date(as_of: date, horizon: int) -> date:
    """h-th weekday after as_of. NSE holidays are NOT modelled, so this is an estimate only."""
    return pd.Timestamp(np.busday_offset(as_of.isoformat(), horizon, roll="forward")).date()


def attach_interval(repo: Repository, pred: Prediction, df: pd.DataFrame, interval_fn) -> None:
    """Compute and store the calibrated forecast ranges once per prediction (idempotent)."""
    if interval_fn is None or pred.interval is not None:
        return
    res = interval_fn(df, pred.horizon, symbol=pred.symbol)
    if res is None:
        return
    ref = pred.reference_close
    r = {lv: (ref * (1 + lo), ref * (1 + hi)) for lv, (lo, hi) in res.ranges.items()}
    repo.add_interval(PredictionInterval(
        prediction_id=pred.id, method=res.method, sigma_daily=res.sigma_daily,
        lo50=r[0.5][0], hi50=r[0.5][1], lo80=r[0.8][0], hi80=r[0.8][1],
        lo95=r[0.95][0], hi95=r[0.95][1]))
    repo.s.refresh(pred)


def predict_for_symbol(repo: Repository, symbol: str, horizons: list[int] | None = None,
                       model_version_id: int | None = None, today: date | None = None,
                       interval_fn=None) -> tuple[list[dict], list[str]]:
    """Returns (forecast dicts, warnings). Direct multi-horizon: each horizon uses its own model
    trained on that horizon's target; no predicted value is ever fed back as an input."""
    horizons = sorted(set(horizons or [1]))
    df = repo.load_bars(symbol)
    if df.empty:
        raise NotFoundError(f"No cached price data for {symbol}.")
    as_of_ts, X = latest_features(df)
    as_of = as_of_ts.date()
    ref_close = float(df["close"].iloc[-1])  # raw (unadjusted) latest close, in rupees
    warnings: list[str] = []
    if ((today or date.today()) - as_of).days > STALE_DATA_DAYS:
        warnings.append(f"Latest data is from {as_of}; refresh data before relying on this forecast.")

    out: list[dict] = []
    for h in horizons:
        mv: ModelVersion | None
        if model_version_id is not None:
            mv = repo.get_model_version(model_version_id)
            if mv is None or mv.symbol != symbol:
                raise NotFoundError(f"Model version {model_version_id} not found for {symbol}.")
            if mv.horizon != h:
                raise NoModelError(f"Model version {mv.id} was trained for horizon {mv.horizon}, "
                                   f"not {h}.")
        else:
            mv = repo.selected_model(symbol, h)
            if mv is None:
                raise NoModelError(
                    f"No trained model for {symbol} at horizon {h}. Train with horizon {h} first.")
        art = load_artifact(mv.artifact_path)
        if list(art["feature_names"]) != list(mv.feature_names):
            raise InvalidPredictionError("Artifact feature list does not match its DB record.")
        ret = float(np.asarray(art["estimator"].predict(X[art["feature_names"]]))[0])
        price = ref_close * (1.0 + ret)
        if not (np.isfinite(ret) and np.isfinite(price) and price > 0):
            raise InvalidPredictionError(f"Model {mv.version} produced an invalid forecast ({price}).")

        existing = repo.find_prediction(mv.id, as_of, h)
        if existing is not None:
            pred = existing  # idempotent: same model + same data => same forecast, no duplicate row
        else:
            pred = repo.add_prediction(Prediction(
                symbol=symbol, model_version_id=mv.id, horizon=h, as_of_date=as_of,
                target_date_est=estimate_target_date(as_of, h), reference_close=ref_close,
                predicted_price=price, predicted_change=price - ref_close,
                predicted_change_pct=(price / ref_close - 1) * 100, method="direct"))
        attach_interval(repo, pred, df, interval_fn)
        if (as_of - date.fromisoformat(art["train_end"])).days > 45:
            warnings.append(f"Model {mv.version} was trained on data through {art['train_end']}; "
                            f"consider retraining.")
        out.append(prediction_to_dict(pred, mv))
    return out, warnings


def prediction_to_dict(p: Prediction, mv: ModelVersion | None = None) -> dict:
    mv = mv or p.model_version
    d = {
        "id": p.id, "symbol": p.symbol, "horizon": p.horizon, "kind": "forecast",
        "method": p.method, "as_of_date": p.as_of_date.isoformat(),
        "target_date_estimate": p.target_date_est.isoformat(),
        "created_at": p.created_at.isoformat() + "Z", "reference_close": p.reference_close,
        "predicted_price": p.predicted_price, "predicted_change": p.predicted_change,
        "predicted_change_pct": p.predicted_change_pct,
        "model_version_id": mv.id, "model_version": mv.version, "algorithm": mv.algorithm,
        "model_is_baseline": mv.algorithm in ("naive", "drift"),
        "model_is_selected": mv.is_selected,
        "uncertainty_note": (
            "Next-day forecast; typical error is comparable to a normal daily move."
            if p.horizon == 1 else
            f"{p.horizon}-trading-day direct forecast: uncertainty grows with horizon; "
            f"treat as a rough central estimate."),
        "outcome": None, "interval": None,
    }
    iv = p.interval
    if iv is not None:
        ref = p.reference_close
        d["interval"] = {
            "method": iv.method,
            "implied_daily_vol_pct": (iv.hi80 - iv.lo80) / ref / (2 * Z80) / (p.horizon ** 0.5) * 100,
            "ranges": {str(int(lv * 100)): {"lower": lo, "upper": hi,
                                            "lower_pct": (lo / ref - 1) * 100,
                                            "upper_pct": (hi / ref - 1) * 100}
                       for lv, lo, hi in ((0.5, iv.lo50, iv.hi50), (0.8, iv.lo80, iv.hi80),
                                          (0.95, iv.lo95, iv.hi95))},
            "note": "Calibrated ranges around the latest close: the actual close should fall inside "
                    "the 80% range about 4 times out of 5 (79.9% out-of-sample for Nifty 200 stocks, "
                    "2022-2026; see `validation` for this stock's group). They describe uncertainty, "
                    "not direction.",
        }
    o = p.outcome
    if o is not None:
        d["outcome"] = {
            "actual_date": o.actual_date.isoformat(), "actual_close": o.actual_close,
            "error": o.error, "abs_error": o.abs_error, "pct_error": o.pct_error,
            "direction_correct": o.direction_correct,
            "recorded_at": o.recorded_at.isoformat() + "Z"}
        if iv is not None:
            d["outcome"].update({
                "within_range_50": iv.lo50 <= o.actual_close <= iv.hi50,
                "within_range_80": iv.lo80 <= o.actual_close <= iv.hi80,
                "within_range_95": iv.lo95 <= o.actual_close <= iv.hi95})
    return d
