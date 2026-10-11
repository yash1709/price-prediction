"""Train, walk-forward-evaluate, persist and select models for one symbol."""
from __future__ import annotations

import logging
import uuid
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Callable

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone

from app.core.config import Settings
from app.core.errors import InsufficientDataError
from app.database.models import BacktestPrediction, ModelVersion, TrainingRun, utcnow
from app.database.repository import Repository
from app.features.engineering import FEATURE_NAMES, build_dataset
from app.models.evaluation import (
    compute_metrics, relative_to_baseline, select_model, walk_forward_folds,
    walk_forward_predict,
)
from app.models.registry import ALGORITHMS, get_algorithm

log = logging.getLogger(__name__)
ProgressFn = Callable[[float, str], None]


def artifact_path(settings: Settings, symbol: str, version: str) -> Path:
    d = settings.artifacts_path / symbol
    d.mkdir(parents=True, exist_ok=True)
    return d / f"{version}.joblib"


def load_artifact(path: str | Path) -> dict:
    return joblib.load(path)


def train_symbol(repo: Repository, settings: Settings, symbol: str,
                 algorithms: list[str] | None = None, horizons: list[int] | None = None,
                 start: date | None = None, end: date | None = None,
                 progress: ProgressFn | None = None, run_id: str | None = None) -> TrainingRun:
    """Walk-forward evaluate every algorithm per horizon, refit on all data, save + select."""
    algorithms = list(dict.fromkeys(algorithms or list(ALGORITHMS)))
    for a in algorithms:
        get_algorithm(a)  # validate names early
    if "naive" not in algorithms:
        algorithms.insert(0, "naive")  # the benchmark is mandatory for fair comparison
    horizons = sorted(set(horizons or [1]))
    notify = progress or (lambda f, m: None)

    df = repo.load_bars(symbol, start, end)
    if df.empty:
        raise InsufficientDataError(f"No cached price data for {symbol}; fetch data first.")

    run = repo.add_run(TrainingRun(
        id=run_id or str(uuid.uuid4()), symbol=symbol, status="running", algorithms=algorithms,
        horizons=horizons, train_start=df.index[0].date(), train_end=df.index[-1].date(),
        params={"walk_forward_folds": settings.walk_forward_folds,
                "test_size": settings.walk_forward_test_size,
                "min_train_rows": settings.min_train_rows, "features": FEATURE_NAMES,
                "target": "return = adj_close[t+h]/adj_close[t]-1"}))
    repo.s.commit()  # make the 'running' row visible to other sessions
    try:
        total_steps = len(horizons) * len(algorithms)
        step = 0
        summary: dict = {"horizons": {}}
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        for h in horizons:
            ds = build_dataset(df, h)
            folds = walk_forward_folds(len(ds.X), h, settings.walk_forward_folds,
                                       settings.walk_forward_test_size, settings.min_train_rows)
            results: dict[str, dict] = {}
            backtests: dict[str, pd.DataFrame] = {}
            fitted: dict[str, object] = {}
            for name in algorithms:
                notify(step / total_steps, f"h={h}: evaluating {name}")
                est = get_algorithm(name).factory()
                bt = walk_forward_predict(ds, est, folds)
                results[name] = compute_metrics(bt["actual_price"], bt["predicted_price"],
                                                bt["reference_price"])
                backtests[name] = bt
                final = clone(est)
                final.fit(ds.X, ds.y)
                fitted[name] = final
                step += 1
            naive_metrics = results["naive"]
            for name in results:
                results[name]["vs_naive"] = relative_to_baseline(results[name], naive_metrics)
            best, verdicts = select_model(results)
            repo.clear_selected(symbol, h)
            summary["horizons"][str(h)] = {"selected": best, "metrics": results,
                                           "n_oos_rows": results[best]["n"],
                                           "n_train_rows": int(len(ds.X)), "n_folds": len(folds)}
            for name in algorithms:
                version = f"{symbol}-{name}-h{h}-{stamp}-{run.id[:6]}"
                path = artifact_path(settings, symbol, version)
                joblib.dump({
                    "estimator": fitted[name], "feature_names": FEATURE_NAMES, "algorithm": name,
                    "horizon": h, "symbol": symbol, "version": version, "target": "return",
                    "train_start": ds.X.index[0].date().isoformat(),
                    "train_end": ds.X.index[-1].date().isoformat(),
                }, path)
                ok, note = verdicts[name]
                if name == "naive" and best == "naive":
                    note = "Selected: no model beat the naive baseline out-of-sample."
                bt_rows = [
                    BacktestPrediction(
                        as_of_date=r.as_of_date.date(), target_date=pd.Timestamp(r.target_date).date(),
                        reference_price=float(r.reference_price),
                        predicted_price=float(r.predicted_price),
                        actual_price=float(r.actual_price), fold=int(r.fold))
                    for r in backtests[name].itertuples()
                ]
                repo.add_model_version(ModelVersion(
                    run_id=run.id, symbol=symbol, algorithm=name, horizon=h, version=version,
                    artifact_path=str(path), feature_names=FEATURE_NAMES,
                    train_start=ds.X.index[0].date(), train_end=ds.X.index[-1].date(),
                    n_train=int(len(ds.X)), metrics=results[name], eligible=ok,
                    selection_note=note, is_selected=(name == best)), bt_rows)
            repo.s.commit()  # release the write lock between horizons (concurrent workers)
        run.status, run.finished_at, run.summary = "succeeded", utcnow(), summary
        repo.s.commit()
        notify(1.0, "done")
        return run
    except Exception as exc:
        repo.s.rollback()
        failed = repo.get_run(run.id)
        if failed is not None:
            failed.status, failed.finished_at, failed.error = "failed", utcnow(), str(exc)[:2000]
            repo.s.commit()
        log.exception("Training failed for %s", symbol)
        raise
