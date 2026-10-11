"""Walk-forward evaluation, metrics and the model-selection rule.

Walk-forward (expanding window): for fold k the model is fitted on rows [0, s_k - h + 1) and
predicts rows [s_k, s_k + test_size). The `h - 1` row purge guarantees that no training target
(which looks h sessions ahead) uses a price that is unknown at the first test prediction time.
Rows are never shuffled.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.base import clone

from app.core.errors import InsufficientDataError
from app.features.engineering import Dataset

MIN_IMPROVEMENT_PCT = 0.5   # a model must beat naive MAE and RMSE by at least this much (relative)
MIN_OOS_ROWS = 40           # fewer out-of-sample rows than this cannot qualify a model


@dataclass(frozen=True)
class Fold:
    index: int
    train_end: int   # exclusive
    test_start: int
    test_end: int    # exclusive


def walk_forward_folds(n_rows: int, horizon: int, n_folds: int, test_size: int,
                       min_train: int) -> list[Fold]:
    """Chronological expanding-window folds over the most recent `n_folds * test_size` rows."""
    if n_rows <= 0 or test_size <= 0 or n_folds <= 0:
        raise InsufficientDataError("Invalid walk-forward configuration or empty dataset.")
    k = n_folds
    while k > 0 and (n_rows - k * test_size) - (horizon - 1) < min_train:
        k -= 1
    if k < 1:
        raise InsufficientDataError(
            f"Need at least {min_train + test_size + horizon - 1} usable rows for walk-forward "
            f"evaluation at horizon {horizon}; have {n_rows}.")
    first = n_rows - k * test_size
    return [Fold(i, first + i * test_size - (horizon - 1), first + i * test_size,
                 first + (i + 1) * test_size) for i in range(k)]


def walk_forward_predict(ds: Dataset, estimator, folds: list[Fold]) -> pd.DataFrame:
    """Out-of-sample predictions for every test row; estimator is cloned & refit per fold."""
    parts = []
    for f in folds:
        model = clone(estimator)
        model.fit(ds.X.iloc[:f.train_end], ds.y.iloc[:f.train_end])
        sl = slice(f.test_start, f.test_end)
        pred_ret = np.asarray(model.predict(ds.X.iloc[sl]), dtype=float)
        ref = ds.ref_price.iloc[sl]
        parts.append(pd.DataFrame({
            "as_of_date": ds.X.index[sl], "target_date": ds.target_date.iloc[sl].values,
            "reference_price": ref.values,
            "predicted_price": ref.values * (1 + pred_ret),
            "actual_price": ds.actual_price.iloc[sl].values, "fold": f.index,
        }))
    return pd.concat(parts, ignore_index=True)


def compute_metrics(actual, predicted, reference) -> dict:
    """MAE, RMSE (rupees), MAPE (%), directional accuracy and bias.

    * MAPE excludes rows with actual <= 0 (undefined / meaningless for prices) and is only
      meaningful for strictly positive series; it over-weights low-priced stocks.
    * Directional accuracy = share of rows where sign(pred - ref) == sign(actual - ref), counted
      only where both changes are non-zero. A model that always predicts "no change" (naive) has
      no direction, so its value is None rather than a misleading 0%/50%.
    """
    a, p, r = (np.asarray(x, dtype=float) for x in (actual, predicted, reference))
    err = p - a
    pos = a > 0
    d_act, d_pred = a - r, p - r
    dmask = (d_act != 0) & (d_pred != 0)
    return {
        "n": int(len(a)),
        "mae": float(np.mean(np.abs(err))),
        "rmse": float(np.sqrt(np.mean(err ** 2))),
        "mape": float(np.mean(np.abs(err[pos] / a[pos])) * 100) if pos.any() else None,
        "directional_accuracy": (float(np.mean(np.sign(d_act[dmask]) == np.sign(d_pred[dmask])))
                                 if dmask.any() else None),
        "n_directional": int(dmask.sum()),
        "bias": float(np.mean(err)),
    }


def relative_to_baseline(metrics: dict, naive: dict) -> dict:
    """% improvement over naive (positive = better than naive)."""
    out = {}
    for k in ("mae", "rmse"):
        out[f"{k}_improvement_pct"] = (
            (naive[k] - metrics[k]) / naive[k] * 100 if naive[k] > 0 else None)
    return out


def select_model(results: dict[str, dict], baseline_name: str = "naive"
                 ) -> tuple[str, dict[str, tuple[bool, str]]]:
    """Predefined selection rule.

    A non-naive model is ELIGIBLE iff, on the out-of-sample walk-forward rows, it has at least
    MIN_OOS_ROWS rows and beats naive on BOTH MAE and RMSE by >= MIN_IMPROVEMENT_PCT.
    Among eligible models pick the lowest RMSE (MAE breaks ties). If none is eligible the naive
    baseline is selected. Training error plays no part.
    """
    naive = results[baseline_name]
    verdicts: dict[str, tuple[bool, str]] = {baseline_name: (True, "Benchmark / fallback.")}
    for name, m in results.items():
        if name == baseline_name:
            continue
        rel = relative_to_baseline(m, naive)
        mae_i, rmse_i = rel["mae_improvement_pct"], rel["rmse_improvement_pct"]
        if m["n"] < MIN_OOS_ROWS:
            verdicts[name] = (False, f"Only {m['n']} out-of-sample rows (< {MIN_OOS_ROWS}).")
        elif mae_i is None or rmse_i is None or mae_i < MIN_IMPROVEMENT_PCT or rmse_i < MIN_IMPROVEMENT_PCT:
            verdicts[name] = (False, f"Does not beat naive by >= {MIN_IMPROVEMENT_PCT}% on both "
                                     f"MAE ({_fmt(mae_i)}) and RMSE ({_fmt(rmse_i)}).")
        else:
            verdicts[name] = (True, f"Beats naive: MAE {mae_i:+.2f}%, RMSE {rmse_i:+.2f}%.")
    eligible = [n for n, (ok, _) in verdicts.items() if ok and n != baseline_name]
    if not eligible:
        return baseline_name, verdicts
    best = min(eligible, key=lambda n: (results[n]["rmse"], results[n]["mae"]))
    return best, verdicts


def _fmt(v: float | None) -> str:
    return "n/a" if v is None else f"{v:+.2f}%"
