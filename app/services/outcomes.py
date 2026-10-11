"""Record actual closes for past forecasts and summarise realised performance."""
from __future__ import annotations

import numpy as np

from app.database.models import Outcome
from app.database.repository import Repository
from app.models.evaluation import compute_metrics


def resolve_outcomes(repo: Repository, symbol: str) -> int:
    """For each unresolved forecast, find the h-th trading bar after `as_of_date` in the cached
    data (holiday-proof: it counts real bars, not calendar days) and store the error."""
    pending = repo.unresolved_predictions(symbol)
    if not pending:
        return 0
    bars = repo.load_bars(symbol)
    n = 0
    for p in pending:
        later = bars[bars.index > p.as_of_date.isoformat()]
        if len(later) < p.horizon:
            continue  # the target session has not happened (or not been fetched) yet
        row = later.iloc[p.horizon - 1]
        actual = float(row["close"])
        err = p.predicted_price - actual
        d_act, d_pred = actual - p.reference_close, p.predicted_price - p.reference_close
        correct = None if (d_act == 0 or d_pred == 0) else bool(np.sign(d_act) == np.sign(d_pred))
        repo.add_outcome(Outcome(
            prediction_id=p.id, actual_date=later.index[p.horizon - 1].date(), actual_close=actual,
            error=err, abs_error=abs(err), pct_error=err / actual * 100, direction_correct=correct))
        n += 1
    return n


def performance_summary(repo: Repository, symbol: str) -> list[dict]:
    """Realised (live) forecast accuracy per model version, compared with the naive forecast
    (= reference close) on exactly the same resolved rows."""
    preds = [p for p in repo.list_predictions(symbol, limit=10_000) if p.outcome is not None]
    groups: dict[int, list] = {}
    for p in preds:
        groups.setdefault(p.model_version_id, []).append(p)
    out = []
    for mv_id, rows in groups.items():
        actual = [p.outcome.actual_close for p in rows]
        pred = [p.predicted_price for p in rows]
        ref = [p.reference_close for p in rows]
        m = compute_metrics(actual, pred, ref)
        naive = compute_metrics(actual, ref, ref)
        mv = rows[0].model_version
        out.append({
            "model_version_id": mv_id, "model_version": mv.version, "algorithm": mv.algorithm,
            "horizon": mv.horizon, "metrics": m, "naive_metrics": naive,
            "beats_naive_mae": m["mae"] < naive["mae"],
            "note": "Small samples are noisy; do not over-interpret." if m["n"] < 30 else None,
        })
    return sorted(out, key=lambda r: (r["horizon"], r["model_version_id"]))
