"""Chronological splitting, walk-forward behaviour, metrics and the selection rule."""
import numpy as np
import pandas as pd
import pytest
from sklearn.base import BaseEstimator, RegressorMixin

from app.core.errors import InsufficientDataError
from app.features.engineering import build_dataset
from app.models.baseline import DriftModel, NaiveModel
from app.models.evaluation import (
    MIN_OOS_ROWS, compute_metrics, relative_to_baseline, select_model, walk_forward_folds,
    walk_forward_predict,
)

SEEN: list[tuple[pd.Index, pd.Index]] = []


class Recorder(BaseEstimator, RegressorMixin):
    """Records the index of every training set and every predicted set."""
    def fit(self, X, y):
        self.train_idx_ = X.index
        return self

    def predict(self, X):
        SEEN.append((self.train_idx_, X.index))
        return np.zeros(len(X))


@pytest.mark.parametrize("h", [1, 5, 10])
def test_folds_are_chronological_contiguous_and_purged(h):
    folds = walk_forward_folds(1000, h, n_folds=6, test_size=21, min_train=250)
    assert len(folds) == 6
    for prev, nxt in zip(folds, folds[1:]):
        assert nxt.test_start == prev.test_end            # test blocks tile without gaps
        assert nxt.train_end > prev.train_end             # expanding window
    for f in folds:
        # training row i has target at i+h; it must be <= f.test_start (known at first test time)
        assert f.train_end - 1 + h <= f.test_start
        assert f.train_end <= f.test_start                # never train on test rows
    assert folds[-1].test_end == 1000


def test_folds_shrink_when_data_is_short_and_fail_when_too_short():
    assert len(walk_forward_folds(400, 1, 8, 21, 250)) < 8
    with pytest.raises(InsufficientDataError):
        walk_forward_folds(100, 1, 8, 21, 250)


def test_walk_forward_never_trains_on_test_period(prices):
    SEEN.clear()
    ds = build_dataset(prices, 5)
    folds = walk_forward_folds(len(ds.X), 5, 3, 21, 150)
    bt = walk_forward_predict(ds, Recorder(), folds)
    assert len(SEEN) == 3
    for train_idx, test_idx in SEEN:
        assert train_idx.max() < test_idx.min()
        assert train_idx.is_monotonic_increasing and test_idx.is_monotonic_increasing
        # target dates of the training rows are all before the first test as-of date
        tgt = ds.target_date.loc[train_idx]
        assert tgt.max() <= test_idx.min()
    assert bt["as_of_date"].is_monotonic_increasing and len(bt) == 63


def test_naive_walk_forward_reproduces_hold_last_price(prices):
    ds = build_dataset(prices, 1)
    bt = walk_forward_predict(ds, NaiveModel(), walk_forward_folds(len(ds.X), 1, 2, 21, 150))
    np.testing.assert_allclose(bt["predicted_price"], bt["reference_price"])
    assert (bt["target_date"] > bt["as_of_date"]).all()


def test_scaler_is_fit_on_training_rows_only(prices):
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import Ridge
    ds = build_dataset(prices, 1)
    f = walk_forward_folds(len(ds.X), 1, 2, 21, 150)[0]
    pipe = Pipeline([("s", StandardScaler()), ("m", Ridge())]).fit(ds.X.iloc[:f.train_end], ds.y.iloc[:f.train_end])
    np.testing.assert_allclose(pipe["s"].mean_, ds.X.iloc[:f.train_end].mean().values)


def test_metrics_values_and_definitions():
    actual, pred, ref = [100.0, 110.0, 105.0, 105.0], [102.0, 108.0, 104.0, 106.0], [99.0, 100.0, 110.0, 105.0]
    m = compute_metrics(actual, pred, ref)
    assert m["mae"] == pytest.approx(np.mean([2, 2, 1, 1]))
    assert m["rmse"] == pytest.approx(np.sqrt(np.mean([4, 4, 1, 1])))
    assert m["mape"] == pytest.approx(np.mean([2 / 100, 2 / 110, 1 / 105, 1 / 105]) * 100)
    # direction: rows 1-3 have non-zero actual & predicted change: up/up ✓, up/up ✓, down/down ✓
    # row 4: actual change 0 -> excluded
    assert m["n_directional"] == 3 and m["directional_accuracy"] == 1.0
    assert m["bias"] == pytest.approx(np.mean([2, -2, -1, 1]))


def test_mape_excludes_non_positive_actuals_and_naive_has_no_direction():
    m = compute_metrics([0.0, 100.0], [1.0, 90.0], [100.0, 100.0])
    assert m["mape"] == pytest.approx(10.0) and np.isfinite(m["mape"])
    assert compute_metrics([0.0], [1.0], [1.0])["mape"] is None
    naive = compute_metrics([101.0, 99.0], [100.0, 100.0], [100.0, 100.0])
    assert naive["directional_accuracy"] is None


def _m(mae, rmse, n=100):
    return {"mae": mae, "rmse": rmse, "n": n, "mape": 1.0, "directional_accuracy": 0.5}


def test_selection_requires_beating_naive_and_ignores_training_error():
    res = {"naive": _m(10, 12), "ridge": _m(10.2, 12.1), "xgboost": _m(9, 11), "random_forest": _m(9.5, 11.5)}
    best, verdicts = select_model(res)   # lowest RMSE among eligible wins
    assert best == "xgboost" and not verdicts["ridge"][0] and verdicts["random_forest"][0]


def test_selection_falls_back_to_naive_when_nothing_qualifies():
    res = {"naive": _m(10, 12), "ridge": _m(10.5, 12.5), "xgboost": _m(9.99, 11.99),
           "random_forest": _m(8, 9, n=MIN_OOS_ROWS - 1)}
    best, verdicts = select_model(res)
    assert best == "naive"
    assert "out-of-sample rows" in verdicts["random_forest"][1]
    assert relative_to_baseline(res["ridge"], res["naive"])["mae_improvement_pct"] < 0


def test_drift_model_predicts_training_mean():
    m = DriftModel().fit(np.zeros((5, 2)), np.array([0.01, 0.03, 0.02, 0.0, 0.04]))
    np.testing.assert_allclose(m.predict(np.zeros((3, 2))), 0.02)
