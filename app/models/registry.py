"""Algorithm registry: one common interface (sklearn-style estimators predicting the return target).

Add a model by registering a factory here; training/evaluation/prediction need no other change.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from app.models.baseline import DriftModel, NaiveModel

SEED = 42


@dataclass(frozen=True)
class Algorithm:
    name: str
    description: str
    factory: Callable[[], object]
    is_baseline: bool = False


def _ridge():
    # The scaler lives INSIDE the pipeline, so it is re-fitted on training rows of every fold only.
    return Pipeline([("scale", StandardScaler()), ("model", Ridge(alpha=10.0))])


def _rf():
    return RandomForestRegressor(n_estimators=200, max_depth=6, min_samples_leaf=10,
                                 max_features=0.5, n_jobs=-1, random_state=SEED)


def _xgb():
    from xgboost import XGBRegressor
    return XGBRegressor(n_estimators=250, learning_rate=0.03, max_depth=3, subsample=0.8,
                        colsample_bytree=0.8, min_child_weight=10, reg_lambda=5.0,
                        random_state=SEED, n_jobs=2, verbosity=0)


def _hgb():
    return HistGradientBoostingRegressor(max_iter=200, learning_rate=0.04, max_depth=3,
                                         min_samples_leaf=20, l2_regularization=1.0,
                                         random_state=SEED)


def _build() -> dict[str, Algorithm]:
    algos = [
        Algorithm("naive", "Next close = latest close (benchmark)", NaiveModel, True),
        Algorithm("drift", "Latest close grown by mean historical return", DriftModel, True),
        Algorithm("ridge", "Ridge regression on standardised features", _ridge),
        Algorithm("random_forest", "Random forest regressor", _rf),
    ]
    try:
        import xgboost  # noqa: F401
        algos.append(Algorithm("xgboost", "XGBoost gradient-boosted trees", _xgb))
    except ImportError:  # documented fallback: sklearn's gradient boosting
        algos.append(Algorithm("hist_gradient_boosting",
                               "sklearn HistGradientBoosting (xgboost not installed)", _hgb))
    return {a.name: a for a in algos}


ALGORITHMS: dict[str, Algorithm] = _build()
DEFAULT_ALGORITHMS: list[str] = list(ALGORITHMS)


def get_algorithm(name: str) -> Algorithm:
    try:
        return ALGORITHMS[name]
    except KeyError:
        raise ValueError(f"Unknown algorithm '{name}'. Available: {sorted(ALGORITHMS)}") from None
