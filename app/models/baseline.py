"""Baseline regressors (they predict the *return* target, see features/engineering.py).

NaiveModel  : next close == latest close  (predicted return 0). The benchmark every model must beat.
DriftModel  : latest close grown by the training-sample mean return (random walk with drift).

An ARIMA-style model was deliberately not added: ridge regression on lagged returns is already an
autoregressive model, and on daily equity returns the extra dependency has not earned its place.
"""
from __future__ import annotations

import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin


class NaiveModel(BaseEstimator, RegressorMixin):
    def fit(self, X, y=None):  # noqa: ANN001
        self.n_features_in_ = np.asarray(X).shape[1]
        return self

    def predict(self, X):  # noqa: ANN001
        return np.zeros(len(X), dtype=float)


class DriftModel(BaseEstimator, RegressorMixin):
    def fit(self, X, y):  # noqa: ANN001
        self.n_features_in_ = np.asarray(X).shape[1]
        self.mean_return_ = float(np.mean(y))
        return self

    def predict(self, X):  # noqa: ANN001
        return np.full(len(X), self.mean_return_, dtype=float)
