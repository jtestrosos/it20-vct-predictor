"""
Team-Symmetric model wrapper.
Shared between train_model.py and app.py so that pickle can resolve the class.
"""

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, clone


class TeamSymmetric(ClassifierMixin, BaseEstimator):
    """Trains on each map twice (as-is and with Team A/B swapped) and averages
    p(x) with 1 - p(swap(x)), so the prediction cannot depend on which team
    happens to be called 'Team A'. Swapping = negating the Team A-minus-Team B
    features (Pick_Advantage and Agent_*); Map_* features are left untouched."""

    def __init__(self, estimator=None, flip_idx=None):
        self.estimator = estimator
        self.flip_idx = flip_idx

    def _swap(self, Xv):
        Xs = Xv.copy()
        Xs[:, self.flip_idx] *= -1
        return Xs

    def fit(self, X, y):
        Xv, yv = np.asarray(X, dtype=float), np.asarray(y)
        self.estimator_ = clone(self.estimator).fit(
            np.vstack([Xv, self._swap(Xv)]), np.concatenate([yv, 1 - yv]))
        self.classes_ = np.array([0, 1])
        return self

    def predict_proba(self, X):
        Xv = np.asarray(X, dtype=float)
        p = 0.5 * (self.estimator_.predict_proba(Xv)[:, 1]
                    + 1 - self.estimator_.predict_proba(self._swap(Xv))[:, 1])
        return np.column_stack([1 - p, p])

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] > 0.5).astype(int)

    @property
    def feature_importances_(self):
        return self.estimator_.feature_importances_
