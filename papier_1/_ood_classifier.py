import numpy as np
from aiirw import AI_IRW
from scipy.spatial.distance import cdist
from sklearn.base import ClassifierMixin
from typing import Callable

__available_similarities__ = ["mahalanobis", "IRW"]


def partial_wrapper(f, **kwargs):
    def wrapper(*args):
        return f(*args, **kwargs).flatten()

    return wrapper


class OODDetector(ClassifierMixin):
    def __init__(
        self, tau: float, base_distribution: np.ndarray, similarity: str = "mahalanobis"
    ):
        super().__init__()

        assert similarity in __available_similarities__, "Similarity is not available"

        self.tau = tau
        self.similarity = similarity

        self.base_distribution = base_distribution

        if base_distribution is None:
            raise ValueError("Base distribution is not provided")

    def clone(self):
        return self.__class__(
            tau=self.tau,
            base_distribution=self.base_distribution,
            similarity=self.similarity,
        )

    def _prefit(self):
        # Faire gaffe a pred =0 ou pred =1
        if self.similarity == "mahalanobis":
            m = np.mean(self.base_distribution, axis=0).reshape(
                1, self.base_distribution.shape[1]
            )
            VI = np.linalg.inv(np.cov(self.base_distribution.T))
            self._compute_sim = partial_wrapper(
                cdist, XB=m, metric="mahalanobis", VI=VI
            )
        elif self.similarity == "IRW":
            self._compute_sim = lambda x: 1 - partial_wrapper(
                AI_IRW,
                X=self.base_distribution,
                n_dirs=1000,
                AI=False,
                robust=False,
                random_state=None,
            )(x)
        return None

    def fit(self, X: np.ndarray, y: np.ndarray):
        self._prefit()
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if len(X.shape) == 1:
            X = X[None, :]
        return (self._compute_sim(X) > self.tau).astype(int)

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        y_pred = self.predict(X)
        return np.mean((y_pred == y).astype(int))

    def get_params(self, deep=True):
        return {
            "tau": self.tau,
            "similarity": self.similarity,
            "base_distribution": self.base_distribution,
        }

    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if len(X.shape) == 1:
            X = X[None, :]
        return self._compute_sim(X)

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        return self.predict_proba(X)
