import numpy as np
from aiirw import AI_IRW
from scipy.spatial.distance import cdist
from sklearn.base import ClassifierMixin

__available_similarities__ = ["mahalanobis", "IRW"]


def partial_wrapper(f, **kwargs):
    def wrapper(*args):
        return f(*args, **kwargs).flatten()

    return wrapper


class OODDetector(ClassifierMixin):
    def __init__(self, tau: float, similarity: str = "mahalanobis"):
        super().__init__()

        assert similarity in __available_similarities__, "Similarity is not available"

        self.tau = tau
        self.similarity = similarity

    def fit(self, base_distribution: np.ndarray):
        # Faire gaffe a pred =0 ou pred =1
        self.base_distribution = base_distribution
        if self.similarity == "mahalanobis":
            m = np.mean(base_distribution, axis=0).reshape(
                1, base_distribution.shape[1]
            )
            VI = np.linalg.inv(np.cov(base_distribution.T))
            self._compute_sim = partial_wrapper(
                cdist, XB=m, metric="mahalanobis", VI=VI
            )
        elif self.similarity == "IRW":
            self._compute_sim = partial_wrapper(
                AI_IRW,
                X=base_distribution,
                n_dirs=1000,
                AI=False,
                robust=False,
                random_state=None,
            )
        return None

    def predict(self, X: np.ndarray) -> np.ndarray:
        if len(X.shape) == 1:
            X = X[None, :]
        return (self._compute_sim(X) > self.tau).astype(int)

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        y_pred = self.predict(X)
        return np.mean((y_pred == y).astype(int))
