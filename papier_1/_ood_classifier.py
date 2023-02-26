from typing import Callable, Optional

import numpy as np
import ot
from aiirw import AI_IRW
from scipy.spatial.distance import cdist
from sklearn.base import ClassifierMixin


def partial_wrapper(f, **kwargs):
    """
    It takes a function and some keyword arguments, and returns a function that takes some positional
    arguments, calls the original function with the positional arguments and the keyword arguments, and
    returns the flattened result

    :param f: the function to be wrapped
    :return: The function wrapper is being returned.
    """

    def wrapper(*args):
        return f(*args, **kwargs)

    return wrapper


def softmax(A: np.ndarray) -> np.ndarray:
    """
    It takes an array of numbers and returns an array of numbers that are the softmax of the original
    numbers

    :param A: The input to the softmax function
    :type A: np.ndarray
    :return: The softmax function is being returned.
    """
    e_A = np.exp(A)
    return np.diag(1 / np.sum(e_A, axis=-1)) @ e_A


def logSumExp(A: np.ndarray) -> np.ndarray:
    """
    > soft surrogate for the max function

    :param A: a 2D array of shape (N, K)
    :type A: np.ndarray
    :return: The log of the sum of the exponentials of the array A.
    """
    return np.log(np.sum(np.exp(A), axis=-1))


class OODDetector(ClassifierMixin):
    def __init__(
        self,
        tau: float = 1,
        base_distribution: Optional[np.ndarray] = None,
        similarity: str = "mahalanobis",
        T: float = 1.0,
        k: int = 10,
    ):
        super().__init__()

        assert similarity in [
            "mahalanobis",
            "IRW",
            "MSP",
            "E",
            "wass2unif",
            "wass2data",
        ], "Similarity is not available"

        if base_distribution is None:
            assert similarity in [
                "MSP",
                "E",
                "wass2unif",
            ], "Similarity is not available"

        self.tau = tau
        self.similarity = similarity
        self.T = T
        self.k = k

        self.base_distribution = base_distribution

    def clone(self):
        """
        It returns a new instance of the same class, with the same parameters
        :return: A new instance of the class.
        """
        return self.__class__(
            tau=self.tau,
            base_distribution=self.base_distribution,
            similarity=self.similarity,
        )

    def _prefit(self):
        """
        `_prefit` is a function that takes in a similarity metric and returns a function that computes the
        similarity between a given point and the base distribution
        :return: None
        """
        if self.similarity == "mahalanobis":
            m = np.mean(self.base_distribution, axis=0).reshape(
                1, self.base_distribution.shape[1]
            )
            VI = np.linalg.inv(np.cov(self.base_distribution.T))
            self._compute_sim = lambda x: -partial_wrapper(
                cdist, XB=m, metric="mahalanobis", VI=VI
            )(x)
        elif self.similarity == "IRW":
            self._compute_sim = partial_wrapper(
                AI_IRW,
                X=self.base_distribution,
                n_dirs=1000,
                AI=False,
                robust=False,
                random_state=None,
            )

        elif self.similarity == "MSP":
            self._compute_sim = lambda x: 1 - np.max(softmax(x), axis=-1)

        elif self.similarity == "E":
            self._compute_sim = lambda x: self.T * logSumExp(x / self.T)

        elif self.similarity == "wass2unif":
            self._compute_sim = lambda x: -np.mean(
                np.abs(x - ot.unif(x.shape[-1])), axis=-1
            )

        elif self.similarity == "wass2data":
            self._compute_sim = lambda x: -np.mean(
                np.sort(
                    partial_wrapper(
                        cdist, XB=self.base_distribution, metric="cityblock"
                    )(x),
                    axis=-1,
                )[:, : self.k],
                axis=-1,
            )

        return None

    def fit(self, X: Optional[np.ndarray] = None, y: Optional[np.ndarray] = None):
        self._prefit()
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if len(X.shape) == 1:
            X = X[None, :]
        return (self._compute_sim(X) <= self.tau).astype(int)

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        y_pred = self.predict(X)
        return np.mean((y_pred == y).astype(int))

    def get_params(self, deep=True):
        base = {
            "tau": self.tau,
            "similarity": self.similarity,
        }

        if self.similarity == "E":
            base["T"] = self.T

        if self.similarity == "wass2data":
            base["k"] = self.k

        return base

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
