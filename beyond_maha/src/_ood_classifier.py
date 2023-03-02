from typing import Optional

import numpy as np
import ot
from .aiirw import AI_IRW
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


def random_sampler_wrapper(f, base_distribution, sampling_ratio):
    def wrapper(*args):
        n = base_distribution.shape[0]
        idxs = np.random.choice(n, size=int(n * sampling_ratio), replace=False)
        ds = base_distribution[idxs, :]
        return f(*args, ds=ds)

    return wrapper


class OODDetector(ClassifierMixin):
    def __init__(
        self,
        tau: float = 1,
        base_distribution: Optional[np.ndarray] = None,
        base_ood_distribution: Optional[np.ndarray] = None,
        similarity: str = "mahalanobis",
        T: float = 1.0,
        k: int = 4,
        sampling_ratio: Optional[float] = None,
    ):
        super().__init__()

        assert similarity in [
            "mahalanobis",
            "IRW",
            "MSP",
            "E",
            "wass2unif",
            "wass2data",
            "wasscombo",
            "LiLO",
        ], "Similarity is not available"

        if base_distribution is None:
            assert similarity in [
                "MSP",
                "E",
                "wass2unif",
            ], "You must provide a train distribution for data-driven detector"

        if base_ood_distribution is None:
            assert (
                similarity != "LiLO"
            ), "You must provide a train ood distribution for LiLO detector"

        self.tau = tau
        self.similarity = similarity
        self.T = T
        self.k = k

        self.base_distribution = base_distribution
        self.base_ood_distribution = base_ood_distribution
        self.sampling_ratio = sampling_ratio  # allows for lesser computations

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

    def _prefit(self) -> None:
        """
        `_prefit` is a function that takes in a similarity metric and returns a function that computes the
        similarity between a given point and the base distribution
        :return: None
        """
        if self.similarity == "mahalanobis":
            if len(self.base_distribution.shape) == 3:
                self.base_distribution = np.mean(self.base_distribution, axis=-1)

            m = np.mean(self.base_distribution, axis=0).reshape(
                1, self.base_distribution.shape[1]
            )
            VI = np.linalg.inv(np.cov(self.base_distribution.T))
            self._compute_dist = lambda x: partial_wrapper(
                cdist, XB=m, metric="mahalanobis", VI=VI
            )(np.mean(x, axis=-1))
        elif self.similarity == "IRW":
            if len(self.base_distribution.shape) == 3:
                self.base_distribution = np.mean(self.base_distribution, axis=-1)

            self._compute_dist = lambda x: 1 - partial_wrapper(
                AI_IRW,
                X=self.base_distribution,
                n_dirs=1000,
                AI=True,
                robust=False,
                random_state=None,
            )(np.mean(x, axis=-1))

        elif self.similarity == "MSP":
            self._compute_dist = lambda x: np.max(softmax(x), axis=-1)

        elif self.similarity == "E":

            def energy(x):
                return self.T * logSumExp(x / self.T)

            self._compute_dist = lambda x: np.max(energy(x)) - energy(x)

        elif self.similarity == "wass2unif":
            self._compute_dist = lambda x: np.sum(
                np.abs(x - ot.unif(x.shape[-1])), axis=-1
            )

        elif self.similarity == "wass2data":

            def compute_dist_wrt(x, ds):
                return np.sum(
                    np.sort(
                        partial_wrapper(cdist, XB=ds, metric="cityblock")(x),
                        axis=-1,
                    )[:, : self.k],
                    axis=-1,
                )

            if self.sampling_ratio:
                self._compute_dist = random_sampler_wrapper(
                    compute_dist_wrt, self.base_distribution, self.sampling_ratio
                )
            else:
                self._compute_dist = lambda x: compute_dist_wrt(
                    x, self.base_distribution
                )

        elif self.similarity == "wasscombo":

            def _wtd(x: np.ndarray, ds: np.ndarray) -> np.ndarray:
                return np.sum(
                    np.sort(
                        partial_wrapper(cdist, XB=ds, metric="cityblock")(x),
                        axis=-1,
                    )[:, : self.k],
                    axis=-1,
                )

            if self.sampling_ratio:
                wtd = random_sampler_wrapper(
                    _wtd, self.base_distribution, self.sampling_ratio
                )
            else:
                wtd = lambda x: _wtd(x, self.base_distribution)

            def wtu(x: np.ndarray) -> np.ndarray:
                return np.sum(np.abs(x - ot.unif(x.shape[-1])), axis=-1)

            WTU = wtu(self.base_distribution)

            self.tau_u = np.percentile(WTU, 99)

            self._compute_dist = lambda x: (wtu(x) > self.tau_u) * wtu(x) + (
                wtu(x) <= self.tau_u
            ) * wtd(x)

        elif self.similarity == "LiLO":
            embd_ood_mean = np.mean(self.base_ood_distribution, axis=0)
            embd_train_mean = np.mean(self.base_distribution, axis=0)

            OOD_cov = np.mean(
                [
                    self.base_ood_distribution[i].T @ self.base_ood_distribution[i]
                    for i in range(self.base_ood_distribution.shape[0])
                ],
                axis=0,
            )
            train_cov = np.mean(
                [
                    self.base_distribution[i].T @ self.base_distribution[i]
                    for i in range(self.base_distribution.shape[0])
                ],
                axis=0,
            )

            G = (
                OOD_cov
                + train_cov
                - (embd_train_mean - embd_ood_mean).T
                @ (embd_train_mean - embd_ood_mean)
            )
            eig_val, eig_vec = np.linalg.eigh(G)
            alpha = eig_vec[0]

            def l2_custom(x: np.ndarray, ds: np.ndarray) -> np.ndarray:
                return np.sum(cdist(ds @ alpha, x @ alpha), axis=0)

            if self.sampling_ratio:
                self._compute_dist = random_sampler_wrapper(
                    l2_custom, self.base_distribution, self.sampling_ratio
                )
            else:
                self._compute_dist = lambda x: l2_custom(x, self.base_distribution)

        return None

    def fit(self, X: Optional[np.ndarray] = None, y: Optional[np.ndarray] = None):
        self._prefit()
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if len(X.shape) == 1:
            X = X[None, :]
        return (self._compute_dist(X) <= self.tau).astype(int)

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
        return self._compute_dist(X)

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        return self.predict_proba(X)
