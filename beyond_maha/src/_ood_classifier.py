from typing import Optional

import numpy as np
import ot
from scipy.spatial.distance import cdist
from sklearn.base import ClassifierMixin

from .aiirw import AI_IRW
from .utils._ood_classifier_utils import (check_fitted, logSumExp,
                                          partial_wrapper,
                                          random_sampler_wrapper, softmax)


class OODDetector(ClassifierMixin):
    """
    We made the choice not to take the training distributions as inputs of the fit method as it could be
    done in sklearn because of a sense-problem. The OOD detector is classifier that detects OOD samples.
    Therefore its target is a vector of 1 if OOD and else 0.
    Therefore, calling .fit(X,y) to fit the classifier to its base distribution would be counterintuitive
    because it is not the distribution we seek to fundamentally learn.
    Moreover, this is an out-of-the-bag classifier, hence it is would not make sense to "fit" it to some
    data in the scikit-learn sense.

    To keep the formalism, we kept a fit function that actually does a prefitting.
    """

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

        self.__is_fitted__ = False

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
                """
                `compute_dist_wrt` computes the sum of the k-nearest $\ell_1$-distances of a point x to a dataset ds

                :param x: the point we're trying to find the distance to
                :param ds: the dataset
                :return: The distance between the point and the k nearest neighbors.
                """
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
                """
                > For each row in `X`, find the `k` closest rows in `ds` and return the sum of the $\ell_1$distances between
                the row in `X` and the `k` closest rows in `ds`

                :param x: np.ndarray, ds: np.ndarray
                :type x: np.ndarray
                :param ds: the dataset
                :type ds: np.ndarray
                :return: The sum of the k-nearest neighbors of each point in the dataset.
                """
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
            eig_val, eig_vec = np.linalg.eigh(
                G
            )  # returns eigvecs sorted in increasing eigvals
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
        self.__is_fitted__ = True
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        check_fitted(self)
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
        check_fitted(self)
        if len(X.shape) == 1:
            X = X[None, :]
        return self._compute_dist(X)

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        return self.predict_proba(X)
