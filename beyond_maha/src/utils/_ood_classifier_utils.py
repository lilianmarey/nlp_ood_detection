import numpy as np


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
    """
    It takes a function, a base distribution, and a sampling ratio, and returns a new function that
    takes the same arguments as the original function, but also takes a sample of the base distribution

    :param f: the function to be wrapped
    :param base_distribution: the original distribution
    :param sampling_ratio: the ratio of the dataset to be sampled
    :return: A function that takes in a function f, a base distribution, and a sampling ratio.
    """

    def wrapper(*args):
        n = base_distribution.shape[0]
        idxs = np.random.choice(n, size=int(n * sampling_ratio), replace=False)
        ds = base_distribution[idxs, :]
        return f(*args, ds=ds)

    return wrapper


def check_fitted(estimator):
    assert estimator.__is_fitted__, "Please fit the estimator before usage"
