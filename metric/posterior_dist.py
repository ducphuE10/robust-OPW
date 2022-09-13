import numpy as np


def add_posterior_dist(a, b, M, lambda2=0.1):
    r"""Add posterior distribution to the distance matrix

    Parameters
    ----------
    a: array-like, shape (M, )
        source weight, sum to 1
    b: array-like, shape (N, )
        target weight, sum to 1
    M: array-like, distance matrix shape (M, N)


    Returns
    -------
    M_new: array-like, distance matrix with posterior distribution
    """

    assert a.shape[0] == M.shape[0] and b.shape[0] == M.shape[1]

    a = a / np.sum(a)
    b = b / np.sum(b)

    evidence = np.outer(a, b)
    M_new = M - lambda2 * np.log(evidence)
    return M_new


