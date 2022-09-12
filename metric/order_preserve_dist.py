import pot
import numpy as np


def order_preserve(x1, x2=None, metric='sqeuclidean', lambda1=50, lambda2=0.1, delta=1):
    r"""Compute distance between samples in :math:`\mathbf{x_1}` and :math:`\mathbf{x_2}` and add order preserve prior.

    Parameters
    ----------

    x1 : array-like, shape (n1, d)
        matrix with `n1` samples of dimension `d`

    x2 : array-like, shape (n2, d), optional
        matrix with `n2` samples of dimension `d` (if None then :math:`\mathbf{x_2} = \mathbf{x_1}`)

    metric : str, optional

    Returns
    -------

    M : array-like, shape (`n1`, `n2`)
        distance matrix between samples in :math:`\mathbf{x_1}` and :math:`\mathbf{x_2}`

    """

    D = pot.dist(x1, x2, metric=metric)
    N = x1.shape[0]
    M = x2.shape[0]

    dim = x1.shape[1]

    mid_para = np.sqrt((1/(N**2) + 1/(M**2)))

    row_col_matrix = np.mgrid[1:N+1, 1:M+1]
    row = row_col_matrix[0] / N   # row = (i+1)/N
    col = row_col_matrix[1] / M   # col = (j+1)/M

    d_matrix = np.abs(row - col) / mid_para
    F = d_matrix ** 2

    E = 1 / ((row - col) ** 2 + 1)

    D_hat = D - lambda1 * E - lambda2 * (F / (2 * delta ** 2) + np.log(delta * np.sqrt(2 * np.pi)))

    return D_hat


def relative_element_trans(x, metric='euclidean'):
    if metric == 'euclidean':
        x_tru =  np.concatenate((np.zeros((1,x.shape[1])), x[:-1]), axis=0)
        distance = np.linalg.norm(x - x_tru, axis=1)
        fx = np.cumsum(distance)
        sum_distance = fx[-1]
        fx = fx / sum_distance
        return fx

def t_opw1_dist(x1, x2=None, metric='sqeuclidean', lambda1=50, lambda2=0.1, delta=1):
    r"""Compute distance between samples in :math:`\mathbf{x_1}` and :math:`\mathbf{x_2}` and add topw1 prior.

    Parameters
    ---------
    x1 : array-like, shape (n1, d)
        matrix with `n1` samples of dimension `d`

    x2 : array-like, shape (n2, d), optional
        matrix with `n2` samples of dimension `d` (if None then :math:`\mathbf{x_2} = \mathbf{x_1}`)

    metric : str, optional

    Returns
    -------

    M : array-like, shape (`n1`, `n2`)
        distance matrix between samples in :math:`\mathbf{x_1}` and :math:`\mathbf{x_2}`

    """

    D = pot.dist(x1, x2, metric=metric)
    N = x1.shape[0]
    M = x2.shape[0]

    dim = x1.shape[1]

    row_col_matrix = np.mgrid[1:N+1, 1:M+1]
    row = row_col_matrix[0] / N   # row = (i+1)/N
    col = row_col_matrix[1] / M   # col = (j+1)/M

    fx1 = relative_element_trans(x1)
    fx2 = relative_element_trans(x2)

    m, n = np.meshgrid(fx2, fx1)
    d_matrix = np.maximum(m,n) / np.minimum(m,n)

    E = 1 / ((row - col) ** 2 + 1)
    F = d_matrix ** 2

    D_hat = D - lambda1 * E - lambda2 * (F / (2 * delta ** 2) + np.log(delta * np.sqrt(2 * np.pi)))
    return D_hat

def t_opw2_dist(x1, x2=None, metric='sqeuclidean', lambda1=50, lambda2=0.1, delta=1):
    r"""Compute distance between samples in :math:`\mathbf{x_1}` and :math:`\mathbf{x_2}` and add topw2 prior.

    Parameters
    ----------
    x1 : array-like, shape (n1, d)
        matrix with `n1` samples of dimension `d`

    x2 : array-like, shape (n2, d), optional
        matrix with `n2` samples of dimension `d` (if None then :math:`\mathbf{x_2} = \mathbf{x_1}`)

    metric : str, optional

    Returns
    -------

    M : array-like, shape (`n1`, `n2`)
        distance matrix between samples in :math:`\mathbf{x_1}` and :math:`\mathbf{x_2}`

    """

    D = pot.dist(x1, x2, metric=metric)
    N = x1.shape[0]
    M = x2.shape[0]

    dim = x1.shape[1]

    row_col_matrix = np.mgrid[1:N+1, 1:M+1]
    row = row_col_matrix[0] / N   # row = (i+1)/N
    col = row_col_matrix[1] / M   # col = (j+1)/M

    fx1 = relative_element_trans(x1)
    fx2 = relative_element_trans(x2)

    m, n = np.meshgrid(fx2, fx1)
    d_matrix = np.abs(m - n) / np.sqrt(1 / N**2 + 1 / M ** 2)

    E = 1 / ((row - col) ** 2 + 1)
    F = d_matrix ** 2

    D_hat = D - lambda1 * E - lambda2 * (F / (2 * delta ** 2) + np.log(delta * np.sqrt(2 * np.pi)))
    return D_hat










