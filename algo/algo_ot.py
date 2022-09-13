import ot
import numpy as np

def relative_element_trans(x, metric='euclidean'):
    if metric == 'euclidean':
        x_tru =  np.concatenate((np.zeros((1,x.shape[1])), x[:-1]), axis=0)
        distance = np.linalg.norm(x - x_tru, axis=1)
        fx = np.cumsum(distance)
        sum_distance = fx[-1]
        fx = fx / sum_distance
        return fx


class BaseOrderPreserve:
    def __init__(self,lambda1, lambda2, delta):
        self.lambda1 = lambda1
        self.lambda2 = lambda2
        self.delta = delta

    def get_d_matrix(self, x1, x2):
        pass

    def fit(self, x1, x2, a, b, metric='sqeuclidean', **kwargs):
        tolerance = .5e-2
        maxIter = 20

        if x1.dim == 1:
            x1 = x1.reshape(-1, 1)
            x2 = x2.reshape(-1, 1)

        d_matrix = self.get_d_matrix(x1, x2)
        P = np.exp(-d_matrix ** 2 / (2 * delta ** 2)) / (delta * np.sqrt(2 * np.pi))
        P = a@b.T * P

        row_col_matrix = np.mgrid[1:N + 1, 1:M + 1]
        row = row_col_matrix[0] / N  # row = (i+1)/N
        col = row_col_matrix[1] / M  # col = (j+1)/M
        S = lambda1 / ((row - col) ** 2 + 1)

        D = ot.dist(x1, x2, metric=metric, **kwargs)

        max_distance = 200 * lambda2
        D = np.clip(D, 0, max_distance)

        K = np.exp((S - D) / lambda2) * P

        compt = 0
        u = np.ones((N, 1)) / N

        while compt < maxIter:
            u = a / (K @ (b / (K.T @ u)))
            assert not np.isnan(u).any(), "nan in u"
            compt += 1

            if compt % 20 == 0 or compt == maxIter:
                v = b / (K.T @ u)
                u = a / (K @ v)

                criterion = np.linalg.norm(
                    np.sum(np.abs(v * (K.T @ u) - b), axis=0), ord=np.inf)
                if criterion < tolerance:
                    break

        U = K * D
        dis = np.sum(u * (U @ v))
        T = np.diag(u[:, 0]) @ K @ np.diag(v[:, 0])

        return dis, T



class TOPW2(BaseOrderPreserve):
    def __init__(self, lambda1, lambda2, delta):
        super().__init__(lambda1, lambda2, delta)

    def get_d_matrix(self, x1, x2):
        N = x1.shape[0]
        M = x2.shape[0]

        fx = relative_element_trans(X)
        fy = relative_element_trans(Y)
        m,n = np.meshgrid(fy, fx)
        # d_matrix = np.maximum(m,n) / np.minimum(m,n)
        d_matrix = np.abs(m - n) / np.sqrt(1 / N **2 + 1 / M ** 2)
        return d_matrix


class TOPW1(BaseOrderPreserve):
    def __init__(self, lambda1, lambda2, delta):
        super().__init__(lambda1, lambda2, delta)

    def get_d_matrix(self, x1, x2):
        N = x1.shape[0]
        M = x2.shape[0]

        fx = relative_element_trans(X)
        fy = relative_element_trans(Y)
        m,n = np.meshgrid(fy, fx)
        d_matrix = np.maximum(m,n) / np.minimum(m,n)
        return d_matrix


class OPW(BaseOrderPreserve):
    def __init__(self, lambda1, lambda2, delta):
        super().__init__(lambda1, lambda2, delta)

    def get_d_matrix(self, x1, x2):

        N = x1.shape[0]
        M = x2.shape[0]

        mid_para = np.sqrt((1/(N**2) + 1/(M**2)))

        row_col_matrix = np.mgrid[1:N+1, 1:M+1]
        row = row_col_matrix[0] / N   # row = (i+1)/N
        col = row_col_matrix[1] / M   # col = (j+1)/M

        d_matrix = np.abs(row - col) / mid_para
        return d_matrix


