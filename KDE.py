import cupy as cp
import numpy as np

from sklearn.base import BaseEstimator, DensityMixin
from sklearn.utils import check_array
from scipy.spatial.distance import pdist, cdist, squareform

def find_optimal_bandwidth(X, l_h, gpu=True, lik=True):
    l_lik = []
    for h in l_h:
        kde = KDE(h=h, gpu=gpu)
        kde.fit(X)
        p_loo = kde.p_loo()
        f_sq = kde.f_sq()
        if lik:
            lik = np.log(p_loo).mean()
            l_lik.append(lik)
        else:
            ise =  - (f_sq - 2 * p_loo.mean())
            l_lik.append(ise)

    max_arg = np.argmax(l_lik)
    return l_h[max_arg]


class KDE(BaseEstimator, DensityMixin):
    def __init__(self, h=1.0, kernel='gaussian', gpu=False, ise=True):
        self.h = h
        self.kernel = kernel
        self.gpu = gpu
        self.ise = ise
        self.dist_mat_ = None

    def _cdist(self, X, Z):
        if self.gpu:
            XZT = X.dot(Z.T)
            A = cp.tile(cp.diag(X.dot(X.T)), (Z.shape[0], 1))
            B = cp.tile(cp.diag(Z.dot(Z.T)), (X.shape[0], 1))
            dist_gpu = A.T + B - 2 * XZT
            return dist_gpu
        else:
            return cdist(X, Z, metric='sqeuclidean')

    def xp(self):
        """For numpy-cupy agnostic code """
        if self.gpu:
            return cp
        else:
            return np

    def fit(self, X, y=None, l_h=None):
        """
        if l_h is given, the bandwidth selection based on LOO-ISE
        is performed.
        """
        xp = self.xp()
        if isinstance(X, np.ndarray):
            X = check_array(X)
        self.X_ = xp.asarray(X)
        self.D_ = X.shape[1]
        self.N_ = self.X_.shape[0]
        N = self.N_

        if l_h is not None and self.kernel == 'gaussian':
            l_error = []
            for hh in l_h:
                loo_ise = self.f_sq(h=hh) - 2 * self.p_loo(h=hh).sum() / N
                l_error.append(loo_ise)
            idx = np.argmin(np.array(l_error))
            h_opt = l_h[idx]
            self.h = h_opt
        return self

    def p_loo(self, h=None, loo=True):
        """compute leave-one-out pdf estimate on training samples
        loo: Bool. Computes p estimate not in loo manner if False."""
        X = self.X_
        N = X.shape[0]
        D = self.D_
        xp = self.xp()
        assert self.kernel in ('gaussian', 'fourth2', 'fourth')
        if h is not None:
            h = h
        else:
            h = self.h

        if self.dist_mat_ is not None:
            dist_mat = self.dist_mat_
        else:
            dist_mat = self._cdist(X, X)

        if self.kernel == 'gaussian':
            Kxx = xp.exp(-dist_mat/(h**2)/2) / ((xp.sqrt(2*np.pi)*h) ** D)
        elif self.kernel == 'fourth2':
            Kxx = xp.exp(-dist_mat/(h**2)/2) / ((xp.sqrt(2*np.pi)*h) ** D)
            Kxx = Kxx * (1 + 0.5 * D - 0.5 * dist_mat / h**2)
        elif self.kernel == 'fourth':
            Kxx = xp.exp(-dist_mat/(h**2)/2) / ((xp.sqrt(2*np.pi)*h) ** D)
            N_1 = X.shape[0]
            N_2 = X.shape[0]
            X_1 = X.reshape((N_1, 1, D))
            X_2 = X.reshape((1, N_2, D))
            diffsq_dimwise = (X_1 - X_2) ** 2.
            poly_term = xp.prod(3. - diffsq_dimwise / (h ** 2), axis=2) / (2**D)
            Kxx = Kxx * poly_term

        if loo:
            for i in range(N):
               Kxx[i, i] = 0.
            p_loo = Kxx.sum(axis=1) / (N-1)
        else:
            p_loo = Kxx.mean(axis=1)
        return cp.asnumpy(p_loo)

    def f_sq(self, h=None):
        """ compute $\int f^2 (x) dx$ """
        xp = self.xp()
        X = self.X_
        N = X.shape[0]
        D = self.D_
        if h is None:
            h = self.h

        if self.kernel == 'gaussian':
            # f_sq = xp.sum(self.predict(X, h=xp.sqrt(2)*h)) / N
            f_sq = xp.sum(self.p_loo(h=xp.sqrt(2)*h, loo=False)) / N
        elif self.kernel == 'fourth':
            X_1 = X.reshape((N, 1, D))
            X_2 = X.reshape((1, N, D))
            diff_dimwise = X_1 - X_2

            # convoluted Gaussian kernel
            diffsq_dimwise = diff_dimwise ** 2.
            dist_mat = diffsq_dimwise.sum(axis=2)
            const = ((xp.sqrt(2*xp.pi)*(xp.sqrt(2)*h)) ** D)
            K = xp.exp(-dist_mat/((xp.sqrt(2)*h)**2)/2) / const

            x_bar = (X_1 + X_2) / 2.
            cross = X_1 * X_2
            h2 = h ** 2.
            s2 = h2 / 2.

            term_1 = 9
            term_2 = -3. / h2 * (diffsq_dimwise/4. + s2) * 2
            #term_3 = 1/(h2**2) * (- (diffsq_dimwise-2*cross)*(x_bar**2 + s2)+cross**2)
            #term_3_1 = 3*(s2**2) - 6*x_bar**2 * s2 + x_bar**4
            #term_3_2 = -2*(2*x_bar)*(x_bar**3 + 3*s2*x_bar)
            #term_3_3 = ((x_bar*2)**2 + 2*cross)*(x_bar**2 +s2)
            #term_3_4 = -2 * (cross*(2*x_bar))*x_bar + cross ** 2
            #term_3 = (term_3_1 + term_3_2 + term_3_3 + term_3_4) / (h2**2)
            term_3 = (3 * (s2**2) - s2*diffsq_dimwise/2 + (diffsq_dimwise**2) / 16.) / (h2**2)

            K_xx = xp.prod((term_1 + term_2 + term_3) / 4., axis=2) * K
            f_sq = K_xx.sum() / (N**2)
        else:
            raise NotImplementedError

        if self.gpu:
            f_sq = cp.asnumpy(f_sq)
        return f_sq

    def predict(self, X, h=None, clip=True, remain_gpu=False):
        xp = self.xp()
        train_X = self.X_
        target_X = xp.asarray(X)
        if h is None:
            h = self.h
        D = self.D_
        N = train_X.shape[0]

        if self.kernel == 'gaussian':
            dist_mat = self._cdist(train_X, target_X)
            # dist_mat = cdist(train_X, target_X, metric='sqeuclidean')
            const = ((xp.sqrt(2 * xp.pi) * h) ** D)
            prob = xp.exp(-dist_mat / (h ** 2) / 2).sum(axis=0) / const / N
        elif self.kernel == 'fourth':
            N_1 = train_X.shape[0]
            N_2 = target_X.shape[0]
            X_1 = train_X.reshape((N_1, 1, D))
            X_2 = target_X.reshape((1, N_2, D))
            diffsq_dimwise = (X_1 - X_2) ** 2.

            # 2nd-order Gaussian kernel
            dist_mat = diffsq_dimwise.sum(axis=2)
            const = ((xp.sqrt(2 * xp.pi) * h) ** D)
            K = xp.exp(-dist_mat / (h ** 2) / 2) / const

            # polynomial term
            poly_term = xp.prod(3. - diffsq_dimwise / (h ** 2), axis=2) / (2**D)

            prob = (K * poly_term).mean(axis=0)
            if clip:
                prob[prob <= 0.] = 0.
        else:
            raise NotImplementedError

        if self.gpu and not remain_gpu:
            prob = cp.asnumpy(prob)
        return prob