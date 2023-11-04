import copy
import cupy as cp
import numpy as np

from sklearn.base import BaseEstimator, DensityMixin
from sklearn.utils import check_array

import scipy.linalg as spl
from scipy.spatial.distance import pdist, cdist, squareform

class GaussianModel(BaseEstimator, DensityMixin):
    def __init__(self, reg=1e-6, gpu=False):
        self.reg = reg
        self.gpu = gpu

    def xp(self):
        if self.gpu:
            return cp
        else:
            return np

    def fit(self, X, y=None, internal=False):
        if not internal:
            X = check_array(X)
        xp = self.xp()
        self.X_ = xp.asarray(X)
        self.D_ = X.shape[1]
        self.N_ = X.shape[0]
        reg = self.reg

        self.mu_ = self.X_.mean(axis=0)
        z = xp.asarray(self.X_ - self.mu_)
        self.sig_ = z.T.dot(z) / self.N_ + reg * xp.eye(self.D_)
        return self

    def predict(self, x_new, internal=False):
        if not internal:
            x_new = check_array(x_new)
        D = self.D_
        sig = self.sig_
        xp = self.xp()
        x_new = xp.asarray(x_new)

        logp = self.log_p(x_new)
        return xp.exp(logp)

    def log_p(self, x_new):
        # x_new = check_array(x_new)
        xp = self.xp()
        D = self.D_
        sig = self.sig_
        mu_ = self.mu_
        x_new = xp.asarray(x_new)
        z_new = x_new - mu_

        normalize = - xp.log(2 * np.pi) * (0.5 * D) - 0.5 * xp.log(xp.linalg.det(sig))
        if self.gpu:
            # print(z_new.dot(xp.linalg.inv(sig).dot(z_new.T)).shape)
            expo = -0.5 * xp.diag(z_new.dot(xp.linalg.inv(sig).dot(z_new.T)))
        else:
            expo = -0.5 * xp.diag(z_new.dot(spl.cho_solve(spl.cho_factor(sig), z_new.T)))
        return normalize + expo


    def grad_over_p(self, x_new):
        """gradient of p over p"""
        xp = self.xp()
        x_new = xp.asarray(x_new)
        D = self.D_
        sig = self.sig_
        mu = self.mu_
        z_new = x_new - mu

        if self.gpu:
            dexp_dx = - z_new.dot(cp.linalg.inv(sig))
        else:
            dexp_dx = - spl.cho_solve(spl.cho_factor(sig), z_new.T).T
        return dexp_dx

    def lap_over_p(self, x_new):
        """laplacian of p. $\nabla^2 p = tr(\nabla\nabla p)$"""
        xp = self.xp()
        x_new = xp.asarray(x_new)
        D = self.D_
        sig = self.sig_
        mu = self.mu_
        z_new = x_new - mu

        invsig = xp.linalg.inv(sig)
        invsig2 = invsig.dot(invsig)
        trinvsig = xp.trace(invsig)
        p = self.predict(x_new, internal=True)

        tr_part = xp.zeros_like(p)
        for i, z_i in enumerate(z_new):
            tr_part[i] = xp.trace(xp.outer(z_i, z_i).dot(invsig2)) - trinvsig
        return tr_part

    def grad_lap_over_p(self, x_new):
        xp = self.xp()
        sig = self.sig_
        mu = self.mu_
        x_new = xp.asarray(x_new)
        z_new = x_new - mu
        invsig2 = xp.linalg.inv(sig.dot(sig))
        return 2 * z_new.dot(invsig2)

    def grad_grad_over_p(self, x_new):
        xp = self.xp()
        sig = self.sig_
        invsig = xp.linalg.inv(sig)
        return xp.ones(x_new.shape[0])[:, None, None] * (- invsig)

    def tr_grad_grad_over_p(self, x_new):
        xp = self.xp()
        sig = self.sig_
        invsig = xp.linalg.inv(sig)
        return xp.ones(x_new.shape[0])[:, None] * (- xp.trace(invsig))


class KDE(BaseEstimator, DensityMixin):
    def __init__(self, h=1.0, kernel='gaussian'):
        self.h = h
        self.kernel = kernel

    def fit(self, X, y=None, l_h=None):
        """
        if l_h is given, the bandwidth selection based on LOO-ISE
        is performed.
        """
        X = check_array(X)
        self.X_ = X
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

    def p_loo(self, h=None):
        """compute leave-one-out pdf estimate on training samples"""
        X = self.X_
        N = X.shape[0]
        D = self.D_
        assert self.kernel == 'gaussian'
        if h is not None:
            h = h
        else:
            h = self.h

        dist_mat = squareform(pdist(X, metric='sqeuclidean'))
        Kxx = np.exp(-dist_mat / (h ** 2) / 2) / ((np.sqrt(2 * np.pi) * h) ** D)
        for i in range(N):
            Kxx[i, i] = 0.
        p_loo = Kxx.sum(axis=1) / (N - 1)
        return p_loo

    def f_sq(self, h=None):
        """ compute $\int f^2 (x) dx$ """
        X = self.X_
        N = X.shape[0]
        D = self.D_
        if h is None:
            h = self.h

        if self.kernel == 'gaussian':
            f_sq = np.sum(self.predict(X, h=np.sqrt(2) * h)) / N
        elif self.kernel == 'fourth':
            X_1 = X.reshape((N, 1, D))
            X_2 = X.reshape((1, N, D))
            diff_dimwise = X_1 - X_2

            # convoluted Gaussian kernel
            diffsq_dimwise = diff_dimwise ** 2.
            dist_mat = diffsq_dimwise.sum(axis=2)
            const = ((np.sqrt(2 * np.pi) * (np.sqrt(2) * h)) ** D)
            K = np.exp(-dist_mat / ((np.sqrt(2) * h) ** 2) / 2) / const

            x_bar = (X_1 + X_2) / 2.
            cross = X_1 * X_2
            h2 = h ** 2.
            s2 = h2 / 2.

            term_1 = 9
            term_2 = -3. / h2 * (diffsq_dimwise / 4. + s2) * 2
            # term_3 = 1/(h2**2) * (- (diffsq_dimwise-2*cross)*(x_bar**2 + s2)+cross**2)
            # term_3_1 = 3*(s2**2) - 6*x_bar**2 * s2 + x_bar**4
            # term_3_2 = -2*(2*x_bar)*(x_bar**3 + 3*s2*x_bar)
            # term_3_3 = ((x_bar*2)**2 + 2*cross)*(x_bar**2 +s2)
            # term_3_4 = -2 * (cross*(2*x_bar))*x_bar + cross ** 2
            # term_3 = (term_3_1 + term_3_2 + term_3_3 + term_3_4) / (h2**2)
            term_3 = (3 * (s2**2) - s2 * diffsq_dimwise / 2 + (diffsq_dimwise**2) / 16.) / (h2**2)

            K_xx = np.prod((term_1 + term_2 + term_3) / 4., axis=2) * K
            f_sq = K_xx.sum() / (N**2)
        else:
            raise NotImplementedError
        return f_sq

    def predict(self, X, h=None, clip=True):
        train_X = self.X_
        target_X = check_array(X)
        if h is None:
            h = self.h
        D = self.D_
        N = train_X.shape[0]

        if self.kernel == 'gaussian':
            dist_mat = cdist(train_X, target_X, metric='sqeuclidean')
            const = ((np.sqrt(2 * np.pi) * h) ** D)
            prob = np.exp(-dist_mat / (h ** 2) / 2).sum(axis=0) / const / N
        elif self.kernel == 'fourth':
            N_1 = train_X.shape[0]
            N_2 = target_X.shape[0]
            X_1 = train_X.reshape((N_1, 1, D))
            X_2 = target_X.reshape((1, N_2, D))
            diffsq_dimwise = (X_1 - X_2) ** 2.

            # 2nd-order Gaussian kernel
            dist_mat = diffsq_dimwise.sum(axis=2)
            const = ((np.sqrt(2 * np.pi) * h) ** D)
            K = np.exp(-dist_mat / (h ** 2) / 2) / const

            # polynomial term
            poly_term = np.prod(3. - diffsq_dimwise / (h**2), axis=2) / (2**D)

            prob = (K * poly_term).mean(axis=0)
            if clip:
                prob[prob <= 0.] = 0.
        else:
            raise NotImplementedError

        return prob

    def get_K_xx(self, pdist_mat, h=None):
        """
        compute a Gaussian kernel matrix with (X1.shape[0], X2.shape[0])
        """
        if h is None:
            h = self.h
        D = self.D_
        const = ((np.sqrt(2 * np.pi) * h) ** D)
        K_xx = np.exp(-pdist_mat / (h ** 2) / 2) / const
        return K_xx


class KernelRatio(BaseEstimator):
    eps = 0

    def _pdist(self, X):
        if self.gpu:
            XXT = X.dot(X.T)  # NxN
            A = cp.tile(cp.diag(XXT), (X.shape[0], 1))
            dist_gpu = A + A.T - 2 * XXT
            return dist_gpu
        else:
            return squareform(pdist(X, metric='sqeuclidean'))

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

    def get_K_xx(self, x, z=None):
        """
        compute a Gaussian kernel matrix with (X1.shape[0], X2.shape[0])
        """
        xp = self.xp()
        D = self.D_
        h = self.h
        if z is None:
            pdist_mat = self._pdist(x)
        else:
            pdist_mat = self._cdist(x, z)
        # const = ((xp.sqrt(2 * np.pi) * h) ** D)
        K_xx = xp.exp(- pdist_mat / (h ** 2) / 2) # / const
        return K_xx

    def predict(self, X_new):
        xp = self.xp()
        X_new = xp.asarray(X_new)
        a1 = self.a1_
        a2 = self.a2_
        N1 = self.N1_
        N2 = self.N2_
        K1 = self.get_K_xx(X_new, self.X1_)
        K2 = self.get_K_xx(X_new, self.X2_)
        f1 = K1.dot(a1) / N1
        f2 = K2.dot(a2) / N2

        for i in range(K1.shape[0]):
            K1[i, i] = 0
        f1 = K1.dot(a1) / (N1 - 1)
        f = xp.log((f1 + self.eps * a1) / (f2 + self.eps * a1))
        return f

    def kl(self):
        log_ratio = self.predict(self.X1_)
        return np.nanmean(log_ratio)

class KernelRatioNaive(KernelRatio):
    """kernel density estimate plug-in estimator"""
    def __init__(self, h=0.6, gpu=False):
        self.h = h
        self.gpu = gpu 

    def fit(self, X1, X2):
        xp = self.xp()
        self.N1_, self.D_ = X1.shape
        self.N2_, _ = X2.shape
        # self._separate_class(X, y)
        self.a1_ = xp.ones(self.N1_)
        self.a2_ = xp.ones(self.N2_)
        self.X1_ = xp.asarray(X1)
        self.X2_ = xp.asarray(X2)

class KernelRatioAlpha(KernelRatio):
    """kernel density estimate plug-in estimator"""
    def __init__(self, h=0.6, gpu=True):
        self.h = h
        self.gpu = gpu 

    def fit(self, X1, X2, a1, a2):
        xp = self.xp()
        self.N1_, self.D_ = X1.shape
        self.N2_, _ = X2.shape
        # self._separate_class(X, y)
        self.a1_ = a1
        self.a2_ = a2
        self.X1_ = xp.asarray(X1)
        self.X2_ = xp.asarray(X2)

class KernelRatioGaussian(KernelRatio):
    def __init__(self, h=0.6, s=0.5, gp_h=0.6, gp_l=0.1, reg=0.1, grid_sample=None, gpu=False, einsum_batch=200,
                 kmeans=False, stabilize=False, solver='gp', para_h=0.6, para_l=0.1, online=False,
                 trunc=None):
        self.h = h
        self.s = s
        self.gp_h = gp_h
        self.gp_l = gp_l
        self.reg = reg  # gaussian covariance regularization parameter
        self.grid_sample = grid_sample  # number of data point used in PDE solving
        self.gpu = gpu 
        self.einsum_batch = einsum_batch
        self.kmeans = kmeans
        self.stabilize = stabilize
        self.solver = solver
        self.para_h = para_h
        self.para_l = para_l
        self.online = online  # If true, kernel matrix is computed online. slower but saves memory.
        self.trunc = trunc 

    def fit(self, X1, X2, true_model=None):
        xp = self.xp()
        self.X1_ = xp.asarray(X1)
        self.X2_ = xp.asarray(X2)
        if self.gpu and true_model is not None:
            l_true_model = []
            for tm in true_model:
                tm_ = copy.deepcopy(tm)
                tm_.switch_gpu()
                l_true_model.append(tm_)
            self.true_model = l_true_model
        else:
            self.true_model = true_model

        if self.kmeans:
            self.X1_cpu = X1
            self.X2_cpu = X2
        self.N1_, self.D_ = X1.shape
        self.N2_, _ = X2.shape
        if self.solver == 'gp':
            self._fit_weights()
        elif self.solver == 'para':
            self._fit_weights_para()
        elif self.solver == 'analytic':
            self._fit_weights_analytic()
        elif self.solver == 'analytic_v2':
            self._fit_weights_analytic_v2()
        elif self.solver == 'para_cls':
            self._fit_weights_para_cls()
        else:
            raise ValueError

    def _fit_gaussians(self):
        if self.true_model is not None:
            self.gaussians_ = self.true_model
        else:
            g1 = GaussianModel(reg=self.reg, gpu=self.gpu)
            g2 = GaussianModel(reg=self.reg, gpu=self.gpu)
            g1.fit(self.X1_, internal=True)
            g2.fit(self.X2_, internal=True)
            self.gaussians_ = [g1, g2]

    def _get_grid(self):
        """return grid points on which the partial differential equation will be solved"""
        xp = self.xp()
        if self.grid_sample is None:
            # use all points
            X = xp.vstack([self.X1_, self.X2_])
        elif self.kmeans:
            from sklearn.cluster import KMeans
            if self.grid_sample >= self.N1_:
                grid1 = self.X1_cpu
            else:
                km = KMeans(n_clusters=self.grid_sample)
                km.fit(self.X1_cpu)
                grid1 = km.cluster_centers_

            if self.grid_sample >= self.N2_:
                grid2 = self.X2_cpu
            else:
                km = KMeans(n_clusters=self.grid_sample)
                km.fit(self.X2_cpu)
                grid2 = km.cluster_centers_

            X = np.vstack([grid1, grid2])
            X = xp.asarray(X)
        else:
            idx1 = xp.arange(self.N1_)
            idx2 = xp.arange(self.N2_)
            xp.random.shuffle(idx1)
            xp.random.shuffle(idx2)
            idx1 = idx1[:int(self.grid_sample)]
            idx2 = idx2[:int(self.grid_sample)]
            X = xp.vstack([self.X1_[idx1], self.X2_[idx2]])
        self.grid_ = X
        X = xp.asarray(X)
        return X

    def _get_deriv(self, X=None):
        """compute density derivatives"""
        xp = self.xp()
        self._fit_gaussians()
        g1, g2 = self.gaussians_

        if X is None:
            X = self.grid_
        du = g1.grad_over_p(X) - g2.grad_over_p(X)  # (N, D)
        v = g1.lap_over_p(X) - g2.lap_over_p(X)
        if self.solver in {'para', 'para_cls'} :
            return du, v
        Hu = g1.grad_grad_over_p(X) - g2.grad_grad_over_p(X)  # (N, D, D)
        # du_Hu = xp.einsum('ij,ijk->ik', du, Hu)  # todo
        du_Hu = (du.reshape(du.shape + (1,)) * Hu).sum(axis=1)
        Lu = g1.tr_grad_grad_over_p(X) - g2.tr_grad_grad_over_p(X)
        dv = g1.grad_lap_over_p(X) - g2.grad_lap_over_p(X)
        return du, v, Hu, du_Hu, Lu, dv

    def _fit_weights(self):
        import time
        time_1 = time.time()
        xp = self.xp()
        D = self.D_

        # sample points
        X = self._get_grid()

        # build operators
        time_2 = time.time()
        self.gp_ = GPDifferentialOperator(X, h=self.gp_h, l=self.gp_l, gpu=self.gpu)
        G = self.gp_.gradient()
        H, L = self.gp_.hessian(vec=True, lap=True)

        # compute coefficients
        time_3 = time.time()
        du, v, Hu, du_Hu, Lu, dv = self._get_deriv()

        # solve equation
        time_4 = time.time()
        W = self._get_cross_term_double_vec(D)
        U = xp.einsum('ij,ik->ijk', du, du)
        triu_idx = np.triu_indices(D)
        dudu_vec = U[:, triu_idx[0], triu_idx[1]]

        # einsum batches
        if isinstance(self.einsum_batch, int):
            n_grid = X.shape[0]
            n_batch = int(np.ceil(n_grid / self.einsum_batch))
            l_A = []
            l_b = []
            # time_a = time.time()
            for i_batch in range(n_batch):
                start = i_batch * self.einsum_batch
                if i_batch == (n_batch - 1):
                    end = n_grid
                else:
                    end = (i_batch + 1) * self.einsum_batch
                dudu_vec_ = dudu_vec[start:end]
                H_ = H[start:end]
                du_Hu_ = du_Hu[start:end]
                G_ = G[start:end]
                du_ = du[start:end]
                dv_ = dv[start:end]
                v_ = v[start:end]
                Lu_ = Lu[start:end]
                L_ = L[start:end]
                term_1_ = xp.einsum('ij,j->ij', dudu_vec_, W)
                term_1 = xp.einsum('ij,ijk->ik', term_1_, H_)
                term_2 = xp.einsum('ij,ijk->ik', du_Hu_, G_)
                term_3 = Lu_ * xp.einsum('ij,ijk->ik', du_, G_)
                term_4 = xp.einsum('ij,ij->i', dv_, du_)
                term_5 = v_ * Lu_.flatten()

                A_ = 2 * (term_1 + term_2 + term_3) + self.s * L_
                b_ = - (term_4 + term_5)
                l_A.append(A_)
                l_b.append(b_)
            # time_b = time.time()
            # print(time_b - time_a)
            A = xp.concatenate(l_A)
            b = xp.concatenate(l_b)
            # time_c = time.time()
            # print(time_c - time_b)

        elif self.einsum_batch == 'for':
            """slower than einsum batch. einsum batch is slower when the batch size is small."""
            l_A = []
            l_b = []
            n_grid = X.shape[0]
            W = xp.asarray(W)
            # time_a = time.time()
            for i in range(n_grid):
                dudu_vec_ = dudu_vec[i]
                H_ = H[i]
                du_Hu_ = du_Hu[i]
                G_ = G[i]
                du_ = du[i]
                dv_ = dv[i]
                v_ = v[i]
                Lu_ = Lu[i]
                L_ = L[i]

                term_1 = (dudu_vec_ * W).dot(H_)
                term_2 = du_Hu_.dot(G_)
                term_3 = Lu_ * du_.dot(G_)
                term_4 = dv_.dot(du_)
                term_5 = v_ * Lu_
                # print(term_1.shape, term_2.shape, term_3.shape, term_4.shape, term_5.shape)
                A_ = 2 * (term_1 + term_2 + term_3) + self.s * L_
                b_ = - (term_4 + term_5)

                l_A.append(A_)
                l_b.append(b_)

            # time_b = time.time()
            # print(time_b - time_a)
            A = xp.stack(l_A)
            b = xp.concatenate(l_b)
            # time_c = time.time()
            # print(time_c - time_b)

        else:
            term_1 = xp.einsum('ij,j,ijk->ik', dudu_vec, W, H)
            term_2 = xp.einsum('ij,ijk->ik', du_Hu, G)
            term_3 = Lu * xp.einsum('ij,ijk->ik', du, G)
            term_4 = xp.einsum('ij,ij->i', dv, du)
            term_5 = v * Lu.flatten()

            A = 2 * (term_1 + term_2 + term_3) + self.s * L
            b = - (term_4 + term_5)
        # print(v.shape, Lu.shape)
        assert A.shape == (X.shape[0], X.shape[0])
        assert b.shape == (X.shape[0],)

        # sol = sp.linalg.cho_solve(sp.linalg.cho_factor(A), b)
        sol = xp.linalg.inv(A).dot(b)

        self.G = G
        self.du = du
        self.v = v

        if self.stabilize:
            stable_max = -0.1
            shift = stable_max - sol.max()
            # print(sol.max(), shift)
            sol += shift
            # print(sol.max())

        # infer whole weights
        time_4 = time.time()
        if self.grid_sample is None:
            self.a1_ = xp.exp(sol[:self.N1_])
            self.a2_ = xp.exp(sol[self.N1_:])
            self.sol_ = sol
        else:
            sol1 = self.gp_.predict(sol, self.X1_)
            sol2 = self.gp_.predict(sol, self.X2_)
            self.a1_ = xp.exp(sol1)
            self.a2_ = xp.exp(sol2)
            self.sol_ = sol

    def get_diff_dist_K(self, basis, data, h=None):
        D = self.D_
        xp = self.xp()
        Xb = data
        if h is None:
            h = self.para_h
        Xi = Xb.reshape((1, data.shape[0], D))
        basis_ = basis.reshape((basis.shape[0], 1, D))
        diff = (Xi - basis_)  # (MxBxD)
        dist = (diff ** 2).sum(axis=2)  # (MxB,)
        if self.trunc is None:
            K = xp.exp(- dist / h ** 2 / 2)  # (Mx B)
        else:
            K = xp.exp(- dist / h ** 2 / 2)  # (Mx B)
            K[K<=np.exp(-0.5*self.trunc**2)] = 0
        return diff, dist, K

    def _fit_weights_para(self):
        # import time
        # mempool = cp.get_default_memory_pool()
        # print(mempool.used_bytes() / 1024 / 1024, mempool.total_bytes() / 1024 / 1024)
        # time_1 = time.time()
        xp = self.xp()
        D = self.D_
        X = xp.vstack([self.X1_, self.X2_])
        # X = xp.vstack([self.X1_])

        # sample points
        basis = self._get_grid()
        self.basis = basis

        # compute coefficients
        # time_3 = time.time()
        # print(mempool.used_bytes() / 1024 / 1024, mempool.total_bytes() / 1024 / 1024)
        du, v = self._get_deriv(X=X)
        # print(' deriv {:.4f}sec'.format(time_3 - time_1))
        # print(mempool.used_bytes() / 1024 / 1024, mempool.total_bytes() / 1024 / 1024)

        # compute kernel matrix
        # dist = self._cdist(basis, X)
        # diff = X.reshape((1, X.shape[0], X.shape[1])) - basis.reshape((basis.shape[0], 1, basis.shape[1]))
        # h = 0.8
        # K = np.exp(- dist / h**2 / 2)
        # dK = diff * K.reshape(K.shape + (1,))
        if not self.online:
            diff_, dist_, K_ = self.get_diff_dist_K(basis, X)

        # print('dist compt {:.4f}sec'.format(time.time() - time_1))
        # print(mempool.used_bytes() / 1024 / 1024)
        # einsum batches
        if isinstance(self.einsum_batch, int):
            batch_size = self.einsum_batch
            n_batch = int(np.ceil(X.shape[0] / batch_size))
            A = xp.zeros((basis.shape[0], basis.shape[0]))  # M x M
            b = xp.zeros((basis.shape[0],))
            # C = np.zeros((basis.shape[0], basis.shape[0]))  # M x M
            l_K = []

            basis_ = basis.reshape((basis.shape[0], 1, basis.shape[1]))
            for i_b in range(n_batch):
                b_s = i_b * batch_size
                b_e = min((i_b + 1) * batch_size, X.shape[0])
                B = b_e - b_s
                Xb = X[b_s:b_e]
                dub = du[b_s:b_e]  # B x D
                vb = v[b_s:b_e]

                # kernel computation
                # Xi = Xb.reshape((1, B, D))
                # diff = (Xi - basis_)  # (MxBxD)
                # dist = (diff ** 2).sum(axis=2)  # (MxB,)
                # Ki = xp.exp(- dist / self.para_h ** 2 / 2)  # (Mx B)
                if self.online:
                    diff, dist, Ki = self.get_diff_dist_K(basis, Xb)
                else:
                    diff = diff_[:, b_s:b_e, :]
                    dist = dist_[:, b_s:b_e]
                    Ki = K_[:, b_s:b_e]

                # dk
                dk = - Ki.reshape(Ki.shape + (1,)) * diff / self.para_h ** 2  # M x B x  D
                dudk = (dub.reshape((1, B, D)) * dk).sum(axis=2)  # MxB
                A += dudk.dot(dudk.T)  # M x M
                b += (vb * dudk).sum(axis=1)  # M
                # C += dk.T.dot(dk)
                l_K.append(Ki)

            A /= X.shape[0]
            b /= X.shape[0]
            # C /= X.shape[0]
            A *= 2
            K = xp.concatenate(l_K, axis=1)

        elif self.einsum_batch == 'for':
            """slower than einsum batch. einsum batch is slower when the batch size is small."""
            A = xp.zeros((basis.shape[0], basis.shape[0]))  # M x M
            b = xp.zeros((basis.shape[0],))
            # C = np.zeros((basis.shape[0], basis.shape[0]))  # M x M
            l_K = []

            basis_ = basis.reshape((basis.shape[0], 1, basis.shape[1]))
            for i in range(X.shape[0]):
                # kernel computation
                Xi = X[i].reshape((1, 1, X.shape[1]))
                diff = (Xi - basis_).sum(axis=1)  # (MxD)
                dist = (diff ** 2).sum(axis=1)  # (M,)
                Ki = xp.exp(- dist / self.para_h ** 2 / 2)  # (M, )

                # dk
                dk = - Ki * diff.T / self.para_h ** 2  # D x  M
                dudk = du[i].dot(dk)  # M
                A += xp.outer(dudk, dudk)
                b += v[i] * dudk
                # C += dk.T.dot(dk)
                l_K.append(Ki)

            A /= X.shape[0]
            b /= X.shape[0]
            # C /= X.shape[0]
            A *= 2
            K = xp.stack(l_K, axis=1)
        else:
            raise ValueError
        assert A.shape == (basis.shape[0], basis.shape[0])
        assert b.shape == (basis.shape[0],)
        # time_4 = time.time()
        # print('{:.4f}sec'.format(time_4 - time_1))

        sol_w = - xp.linalg.inv(A + self.para_l * xp.eye(basis.shape[0])).dot(b)
        self.sol_w = sol_w
        # K2 = xp.exp(- self._cdist(basis, self.X2_) / self.para_h ** 2 / 2)
        # K = xp.hstack([K, K2])
        sol = sol_w.dot(K)
        self.sol_ = sol
        # time_5 = time.time()
        # print('{:.4f}sec'.format(time_5 - time_1))

        if self.stabilize:
            stable_max = -0.1
            shift = stable_max - sol.max()
            sol += shift

        # infer whole weights
        self.a1_ = xp.exp(sol[:self.N1_])
        self.a2_ = xp.exp(sol[self.N1_:])
        self.sol_ = sol
        # time_6 = time.time()
        # print('{:.4f}sec'.format(time_6 - time_1))

    def _fit_weights_para_cls(self):
        xp = self.xp()
        D = self.D_
        X = xp.vstack([self.X1_, self.X2_])

        # sample points
        basis = self._get_grid()
        self.basis = basis

        # compute coefficients
        du, v = self._get_deriv(X=X)
        g1, g2 = self.gaussians_

        # compute kernel matrix
        if not self.online:
            diff_, dist_, K_ = self.get_diff_dist_K(basis, X)

        # einsum batches
        if isinstance(self.einsum_batch, int):
            batch_size = self.einsum_batch
            n_batch = int(np.ceil(X.shape[0] / batch_size))
            A = xp.zeros((basis.shape[0], basis.shape[0]))  # M x M
            b = xp.zeros((basis.shape[0],))
            l_K = []

            basis_ = basis.reshape((basis.shape[0], 1, basis.shape[1]))
            for i_b in range(n_batch):
                b_s = i_b * batch_size
                b_e = min((i_b + 1) * batch_size, X.shape[0])
                B = b_e - b_s
                Xb = X[b_s:b_e]
                dub = du[b_s:b_e]  # B x D
                vb = v[b_s:b_e]

                # kernel computation
                if self.online:
                    diff, dist, Ki = self.get_diff_dist_K(basis, Xb)
                else:
                    diff = diff_[:, b_s:b_e, :]
                    dist = dist_[:, b_s:b_e]
                    Ki = K_[:, b_s:b_e]

                # prob
                p1 = g1.predict(Xb, internal=True)
                p2 = g2.predict(Xb, internal=True)
                coef = (p1 * p2 / (p1 + p2) ** 2)[None,:]  # 1xB
 

                # dk
                dk = - Ki.reshape(Ki.shape + (1,)) * diff / self.para_h ** 2  # M x B x  D
                dudk = (dub.reshape((1, B, D)) * dk).sum(axis=2)  # MxB
                A += (dudk * coef).dot(dudk.T)  # M x M
                b += (vb * dudk * coef).sum(axis=1)  # M
                l_K.append(Ki)

            A /= X.shape[0]
            b /= X.shape[0]
            A *= 2
            K = xp.concatenate(l_K, axis=1)
        else:
            raise ValueError
        assert A.shape == (basis.shape[0], basis.shape[0])
        assert b.shape == (basis.shape[0],)

        sol_w = - xp.linalg.inv(A + self.para_l * xp.eye(basis.shape[0])).dot(b)
        self.sol_w = sol_w
        sol = sol_w.dot(K)
        self.sol_ = sol

        if self.stabilize:
            stable_max = -0.1
            shift = stable_max - sol.max()
            sol += shift

        # infer whole weights
        self.a1_ = xp.exp(sol[:self.N1_])
        self.a2_ = xp.exp(sol[self.N1_:])
        self.sol_ = sol


    def compute_alpha(self, new_X):
        xp = self.xp()
        if self.solver == 'para':
            dist = cdist(cp.asnumpy(self.basis), cp.asnumpy(new_X), metric='sqeuclidean')
            K = np.exp(- dist / self.para_h ** 2 / 2)
            beta = cp.asnumpy(self.sol_w).dot(K)
            return cp.asnumpy(np.exp(beta))
        else:
            return None

    def _fit_weights_analytic(self):
        xp = self.xp()
        # fit gaussians
        self._fit_gaussians()
        # compute pooled covariance and mean
        g1, g2 = self.gaussians_
        self.pooled_S = (g1.sig_ * len(self.X1_) + g2.sig_ * len(self.X2_)) / (len(self.X1_) + len(self.X2_))
        self.pooled_mu = (g1.mu_ + g2.mu_) / 2

        # compute analytic beta
        S_inv = xp.linalg.inv(self.pooled_S)
        xx1 = self.X1_ - self.pooled_mu
        xx2 = self.X2_ - self.pooled_mu
        beta1 = (xx1 * S_inv.dot(xx1.T).T).sum(axis=1) * 0.5
        beta2 = (xx2 * S_inv.dot(xx2.T).T).sum(axis=1) * 0.5

        # compute alpha
        self.a1_ = xp.exp(beta1)
        self.a2_ = xp.exp(beta2)

    def _fit_weights_analytic_v2(self):
        """heteoscedastic gaussian assumption"""
        xp = self.xp()
        # fit gaussians
        self._fit_gaussians()
        # compute pooled covariance and mean
        g1, g2 = self.gaussians_

        # compute analytic beta
        S1inv = xp.linalg.inv(g1.sig_)
        S2inv = xp.linalg.inv(g2.sig_)
        pooled_S = (g1.sig_ + g2.sig_) / 2
        pooled_Sinv = xp.linalg.inv(pooled_S)
        mu1S1inv = g1.mu_.dot(S1inv)
        mu2S2inv = g2.mu_.dot(S2inv)
        b = (mu1S1inv + mu2S2inv) * 0.5

        xx1 = self.X1_
        xx2 = self.X2_
        beta1 = (xx1 * pooled_Sinv.dot(xx1.T).T).sum(axis=1) * 0.5 + (b * xx1).sum(axis=1)
        beta2 = (xx2 * pooled_Sinv.dot(xx2.T).T).sum(axis=1) * 0.5 + (b * xx2).sum(axis=1)

        # compute alpha
        self.a1_ = xp.exp(beta1)
        self.a2_ = xp.exp(beta2)

    def objective(self):
        w = cp.asnumpy(self.sol_w)
        A = self.A
        b = self.b
        obj = w.dot(A.dot(w)) + 2 * w.dot(b)
        reg = w.dot(w) * self.para_l
        return cp.asnumpy(obj), reg

    def run_kl_batch(self, X1, X2s, batch=3):
        """compute KL divergence for a number of distributions simultaneously """
        xp = self.xp()
        X1_ = xp.asarray(X1)
        # X2s_ = xp.asarray(X2s)  do not transfer to GPU
        self.N1_, self.D_ = X1.shape
        self.M2_, self.N2_, _ = X2s.shape
        assert self.solver == 'para'
        n_batch = int(np.ceil(self.M2_ / batch))
        D = self.D_
        n_basis = self.N1_ + self.N2_
        l_kl = []

        # fit gaussians
        g1 = GaussianModel(reg=self.reg, gpu=self.gpu)
        g1.fit(X1_, internal=True)
        g1_grad_over_p = g1.grad_over_p(X1_)
        g1_lap_over_p = g1.lap_over_p(X1_)

        diff11, dist11, K11 = self.get_diff_dist_K(X1_, X1_, h=1)
        K11_p = K11 ** (1 / self.para_h**2)
        dk1 = - K11_p.reshape(K11_p.shape + (1,)) * diff11 / self.para_h ** 2  # M x B x  D
        from tqdm import tqdm
        for b in tqdm(range(n_batch)):
            b_s = b * batch
            b_e = (b + 1) * batch if b != (n_batch - 1) else len(X2s)
            X2s_ = xp.asarray(X2s[b_s:b_e])
            diff21, dist21, K21 = self.get_diff_dist_K(X2s_.reshape(((b_e - b_s) * self.N2_, D)), X1_, h=1)
            # diff : X1 - X2

            l_g2 = []
            for i_x2, x2_ in enumerate(X2s_):
                x2_s = i_x2 * self.N2_  # start index
                x2_e = (i_x2 + 1) * self.N2_  # end index
                diff22, dist22, K22 = self.get_diff_dist_K(x2_, x2_, h=1)
                diff21_ = diff21[x2_s:x2_e]  # X2 - X1
                dist21_ = dist21[x2_s:x2_e]
                K21_ = K21[x2_s:x2_e]

                g2 = GaussianModel(reg=self.reg, gpu=self.gpu)
                g2.fit(x2_, internal=True)
                l_g2.append(g2)

                # compute derivatives
                du = g1_grad_over_p - g2.grad_over_p(X1_)  # (N, D)
                v = g1_lap_over_p - g2.lap_over_p(X1_)

                # diff = xp.concatenate([diff11, diff21_], axis=0)  # (basis:N1+N2) x (N1) x (D)
                # Ki = xp.vstack([K11, K21_]) ** (1 / self.para_h ** 2)  # 
                Ki = K21_ ** (1 / self.para_h ** 2)  # 
                dub = du
                vb = v

                dk = - Ki.reshape(Ki.shape + (1,)) * diff21_ / self.para_h ** 2  # M x B x  D
                dk = xp.vstack([dk1, dk])
                dudk = (dub.reshape((1, self.N1_, D)) * dk).sum(axis=2)  # MxB
                A = 2 * dudk.dot(dudk.T) / self.N1_  # M x M
                b = (vb * dudk).sum(axis=1) / self.N1_  # M
                sol_w = - xp.linalg.inv(A + self.para_l * xp.eye(n_basis)).dot(b)
                K_rest = xp.vstack([K21_.T, K22]) ** (1 / self.para_h ** 2)
                Ki = xp.vstack([K11_p, Ki])
                K_whole = xp.hstack([Ki, K_rest])
                a = xp.exp(sol_w.dot(K_whole))
                a1 = a[:self.N1_]
                a2 = a[self.N1_:]

                # KL divergence computation
                K11_h = K11 ** (1 / self.h**2)
                diag_zero_K11_h = K11_h - xp.diag(xp.diag(K11_h))
                K21_h = K21_ ** (1 / self.h ** 2)
                f1 = a1.dot(diag_zero_K11_h) / (self.N1_ - 1)
                log_p1_loo = xp.log(f1)
                log_p2 = xp.log(a2.dot(K21_h) / (self.N2_))
                kl = cp.asnumpy((log_p1_loo - log_p2))
                kl = np.nanmean(kl)
                l_kl.append(kl)
        return np.array(l_kl)
