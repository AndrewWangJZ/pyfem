import numpy as np
import sympy as sym
import copy

class Material(object):
    def __init__(self, n_dim, E, mu):
        self.n_dim = n_dim
        self.E = E
        self.mu = mu
        lam = E * mu / (1 + mu) / (1 - 2 * mu)
        G = 0.5 * E / (1 + mu)
        self.D = np.zeros(shape=(self.n_dim, self.n_dim,
                                 self.n_dim, self.n_dim))
        for i in range(self.n_dim):
            for j in range(self.n_dim):
                self.D[i, j, i, j] = self.D[i, j, j, i] = G
        for i in range(self.n_dim):
            for j in range(self.n_dim):
                self.D[i, i, j, j] = lam
            self.D[i, i, i, i] = lam + 2 * G
        self.Dp = np.zeros_like(self.D)
        self.Dep = self.D - self.Dp

    def yieldFunction(self, sigma, harding=None):
        pass

    def yieldDiffSigma(self, sigma, harding=None):
        pass

    def yieldDiffHarding(self, sigma, harding):
        pass

    def potentialFunction(self, sigma):
        g_func = self.yieldFunction(sigma)
        return g_func

    def potentialDiff(self, sigma):
        g_diff = self.yieldDiffSigma(sigma)
        return g_diff

    def hardeningFunction(self, sum_eq_eps):
        pass

    def updateHarding(self, sum_eq_eps):
        pass

    def hardening_diff(self, sum_eq_eps):
        pass

    def updateDep(self, sigma, sum_eq_eps):
        f_diff = self.yieldDiffSigma(sigma)
        g_diff = self.potentialDiff(sigma)
        h_diff = self.hardening_diff(sum_eq_eps)
        A = -h_diff * np.sqrt(2 / 3 * np.einsum("ij, ij->", g_diff, g_diff))
        H = np.einsum("ij, ijkl, kl->", f_diff, self.D, g_diff) + A
        self.Dp = np.einsum("mnkl, kl, ij, ijrs->mnrs",
                            self.D, g_diff, f_diff, self.D) / H
        self.Dep = self.D - self.Dp

class GaussPoint(object):
    def __init__(self, coord, material):
        self.n_dim = np.shape(coord)[0]
        self.coord = coord
        self.mat = copy.deepcopy(material)
        self.sigma = np.zeros(shape=(self.n_dim, self.n_dim))
        self.sum_eq_eps = 0

    def plasticJudge(self, d_eps, f_tole):
        r_min = 0
        r_max = 1
        r = (r_max + r_min) / 2
        sigma_r = self.sigma + r * np.einsum("ijkl, kl->ij", self.mat.D, d_eps)
        f_r = self.mat.yieldFunction(sigma_r)
        while abs(f_r) > f_tole:
            if f_r > 0:
                r_max = r
            else:
                r_min = r
            r = (r_max + r_min) / 2
            sigma_r = self.sigma + r * np.einsum("ijkl, kl->ij", self.mat.D, d_eps)
            f_r = self.mat.yieldFunction(sigma_r)
        return r, sigma_r

    def plasticReturnMapping(self, d_eps, f_tole):
        iteration = 0
        sigma_test = self.sigma + np.einsum("ijkl, kl->ij", self.mat.D, d_eps)
        sum_eq_eps = self.sum_eq_eps
        f = self.mat.yieldFunction(sigma_test)
        while abs(f) > f_tole:
            f_diff = self.mat.yieldDiffSigma(sigma_test)
            g_diff = self.mat.potentialDiff(sigma_test)
            h_diff = self.mat.yieldDiffHarding() * self.mat.hardening_diff(sum_eq_eps)
            A = h_diff * np.sqrt(2 / 3 * np.einsum("ij, ij->", g_diff, g_diff))
            H = np.einsum("ij, ijkl, kl->", f_diff, self.mat.D, g_diff) - A
            d_lambda = f / H
            d_sigma = -d_lambda * np.einsum("ijkl, kl->ij", self.mat.D, g_diff)
            d_eq_eps = d_lambda * (A / h_diff)
            # print(f + np.einsum("ij, ij->", f_diff, d_sigma) + h_diff * d_eq_eps)
            sigma_test += d_sigma
            sum_eq_eps += d_eq_eps
            h = self.mat.hardeningFunction(sum_eq_eps)
            f = self.mat.yieldFunction(sigma_test, h)
            # print(f)
            iteration += 1
            if iteration >= 1000:
                raise ValueError("Iteration number exceeds!!!")
        self.sum_eq_eps = sum_eq_eps
        self.sigma = sigma_test
        self.mat.updateHarding(sum_eq_eps)
        self.mat.updateDep(sigma_test, sum_eq_eps)

    def PlasticIteration(self, d_eps):
        f_tole = 1e-5
        sigma_1 = self.sigma + np.einsum("ijkl, kl->ij", self.mat.D, d_eps)
        f_1 = self.mat.yieldFunction(sigma_1)
        if f_1 <= f_tole:
            self.sigma = sigma_1
        elif abs(self.mat.yieldFunction(self.sigma)) <= f_tole:
            self.plasticReturnMapping(d_eps, f_tole)
        else:
            r, sigma_r = self.plasticJudge(d_eps, f_tole)
            self.sigma = sigma_r
            self.plasticReturnMapping((1 - r) * d_eps, f_tole)

class ElementType(object):
    def __init__(self, node_list, x, material):
        self.node_list = node_list
        self.x = x
        self.n_node, self.n_dim = np.shape(x)
        self.getGauss(material)
        self.shapeFunction()
        self.getN_diff()
        self.getElementStiffness()

    def getGauss(self, material):
        self.n_gauss = self.n_node
        self.gauss_points = [GaussPoint(np.zeros(self.n_dim), material)
                             for i in range(self.n_gauss)]
        self.W = np.zeros(self.n_gauss)

    def shapeFunction(self):
        self.basis = sym.Matrix([])
        self.N = sym.Matrix([])

    def getN_diff(self):
        self.N_diff = self.N.jacobian(self.basis)
        self.N_array = np.zeros(shape=(self.n_gauss, self.n_node))
        self.N_d_local = np.zeros(shape=(self.n_gauss, self.n_node, self.n_dim))
        for i, gauss_point in enumerate(self.gauss_points):
            subs_basis = [(self.basis[i], gauss_point.coord[i])
                          for i in range(self.n_dim)]
            self.N_array[i] = np.array(self.N.subs(subs_basis)).astype(np.float32).reshape(-1)
            self.N_d_local[i] = self.N_diff.subs(subs_basis)

    def getElementStiffness(self):
        je = np.einsum('pni,nj->pji', self.N_d_local, self.x)
        je_det = np.linalg.det(je)
        je_inv = np.linalg.inv(je)
        self.N_d_global = np.einsum('pmj,pji->pmi', self.N_d_local, je_inv)
        D = np.array([gauss_i.mat.Dep for gauss_i in self.gauss_points])
        self.K_element = np.einsum('pmi, pijkl, pnk, p, p->mjnl',
                                   self.N_d_global, D,
                                   self.N_d_global, je_det, self.W)

    def d_u2d_eps(self, d_u):
        d_eps = (np.einsum("pki, kj->pij", self.N_d_global, d_u) +
                 np.einsum("pki, kj->pji", self.N_d_global, d_u)) / 2
        return d_eps

    def PlasticIteration(self, d_u):
        d_eps = self.d_u2d_eps(d_u)
        for i, gauss_i in enumerate(self.gauss_points):
            gauss_i.PlasticIteration(d_eps[i])

    def updateElementStiffness(self, d_u):
        self.x += d_u
        je = np.einsum('pni,nj->pji', self.N_d_local, self.x)
        je_det = np.linalg.det(je)
        je_inv = np.linalg.inv(je)
        self.N_d_global = np.einsum('pmj,pji->pmi', self.N_d_local, je_inv)
        D = np.array([gauss_i.mat.Dep for gauss_i in self.gauss_points])
        self.K_element = np.einsum('pmi, pijkl, pnk, p, p->mjnl',
                                   self.N_d_global, D,
                                   self.N_d_global, je_det, self.W)
