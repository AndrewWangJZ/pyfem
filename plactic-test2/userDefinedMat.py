from test3.SystemClass import *
from test3.ElasticCal import *

class VonMises(Material):
    def __init__(self, n_dim, E, mu, sigma_s, Et):
        super().__init__(n_dim, E, mu)
        self.Et = Et
        self.sigma_s0 = sigma_s
        self.harding = 0

    def yieldFunction(self, sigma, harding=None):
        f = getMisesStress(sigma)
        if harding:
            f_func = f - harding - self.sigma_s0
        else:
            f_func = f - self.harding - self.sigma_s0
        return f_func

    def yieldDiffSigma(self, sigma, harding=None):
        f_diff = getDeviatonicStress(sigma)
        J2 = np.einsum("ij, ij->", f_diff, f_diff) / 2
        f_diff = f_diff / np.sqrt(J2) * np.sqrt(3) / 2
        return f_diff

    def yieldDiffHarding(self, sigma=None, harding=None):
        return -1

    def hardeningFunction(self, sum_eq_eps):
        Ep = self.E * self.Et / (self.E - self.Et)
        harding = Ep * sum_eq_eps
        return harding

    def updateHarding(self, sum_eq_eps):
        self.harding = self.hardeningFunction(sum_eq_eps)

    def hardening_diff(self, sum_eq_eps=None):
        Ep = self.E * self.Et / (self.E - self.Et)
        h_diff = Ep
        return h_diff

class VonMises2(VonMises):
    def hardeningFunction(self, sum_eq_eps):
        harding = 500e6 * sum_eq_eps ** 0.3
        return harding

    def hardening_diff(self, sum_eq_eps=None):
        if sum_eq_eps and sum_eq_eps > 0:
            h_diff = -500e6 * 0.3 * sum_eq_eps ** (0.3 - 1.)
        else:
            h_diff = 0
        return h_diff