from test3.SystemClass import *

class Q4(ElementType):
    def getGauss(self, material):
        self.n_gauss = 4
        temp = np.sqrt(1 / 3)
        gauss_coord = np.array([[-temp, -temp], [temp, -temp],
                                [temp, temp], [-temp, temp]])
        self.gauss_points = [GaussPoint(coord_i, material)
                             for coord_i in gauss_coord]
        self.W = np.array([1, 1, 1, 1])

    def shapeFunction(self):
        xi, eta = sym.symbols('xi, eta')
        self.basis = sym.Matrix([xi, eta])
        N1 = (1 - xi) * (1 - eta) / 4
        N2 = (1 + xi) * (1 - eta) / 4
        N3 = (1 + xi) * (1 + eta) / 4
        N4 = (1 - xi) * (1 + eta) / 4
        self.N = sym.Matrix([N1, N2, N3, N4])

    def getV(self):
        v = 0
        for i in range(self.n_node - 1):
            v += np.linalg.det(self.x[i:(i + 2)])
        v += np.linalg.det(np.array([self.x[-1], self.x[0]]))
        v /= 2
        return abs(v)