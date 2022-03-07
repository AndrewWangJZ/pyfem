import numpy as np
import sympy
import sympy as sym
import matplotlib.pyplot as plt
import scipy as sp
from scipy.interpolate import griddata
import time


class TwoDimensionShape:
    def __init__(self, Nnum=4):
        self.ndim = 2
        self.Nnum = Nnum
        self.gaussianPoints = self.getGaussian()
        self.N, self.N_diff_local = self.getN_diff()
        self.N_diff_global = self.N_diff_local
        self.elementStiffness = self.getElementStiffness()

    def getGaussian(self):
        temp = np.sqrt(1 / 3)
        return np.array([[-temp, -temp], [temp, -temp], [temp, temp], [-temp, temp]])

    def getN_diff(self):
        xi, eta = sym.symbols('xi, eta')
        basis = sym.Matrix([xi, eta])
        N1 = (1 - xi) * (1 - eta) / 4
        N2 = (1 + xi) * (1 - eta) / 4
        N3 = (1 + xi) * (1 + eta) / 4
        N4 = (1 - xi) * (1 + eta) / 4
        N = sym.Matrix([N1, N2, N3, N4])
        N_diff = N.jacobian(basis)
        N_array = np.zeros(shape=(self.Nnum, self.Nnum))
        N_d_array = np.zeros(shape=(self.Nnum, self.Nnum, self.ndim))
        for i, gaussianPoint in enumerate(self.gaussianPoints):
            N_array[i] = np.array(N.subs([(xi, gaussianPoint[0]), (eta, gaussianPoint[1])])).astype(np.float32).reshape(-1)
            N_d_array[i] = N_diff.subs([(xi, gaussianPoint[0]), (eta, gaussianPoint[1])])
        return N_array, N_d_array

    def getElementStiffness(self, x=None, D=None):
        if x is None:
            # x = np.array([[-0.8, 0.], [0, -0.8], [0.8, 0], [0, 0.8]])
            x = np.array([[-1, -1], [1, -1], [1, 1], [-1, 1]])
        if D is None:
            E, mu = 1e8, 0.2
            lam, G = E * mu / (1 + mu) / (1 - 2 * mu), 0.5 * E / (1 + mu)
            D = np.zeros(shape=(2, 2, 2, 2))
            D[0, 0, 0, 0] = lam + G * 2
            D[1, 1, 1, 1] = lam + G * 2
            D[0, 0, 1, 1] = D[1, 1, 0, 0] = lam
            D[0, 1, 0, 1] = D[0, 1, 1, 0] = D[1, 0, 0, 1] = D[1, 0, 1, 0] = G
        je = np.einsum('ni,pnj->pij', x, self.N_diff_local)  # pij
        je_det = np.linalg.det(je)  # p
        je_inv = np.linalg.inv(je)  # pij -> pji
        self.N_diff_global = np.einsum('pmj,pji->pmi', self.N_diff_local, je_inv)
        # NOTE: VERY IMPORTANT!!!!!
        K_element = np.einsum('pmi, ijkl, pnk,p->mjnl', self.N_diff_global, D, self.N_diff_global, je_det)
        return K_element

    def displacementBoundaryCondition(self, K_global, mask, f):
        mask_free = np.equal(mask, 0)
        K_free = K_global[mask_free][:, mask_free]
        f_free = f[mask_free]
        return K_free, f_free

    def getStrain(self, u, node2Element, nodeCoord):
        uElementNode = u[node2Element]
        coordElementNode = nodeCoord[node2Element]
        je = np.einsum('ni,pnj->pij', nodeCoord, self.N_diff_local)  # pij
        # je_det = np.linalg.det(je)  # p
        je_inv = np.linalg.inv(je)  # pij -> pji
        N_diff_global = np.einsum('pmj,pji->pmi', self.N_diff_local, je_inv)
        u_grad = np.einsum('pmi, qmj->pqij', uElementNode, N_diff_global)
        epsilon = 0.5*(u_grad + np.einsum('pqij->pqji', u_grad))
        epsilonCoord = np.einsum('pmi, qm->pqi', coordElementNode, self.N)
        return epsilon, epsilonCoord

    def solve(self, mask, K_global, K_free, f_free, uValue):
        mask_free = np.equal(mask, 0)
        nNode = mask_free.shape[0]
        u_free = np.linalg.solve(K_free, f_free)
        u = np.zeros_like(mask_free, dtype=np.float)
        tempPointer = 0
        for i in range(nNode):
            for j in range(self.ndim):
                if mask_free[i, j]:
                    u[i, j] = u_free[tempPointer]
                    tempPointer += 1
                else:
                    u[i, j] = uValue[i, j]
        f_calculated = np.einsum('minj, nj->mi', K_global, u)
        return u, f_calculated

    def interpolateStrainToNode(self, nodecoord, node2Elment, epsilon):
        nNode = len(nodecoord)
        epsilonNodeAdd = np.zeros(shape=(nNode, self.ndim, self.ndim))
        nodeAddNum = np.zeros(shape=(nNode))
        N_inv = np.linalg.inv(self.N)
        for i, element in enumerate(node2Elment):
            epsilonElement = epsilon[i]
            epsilonNodeAdd[element] += np.einsum('mn, mij->nij', N_inv, epsilonElement)
            nodeAddNum[element] += 1
        nodeAddNum_inv = 1/nodeAddNum
        epsilonNode = np.einsum('pij, p->pij', epsilonNodeAdd, nodeAddNum_inv)
        return epsilonNode


def stiffnessAssembling(nodeIndex, kList, node2Element, ndim=2, Nnum=4):
    nodeNum = len(nodeIndex)
    elementNum = len(node2Element)
    k_global = np.zeros(shape=(nodeNum, ndim, nodeNum, ndim), dtype=np.float)
    for p in range(elementNum):
        ktemp = kList[p]
        elementNode = node2Element[p]
        for m in range(Nnum):
            for n in range(Nnum):
                k_global[elementNode[m], :, elementNode[n], :] += ktemp[m, :, n, :]
    return k_global


def plotElement(coord, *args):
    coord = np.concatenate((coord, coord[0:1]))
    plt.plot(coord[:, 0], coord[:, 1], *args)
    return

def plotGuassian(coord, *args):
    for gaussianPoints in coord:
        plt.plot(gaussianPoints[:, 0], gaussianPoints[:, 1], *args)
    for i, coorGaussian in enumerate(coord[0]):
        plt.text(x=coorGaussian[0], y=coorGaussian[1], s=str(i))
    return


def saveVTK(fileName, nodeCoord, node2Element, **kwargs):
    nodeNum = len(nodeCoord)
    elementNum = len(node2Element)
    with open(fileName, 'w') as f:
        f.write('# vtk DataFile Version 4.2\n')
        f.write('vtk file generated by meshmagick on %s\n' % time.strftime('%c'))
        f.write('ASCII\n')
        f.write('DATASET UNSTRUCTURED_GRID\n')
        f.write('\n')
        f.write('POINTS %u float\n' % nodeNum)
        for vertex in nodeCoord:
            f.write('%f %f %f\n' % (vertex[0], vertex[1], 0.))
        f.write('\n')
        f.write('CELLS %u %u\n' % (elementNum, 5 * elementNum))
        for face in node2Element:
            if face[0] == face[-1]:  # Triangle
                f.write('3 %u %u %u\n' % (face[0], face[1], face[2]))
            else:  # Quadrangle
                f.write('4 %u %u %u %u\n' % (face[0], face[1], face[2], face[3]))

        f.write('\n')
        f.write('CELL_TYPES %u\n' % len(node2Element))
        for i in range(len(node2Element)):
            f.write('9\n')

        # -----------------------------------------------------
        # writing the point data
        f.write('\nPOINT_DATA %u\n' % (nodeNum))
        # f.write("SCALARS test float\nLOOKUP_TABLE default\n")
        # for uu in u:
        #     f.write('%.8f\n' % (uu[0]))
        for key in kwargs:
            temp = kwargs[key]
            f.write("VECTORS %s float\n" % key)
            if temp[0].shape == (2, 2):
                for uu in temp:
                    f.write('%.8f %.8f %.8f\n' % (uu[0, 0], uu[1, 1], uu[0, 1]))
            else:
                for uu in temp:
                    f.write('%.8f %.8f %.8f\n' % (uu[0], uu[1], 0.))
    return


