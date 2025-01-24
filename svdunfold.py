import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad
from scipy.optimize import curve_fit
from scipy.signal import fftconvolve


class SVDunfold:
    '''
    class to unfold data
    '''

    def __init__(self, x_ini, b, A, xmin,xmax,bmin,bmax,eps):
        self._x_ini=x_ini #simulated to be the real energy
        self._b=b #measured
        self._A=A #2d event matrix
        self._xmin=xmin
        self._xmax=xmax
        self._bmin=bmin
        self._bmax=bmax
        self._r=None
        self._S=None
        self._Xinv=None
        self._d=None
        self._d_reg=None
        self._VT=None
        self._x_unfolded=None
        self._X_unfolded_covariance=None
        self._nx=len(self._x_ini[0])
        self._nb=len(self._b[0])
        self._cov=self.__diagonal_covariance()
        self._C=self.__second_deriv_matrix(eps)
        self._C_inv=np.linalg.inv(self._C)
        #assert(self._A.shape[0] == self._nb), "Wrong dimensions of A: bins in b != rows in A"
        #assert(self._A.shape[1] == self._nx), "Wrong dimensions of A: bins in x_ini != columns in A"

    def __second_deriv_matrix(self, xi):
        c_matrix = np.zeros((self._nx, self._nx))
        for i in range(self._nx):
            c_matrix[i, i] = -2 + xi
            if i - 1 >= 0:
                c_matrix[i, i - 1] = 1
            if i + 1 < self._nx:
                c_matrix[i, i + 1] = 1
        c_matrix[0, 0] = -1 + xi
        c_matrix[self._nx - 1, self._nx - 1] = -1 + xi
        return c_matrix

    def __diagonal_covariance(self):
        B = np.zeros(shape=(self._nb,self._nb))
        std_dev = np.sqrt(self._b[0])
        for i in range(self._nb):
            B[i, i] = std_dev[i] * std_dev[i] #indipendent bin values
        return B

        
    def pre_unfolding(self):
        Q, R, _ = np.linalg.svd(self._cov, full_matrices=False)
        self._r=np.sqrt(R)
        A_tilde=self.__get_A_tilde(Q)
        b_tilde=self.__get_b_tilde(Q)
        self._Xinv=self.__get_X_inv(A_tilde)
        Y=A_tilde@self._C_inv
        U, self._S, self._VT = np.linalg.svd(Y)
        self._d = U.T@b_tilde
       
    def unfolding(self,k):
        tau = self._S[k]**2
        self._d_reg = self._d * self._S**2 / (self._S**2 + tau)
        V=self._VT.T
        z_tau = self._d_reg * self._S / (self._S**2 + tau)
        w_tau = self._C_inv@V@z_tau
        Z_tau = np.zeros((self._nx, self._nx))
        z_diag = self._S**2 / (self._S**2 + tau)**2
        for i in range(self._nx):
            Z_tau[i, i] = z_diag[i]
        W_tau = self._C_inv@V@Z_tau@V.T@self._C_inv
        self._x_unfolded = w_tau * self._x_ini[0]
        self._X_unfolded_covariance = W_tau * (np.outer(self._x_ini[0], self._x_ini[0]))

    def __get_A_tilde(self,Q):
        A_tilde = Q@self._A
        for i in range(self._nb):
            for j in range(self._nx):
                A_tilde[i, j] = A_tilde[i,j] / self._r[i]
        return A_tilde

    def __get_b_tilde(self,Q):
        b_tilde = Q@self._b[0]
        for i in range(b_tilde.shape[0]):
            b_tilde[i] = b_tilde[i] / self._r[i]
        return b_tilde

    def __get_X_inv(self,A_tilde):
        X_inv = np.zeros((self._nx, self._nx))
        for j in range(self._nx):
            for h in range(self._nx):
                for i in range(self._nb):
                    X_inv[j, h] += (A_tilde[i, j] * A_tilde[i, h]) / (self._x_ini[0][j] * self._x_ini[0][h])
        return X_inv

    def get_singular_values(self):
        return self._S

    def get_X_unfolded_covariance(self):
        return self._X_unfolded_covariance

    def get_x_unfolded(self):
        return self._x_unfolded
        
    def get_abs_d(self):
        return np.abs(self._d)
        
    def get_cov(self):
        return self._cov