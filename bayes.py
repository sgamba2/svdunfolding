import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad
from scipy.optimize import curve_fit
from scipy.signal import fftconvolve

class BayesUnfold:
    '''
    Class to perform Bayesian unfolding of data.
    '''
    def __init__(self, xini, response_hist, obs, ZERO_HANDLING, prior):
        self.x_ini = xini
        self.response_hist = response_hist
        self.obs = obs
        self.prior = prior
        self.ZERO_HANDLING = ZERO_HANDLING
        self.nC = len(self.x_ini)
        self.nE = len(self.response_hist)
    
    def __infer_pec(self):
        self.pec = np.zeros((self.nE, self.nC))
        self.E_pec = np.zeros((self.nE + 1, self.nC))
        self.S_pec = np.zeros((self.nE + 1, self.nC))
        self.eff = np.zeros(self.nC)
        
        if abs(self.ZERO_HANDLING) < 2:
            alpha_in = self.ZERO_HANDLING if self.ZERO_HANDLING in {0, 1} else 1 / self.nE
            self.alpha = np.full((self.nE + 1, self.nC), alpha_in, dtype=np.float64)
            for i in range(self.nC):
                self.alpha[:, i] += np.append(self.response_hist[:, i], self.x_ini[i].astype(np.float64) - np.sum(self.response_hist[:, i].astype(np.float64)))
        
        for i in range(self.nC):
            sa = np.sum(self.alpha[:, i])
            self.E_pec[:, i] = self.alpha[:, i] / sa
            self.S_pec[:, i] = np.sqrt(self.alpha[:, i] * (sa - self.alpha[:, i]) / (sa**2 * (sa + 1)))
            self.pec[:, i] = self.E_pec[:self.nE, i]
            self.eff[i] = 1 - self.E_pec[self.nE, i]
    
    def __MC_unfold(self):
        smear= self.pec        
        pec = np.zeros((self.nE, self.nC))
        eff = np.zeros(self.nC)
        sx = np.zeros(self.nC)
        sxij = np.zeros((self.nC, self.nC))
        
        if np.any(smear > 1):
            print("\nP(E_j|C_i) will be sampled using Dirichlet")
        else:
            print("fixed P(E_j|C_i) used")
            pec = smear.copy()
            eff = np.sum(pec, axis=0)

        if self.zero_h < 2:
            gamma_c_fin = np.ones_like(self.obs) + self.obs
            gamma_r_fin = np.ones_like(self.obs)
        else:
            gamma_c_fin, gamma_r_fin = my_gamma_par(self.obs)


        
        for n in range(1, self.N + 1):
            if np.any(smear > 1):
                for i in range(self.nC):
                    dirichlet_sample = np.random.dirichlet(smear[:, i])
                    pec[:, i] = dirichlet_sample[:self.nE]
                    eff[i] = 1 - dirichlet_sample[self.nE]
            
            pce = np.zeros((self.nC, self.nE))
            for j in range(self.nE):
                pce[:, j] = pec[j, :] * self.prior / np.sum(pec[j, :] * self.prior)
            
            evc = np.zeros(self.nC)
            for j in range(self.nE):
                lambda_ = np.random.gamma(gamma_c_fin[j], 1 / gamma_r_fin[j])
                n_mult = np.max(np.round(lambda_))
                scale = lambda_ / n_mult
                evcj = np.random.multinomial(n_mult, pce[:, j]) * scale if self.obs[j] != 0 else np.zeros(self.nC)
                evc += evcj
            
            evc /= eff
            sx += evc
            sxij += np.outer(evc, evc)
        
        self.xm = sx / self.N
        self.prior = self.xm / np.sum(self.xm)
        self.covx = sxij / self.N - np.outer(self.xm, self.xm)
        self.xs = np.sqrt(np.diag(self.covx))
        self.rho = self.covx / np.outer(self.xs, self.xs)

    def __unfold(self):
        pce = np.zeros((self.nC, self.nE))
        for j in range(self.nE):
            pce[:, j] = self.pec[j, :] * self.prior / np.sum(self.pec[j, :] * self.prior)
        self.xm = (pce @ self.obs) / self.eff
        self.prior = self.xm / np.sum(self.xm)
        return self.xm, self.prior

    def processing(self, niter, zero_h, N):
        self.N = N
        self.zero_h = zero_h
        self.intermediate = []
        self.int_prior = []
        self.__infer_pec() 
        for iter in range(1, niter + 1):
            print(f"\n  Iterative unfolding. Iteration {iter} of {niter}:")
            if iter < niter:
                unfolded, int_prior = self.__unfold()
                self.intermediate.append(unfolded)
                self.int_prior.append(int_prior)
            else:
                self.__MC_unfold() 

    def get_xs(self):
        return self.xs
    def get_xm(self):
        return self.xm
    def get_N(self):
        return self.N
    def get_zero_h(self):
        return self.zero_h
    def get_prior(self):
        return self.prior    
    def get_covx(self):
        return self.covx 
    def get_rho(self):
        return self.rho
    def get_ZERO_HANDLING(self):
        return self.ZERO_HANDLING
    def get_intermediate(self):
        return self.intermediate
    def get_int_prior(self):
        return self.int_prior
    def get_nE(self):
        return self.nE
    def get_nC(self):
        return self.nC
    def get_E_pec(self):
        return self.E_pec
    def get_S_pec(self):
        return self.S_pec
    def get_pec(self):
        return self.pec
    def get_alpha(self):
        return self.alpha
    def get_x_ini(self):
        return self.x_ini
    def get_response_hist(self):
        return self.response_hist
    def get_obs(self):
        return self.obs