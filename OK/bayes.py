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
    
    def __find_holes(self,v):
        """
        Finds consecutive null elements of a vector v.
        """
        ind = []
        i1 = i2 = 0
        l = len(v)
    
        for i in range(l):
            if (v[i] or i == l - 1) and i1:
                i2 = i if (i == l - 1 and not v[i]) else i - 1
                ind.append((i1, i2))
                i1 = i2 = 0
            elif not v[i] and not i1:
                i1 = i
                if i == l - 1:
                    ind.append((i1, i1))
    
        return np.array(ind)

    def __my_alphas(self):
        """
        Returns customized Dirichlet alpha's.
        """
        g_reg = np.exp(-1)
        self.alpha = np.zeros((self.nE+1, self.nC))
    
        for i in range(self.nC):
            a_i = np.append(self.response_hist[:, i], self.x_ini[i].astype(np.float64) - np.sum(self.response_hist[:, i].astype(np.float64)))
            a_i[a_i > 0] += 1 / (self.nE+1)
            ind = self.__find_holes(a_i[:self.nE])
            a_i = np.append(self.__reg_alphas(a_i[:self.nE], ind, g_reg, 1 / (self.nE+1)), a_i[-1])
            self.alpha[:, i] = a_i



    def __reg_alphas(self,v, ind, g, alpha_init):
        """
        Regularize Dirichlet posterior alphas in effect bins with no MC entries from a given cause.
        """
        l = len(ind)
        m = len(v)
        if l == 0:
            return v
    
        r = np.zeros(m, dtype=bool)
    
        for i1, i2 in ind:
            if i2 != m - 1:
                for j in range(i2, i1 - 1, -1):
                    v[j] += g ** (i2 - j) * alpha_init
                    r[j] = True
            if i1 != 0:
                for j in range(i1, i2 + 1):
                    v[j] += g ** (j - i1) * alpha_init
                    r[j] = True
    
        sum_alpha_reg = (np.sum(r) / (self.nE + 1)) / np.sum(v[r])
        v[r] *= sum_alpha_reg
    
        return v

    def __my_gamma_par(self):
        """
        Returns customized gamma parameters.
        """
        gamma_c_in = np.ones(self.nE)
        gamma_r_in = np.zeros(self.nE)
    
        c_max = 1
        r_min = c_max / min(10, np.min(self.obs[self.obs > 0]) + 1) if np.any(self.obs > 0) else c_max / 10
    
        gamma_c_in[self.obs == 0] = c_max / np.sum(self.obs == 0)
        gamma_r_in[self.obs == 0] = r_min
    
        gamma_par_c= gamma_c_in + self.obs
        gamma_par_r = gamma_r_in + 1
    
        return gamma_par_c, gamma_par_r


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
        else:
            self.__my_alphas()
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

        if self.ZERO_HANDLING < 2:
            gamma_c_fin = np.ones_like(self.obs) + self.obs
            gamma_r_fin = np.ones_like(self.obs)
        else:
            gamma_c_fin, gamma_r_fin = self.__my_gamma_par()

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

    def processing(self, niter, N):
        self.N = N
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