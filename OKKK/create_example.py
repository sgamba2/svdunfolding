import numpy as np
from scipy import stats
import random


class CreateExample:
    '''
    class to create example
    '''
    def __init__(self, N_detectors, nevents, nevents1,x_min, x_max,x_nbins,b_min,b_max,b_nbins, type_unf):
        self.N_detectors = N_detectors
        self.nevents = nevents
        self.nevents1 = nevents1
        self.xmin = x_min
        self.xmax = x_max
        self.bmin = b_min
        self.bmax = b_max
        self.xnbins = x_nbins
        self.bnbins = b_nbins
        self.type_unf = type_unf
    def add_smearing(self,x, n):
        smear = np.random.normal(self.gaussian_par[0],self.gaussian_par[1][n])
        return x + smear


    def start_x_test_gen(self,x0,gamma0,x1, gamma1,x2, gamma2):
        self.x0 = x0
        self.gamma0 = gamma0
        self.x1 = x1
        self.gamma1 = gamma1
        self.x2 = x2
        self.gamma2 = gamma2
        cauchy_par_test_0=[self.x0,self.gamma0]
        cauchy_par_test_1=[self.x1,self.gamma1]
        cauchy_par_test_2=[self.x2,self.gamma2]
        
        nevents_test_bkg=np.random.randint(self.nevents-np.sqrt(self.nevents),self.nevents+np.sqrt(self.nevents),self.N_detectors)
        nevents_test_2=np.random.randint(self.nevents1-np.sqrt(self.nevents1),self.nevents1+np.sqrt(self.nevents1),self.N_detectors)
        nevents_test_1=np.random.randint(self.nevents1-np.sqrt(self.nevents1),self.nevents1+np.sqrt(self.nevents1),self.N_detectors)
        nevents_test_0=np.random.randint(int((self.nevents1-np.sqrt(self.nevents1))/3.),int((self.nevents1+np.sqrt(self.nevents1))/3.),self.N_detectors)
        xini_peak_test_0 = []
        xini_peak_test_1 = []
        xini_peak_test_2 = []
        xini_peak_test_bkg = []
        for i in range(self.N_detectors):
            xini_peak_test_0.append(stats.cauchy.rvs(cauchy_par_test_0[0], cauchy_par_test_0[1], nevents_test_0[i]))
            xini_peak_test_1.append(stats.cauchy.rvs(cauchy_par_test_1[0], cauchy_par_test_1[1], nevents_test_1[i]))
            xini_peak_test_2.append(stats.cauchy.rvs(cauchy_par_test_2[0], cauchy_par_test_2[1], nevents_test_2[i]))
            xini_peak_test_bkg.append(stats.halfnorm.rvs(0.,(self.xmax - self.xmin) / 3 , nevents_test_bkg[i]))

        xini_peak_test_0 = np.array(xini_peak_test_0, dtype=object)
        xini_peak_test_1 = np.array(xini_peak_test_1, dtype=object)
        xini_peak_test_2 = np.array(xini_peak_test_2, dtype=object)
        xini_peak_test_bkg = np.array(xini_peak_test_bkg, dtype=object)

        xini_peak_test = []
        for i in range(self.N_detectors):
            combined_row = np.concatenate([xini_peak_test_0[i], xini_peak_test_1[i], xini_peak_test_2[i], xini_peak_test_bkg[i]])
            xini_peak_test.append(combined_row)
        self.x_test_gen = np.array(xini_peak_test, dtype=object)

    def start_x_ini_gen(self,d0, dd0, d1, dd1, d2, dd2):
        cauchy_par_0=[self.x0+d0,self.gamma0+dd0]
        cauchy_par_1=[self.x1+d1,self.gamma1+dd1]
        cauchy_par_2=[self.x2+d2,self.gamma2+dd2]
        
        nevents_bkg=np.random.randint(self.nevents-np.sqrt(self.nevents),self.nevents+np.sqrt(self.nevents),self.N_detectors)
        nevents_2=np.random.randint(self.nevents1-np.sqrt(self.nevents1),self.nevents1+np.sqrt(self.nevents1),self.N_detectors)
        nevents_1=np.random.randint(self.nevents1-np.sqrt(self.nevents1),self.nevents1+np.sqrt(self.nevents1),self.N_detectors)
        nevents_0=np.random.randint(int((self.nevents1-np.sqrt(self.nevents1))/3.),int((self.nevents1+np.sqrt(self.nevents1))/3.),self.N_detectors)
        xini_peak_0 = []
        xini_peak_1 = []
        xini_peak_bkg = []
        xini_peak_2 = []

        for i in range(self.N_detectors):
            xini_peak_0.append(stats.cauchy.rvs(cauchy_par_0[0], cauchy_par_0[1], nevents_0[i]))
            xini_peak_1.append(stats.cauchy.rvs(cauchy_par_1[0], cauchy_par_1[1], nevents_1[i]))
            xini_peak_2.append(stats.cauchy.rvs(cauchy_par_2[0], cauchy_par_2[1], nevents_2[i]))
            xini_peak_bkg.append(stats.halfnorm.rvs(0.,(self.xmax - self.xmin) / 3 , nevents_bkg[i]))
        xini_peak_0 = np.array(xini_peak_0, dtype=object)
        xini_peak_1 = np.array(xini_peak_1, dtype=object)
        xini_peak_2 = np.array(xini_peak_2, dtype=object)
        xini_peak_bkg = np.array(xini_peak_bkg, dtype=object)
        xini_peak = []
        for i in range(self.N_detectors):
            combined_row = np.concatenate([xini_peak_0[i], xini_peak_1[i], xini_peak_2[i], xini_peak_bkg[i]])
            xini_peak.append(combined_row)
        self.x_ini_gen = np.array(xini_peak, dtype=object)

    def start_b(self,mu, sigma_0, sigma_1):
        sigma=np.random.uniform(sigma_0,sigma_1,self.N_detectors)
        self.gaussian_par=[mu,sigma]
        b_ini_gen = []
        b_test_gen = []
        for i in range(self.N_detectors):
            b_ini_gen.append([self.add_smearing(x, i) for x in self.x_ini_gen[i]])
            b_test_gen.append([self.add_smearing(x, i) for x in self.x_test_gen[i]])


        self.b_ini_gen = np.array(b_ini_gen, dtype=object)
        self.b_test_gen = np.array(b_test_gen, dtype=object)

    def generate_response_matrix(self,xx, bb, bins_xx,bins_bb):
        return np.histogram2d(xx, bb, bins=[bins_xx,bins_bb])

    def diagonal_covariance(self,b, nb, b_minn, b_maxx):
        data = np.histogram(b, bins=nb, range=(b_minn, b_maxx))[0]
        B = np.zeros(shape=(nb,nb))
        std_dev = np.sqrt(data)
        for i in range(nb):
            B[i, i] = std_dev[i] * std_dev[i] 
        return B
    def processing(self):
        self.bins_b = np.linspace(self.bmin, self.bmax, self.bnbins + 1)
        self.bin_centers_b = (self.bins_b + (self.bins_b[1] - self.bins_b[0]) / 2)[:-1]           
        self.bins_x = np.linspace(self.xmin, self.xmax, self.xnbins + 1)
        self.bin_centers_x = (self.bins_x + (self.bins_x[1] - self.bins_x[0]) / 2)[:-1]
        x_ini = []
        x_test = []
        b_ini = []
        b_test = []
        self.scale_factor = []
        for i in range(self.N_detectors):

            x_ini.append(np.histogram(self.x_ini_gen[i], self.bins_x)[0])
            x_test.append(np.histogram(self.x_test_gen[i], self.bins_x)[0])
            self.scale_factor.append(len(self.x_test_gen[i])/len(self.x_ini_gen[i]))
            b_ini.append(np.histogram(self.b_ini_gen[i], self.bins_b)[0])
            b_test.append(np.histogram(self.b_test_gen[i], self.bins_b)[0])

        self.b_ini = np.array(b_ini, dtype=float)
        self.x_ini = np.array(x_ini, dtype=float)
        self.x_test = np.array(x_test, dtype=float)
        self.b_test = np.array(b_test, dtype=float)
        b_test_err = []
        x_test_err = []
        b_ini_err = []
        for i in range(self.N_detectors):
            b_test_err.append(np.sqrt(self.b_test[i].astype("float")))
            x_test_err.append(np.sqrt(self.x_test[i].astype("float")))
            b_ini_err.append(np.sqrt(self.b_ini[i].astype("float")))
        self.b_test_err = np.array(b_test_err, dtype=float)
        self.x_test_err = np.array(x_test_err, dtype=float)
        self.b_ini_err = np.array(b_ini_err, dtype=float)
        if self.type_unf == "SVD":
            A_matrix = []
            X1 = []
            B = []
            for i in range(self.N_detectors):
                aa = self.x_ini_gen[i]
                bb = self.b_ini_gen[i]
                cc = self.bins_x
                dd = self.bins_b
                A_matrix.append(self.generate_response_matrix(aa, bb, cc, dd)[0].T)
                X1.append(np.sqrt(np.diag(A_matrix[i])))
                aaa = self.b_test_gen[i]
                bbb = self.bnbins
                ccc = self.bmin
                ddd = self.bmax
                B.append(self.diagonal_covariance(aaa, bbb, ccc, ddd))
            self.B = np.array(B, dtype=float)
            self.A_matrix = np.array(A_matrix, dtype=float)
            self.X1 = np.array(X1, dtype=float)
        if self.type_unf == "BAYES":
            response_hist = []
            response_hist_err = []
            efficiencies = []
            efficiencies_err = []
            column_sums = []
            normalization_factor = []
            response = []
            response_err = []
            unfolded_results = []
            for i in range(self.N_detectors):
                response_hist.append(np.histogram2d(self.b_ini_gen[i], self.x_ini_gen[i], bins=[self.bins_b,self.bins_x])[0])
                response_hist_err.append(np.sqrt(response_hist[i]))
                efficiencies.append(np.ones_like(self.b_test[i], dtype=float))
                efficiencies_err.append(np.full_like(efficiencies[i], 0.1, dtype=float))
                column_sums.append(response_hist[i].sum(axis=0))
                normalization_factor.append(efficiencies[i] / column_sums[i])
                response.append(response_hist[i] * normalization_factor[i])
                response_err.append(response_hist_err[i] * normalization_factor[i])
            self.efficiencies = np.array(efficiencies, dtype=float)
            self.efficiencies_err = np.array(efficiencies_err, dtype=float)
            self.response = np.array(response, dtype=float)
            self.response_err = np.array(response_err, dtype=float)
    def get_x_ini_gen(self):
        return self.x_ini_gen
    def get_response(self):
        return self.response
    def get_efficiencies_err(self):
        return self.efficiencies_err
    def get_efficiencies(self):
        return self.efficiencies
    def get_response_err(self):
        return self.response_err
    def get_A_matrix(self):
        return self.A_matrix
    def get_B(self):
        return self.B
    def get_X1(self):
        return self.X1
    def get_b_ini_gen(self):
        return self.b_ini_gen
    def get_x_test_gen(self):
        return self.x_test_gen
    def get_b_test_gen(self):
        return self.b_test_gen
    def get_x_ini(self):
        return self.x_ini
    def get_b_ini(self):
        return self.b_ini
    def get_x_test(self):
        return self.x_test
    def get_x_test_err(self):
        return self.x_test_err
    def get_b_test_err(self):
        return self.b_test_err
    def get_b_ini_err(self):
        return self.b_ini_err
    def get_b_test(self):
        return self.b_test
    def get_bins_x(self):
        return self.bins_x
    def get_bin_centers_x(self):
        return self.bin_centers_x
    def get_scale_factor(self):
        return self.scale_factor