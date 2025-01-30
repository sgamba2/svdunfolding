import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad
from scipy.optimize import curve_fit
from scipy.signal import fftconvolve
from matplotlib.colors import LogNorm
from scipy import stats
import random
from scipy.interpolate import interp1d
from scipy.stats import poisson
from scipy.fft import fft, fftfreq


nevents=100000
x0 = 1842 # M2
gamma0 = 6
x1 =  2047 # M1
gamma1 = 13.2
mu = 0
sigma=14.49
gaussian_par=[mu,sigma]
cauchy_par_test_0=[x0,gamma0]
cauchy_par_test_1=[x1,gamma1]

cauchy_par_0=[x0-10,gamma0+1]
cauchy_par_1=[x1+20,gamma1-1]

x_min=1400
x_max=2500
x_nbins=220
b_min=1400
b_max=2500
b_nbins=220
nevents_test_0=nevents-40000
nevents_test_1=nevents+100000

def bins_and_centers(nbins, low, high):
    bins = np.linspace(low, high, nbins + 1)
    bin_centers = (bins + (bins[1] - bins[0]) / 2)[:-1]
    return bins, bin_centers


def add_smearing(x):
    smear = np.random.normal(gaussian_par[0],gaussian_par[1])
    return x + smear


plt.figure(1)
xini_peak_0 = stats.cauchy.rvs(cauchy_par_0[0], cauchy_par_0[1], nevents) 
xini_peak_1 = stats.cauchy.rvs(cauchy_par_1[0], cauchy_par_1[1], nevents) 


xini_peak = np.append(xini_peak_0, xini_peak_1)

plt.hist(xini_peak, x_nbins, range=[x_min,x_max])
plt.title("xini variable")
plt.ylabel("events")
plt.xlabel("x")
plt.yscale("log")



plt.figure(2)
xini_peak_test_0 = stats.cauchy.rvs(cauchy_par_test_0[0], cauchy_par_test_0[1], nevents_test_0) 
xini_peak_test_1 = stats.cauchy.rvs(cauchy_par_test_1[0], cauchy_par_test_1[1], nevents_test_1) 

xini_peak_test = np.append(xini_peak_test_0, xini_peak_test_1)

plt.hist(xini_peak_test, x_nbins, range=[x_min,x_max])
plt.title("xtest variable")
plt.ylabel("events")
plt.xlabel("x")
plt.yscale("log")

plt.figure(3)
b_ini_gen = []

for i in xini_peak:
    b_ini_gen.append(add_smearing(i))


plt.hist(b_ini_gen, x_nbins, range=[x_min,x_max])
plt.title("bini variable")
plt.ylabel("events")
plt.xlabel("b")
plt.yscale("log")


plt.figure(4)
b_test_gen = []

for i in xini_peak_test:
    b_test_gen.append(add_smearing(i))

plt.hist(b_test_gen, x_nbins, range=[x_min,x_max])
plt.title("btest variable")
plt.ylabel("events")
plt.xlabel("b")
plt.yscale("log")


bins_x, bin_centers_x = bins_and_centers(x_nbins, x_min, x_max)
bins_b, bin_centers_b = bins_and_centers(b_nbins, b_min, b_max)

x_ini_gen = xini_peak
x_ini = np.histogram(x_ini_gen, bins_x)
x_test_gen = xini_peak_test
x_test = np.histogram(x_test_gen, bins_x)

scale_factor = len(x_test_gen)/len(x_ini_gen)

b_ini = np.histogram(b_ini_gen, bins_b)
b_test = np.histogram(b_test_gen, bins_b)
b_test_err = np.sqrt(b_test[0])


plt.figure(figsize=(15, 7))

plt.hist(x_ini_gen, density=False, bins=x_nbins, histtype="stepfilled", color="blue", 
         label="xini -> mie aspettative sulla distribuzione del processo", alpha=0.25, weights=scale_factor*np.ones(len(x_ini_gen)), range=[x_min, x_max])
plt.hist(x_test_gen, density=False, bins=x_nbins, histtype="stepfilled", color="red", label="xtest -> distribuzione reale che non so", alpha=0.25, range=[x_min, x_max])

binsb, bin_centers_b=bins_and_centers(b_nbins, b_min, b_max)

plt.plot(bin_centers_b, b_ini[0]*scale_factor, ".", color="black", label="$b_{ini}$-> mie aspettative sulla risposta del detector e sul processo (xini+error)")
plt.plot(bin_centers_b, b_test[0], ".", color="red", label="$b_{test}$ ->  true measured data distribution (xtest+error)")

plt.xlabel("x")
plt.ylabel("counts")
plt.yscale("log")
plt.legend()


plt.figure(7)
#btest è quello vero misurato (segnale vero+rumore), bini è mie aspettative misura + rumore, xini e xtest uguale a bini e btest senza rumore
response_hist, _, _ = np.histogram2d(b_ini_gen, x_ini_gen, bins=[bins_b,bins_x])
response_hist_err = np.sqrt(response_hist)
plt.imshow(response_hist, norm=LogNorm(vmin=1, vmax=1e4), cmap="YlGnBu", extent=[bins_b[0], bins_b[-1],bins_b[-1],bins_b[0]])
plt.colorbar()
plt.xlabel('Cause bins')
plt.ylabel('Effect bins')
plt.title('Normalizes response matrix')
"""
Whats needed for unfolding is not this 2-d counts histogram, 
but a matrix with the conditional probabilities P(Ei|Cj). 
This matrix, called the normalized response matrix, can be obtained 
by normalizing our 2-d counts matrix. That is, scale each column of 
response_hist such that it adds to our detection efficiencies.
"""
efficiencies = np.ones_like(b_ini[0], dtype=float)
efficiencies_err = np.full_like(efficiencies, 0.01, dtype=float)

column_sums = response_hist.sum(axis=0)
normalization_factor = efficiencies / column_sums
response = response_hist * normalization_factor
response_err = response_hist_err * normalization_factor

plt.figure(8)
plt.imshow(response, norm=LogNorm(vmin=1, vmax=1e4), cmap="YlGnBu", extent=[bins_b[0], bins_b[-1],bins_b[-1],bins_b[0]])
plt.colorbar()
plt.xlabel('Cause bins')
plt.ylabel('Effect bins')
plt.title('Normalizes response matrix')

"""
Observed distribution to unfold (with uncertainties): b_test, b_test_err
Detection efficiencies (with uncertainties): efficiencies, efficiencies_err
Normalized response matrix (with uncertainties): response, response_err
"""



def calc(dist1, dist2):
    dof=len(dist1)
    n1 = np.sum(dist1)
    n2 = np.sum(dist2)
    pi=(dist1+dist2)/(n1+n2)
    stat1=(dist1-n1*pi)*(dist1-n1*pi)/(n1*pi)
    stat2=(dist2-n2*pi)*(dist2-n2*pi)/(n2*pi)
    stat=sum((stat1+stat2)/dof)
    return stat

def one_over_safe(x):
    x = np.asarray(x)
    is_zero = x == 0
    with np.errstate(divide='ignore'):
        inv = 1 / x
    inv[is_zero] = 0
    return inv

def adye_propagation_corrections(dcdP, thetaij, prior, prior):
    return 
def update_errors(thetaij, num, prior_update, prior):
    dcdP= self._initialize_dcdP(Mij, f_norm, n_c, n_c_prev)
    dcdn = thetaij
    if iteration > 0:
        dcdn, dcdP = adye_propagation_corrections(dcdP, thetaij, prior, prior)
    return cov
def update(current_prior, response, data, efficiencies, iteration):
    num = response@current_prior
    one_over_num = one_over_safe(num)
    prior_eff = current_prior * one_over_safe(efficiencies)
    thetaij = response * prior_eff * one_over_num.reshape(-1, 1)
    prior_update = np.dot(data, thetaij)
    cov = update_errors(thetaij, num, prior_update, prior, iteration)
    return prior_update

def unfold(prior, stat, max_iter, response, data, efficiencies):
    current_n_c = prior
    iteration = 0
    tol=0.01
    while stat >= tol and iteration < max_iter:
        unfolded_n_c = update(current_n_c, response, data, efficiencies, iteration)
        iteration += 1
        stat = calc(unfolded_n_c, current_n_c)
        current_n_c = unfolded_n_c
        print('Iteration {}: ts = {:0.4f}, ts_stopping ={}\n'.format(iteration,stat,tol))
        
    return current_n_c





def poisson_covariance(ebins, cbins, pec_err):
    CovPP = np.zeros((cbins * ebins, cbins * ebins))
    for ej in np.arange(0, ebins):
        ejc = ej * cbins
        for ti in np.arange(0, cbins):
            CovPP[ejc+ti, ejc+ti] = pec_err[ej, ti]**2
    return CovPP


stat = np.inf
ts_stopping=0.01 
max_iter=100 
prior = np.full(len(efficiencies), 1/len(efficiencies))
n_c = np.sum(b_ini_gen) * prior  #prior counts distribution
print(np.array(response).shape)
current_n_c=unfold(np.array(n_c), stat, max_iter, np.array(response), np.array(b_test[0]), np.array(efficiencies))

cov_matrix = poisson_cov(response_err.shape[0],response_err.shape[1],response_err)


plt.figure(11)
plt.plot(bin_centers_x, current_n_c, marker=".", linestyle="", color="black")
plt.plot(bin_centers_x, b_test[0], marker=".", linestyle="", color="red")
plt.hist(x_test_gen, bins=x_nbins, range=[x_min,x_max])

plt.yscale("log")




plt.show()















