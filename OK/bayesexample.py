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
from bayes import BayesUnfold
import bayes

nevents=100000
x0 = 1842 # M2
gamma0 = 6
x1 =  2047 # M1
gamma1 = 13.2
mu = 0
sigma=6.
gaussian_par=[mu,sigma]
cauchy_par_test_0=[x0,gamma0]
cauchy_par_test_1=[x1,gamma1]

cauchy_par_0=[x0-10,gamma0+1]
cauchy_par_1=[x1+20,gamma1-1]

x_min=1400
x_max=2500
x_nbins=100
b_min=1400
b_max=2500
b_nbins=100
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
xini_peak_0 = stats.cauchy.rvs(cauchy_par_0[0], cauchy_par_0[1], nevents_test_0) 
xini_peak_1 = stats.cauchy.rvs(cauchy_par_1[0], cauchy_par_1[1], nevents_test_1) 


xini_peak = np.append(xini_peak_0, xini_peak_1)

plt.hist(xini_peak, x_nbins, range=[x_min,x_max])
plt.title("xini variable")
plt.ylabel("events")
plt.xlabel("x")
plt.yscale("log")



plt.figure(2)
xini_peak_test_0 = stats.cauchy.rvs(cauchy_par_test_0[0], cauchy_par_test_0[1], nevents) 
xini_peak_test_1 = stats.cauchy.rvs(cauchy_par_test_1[0], cauchy_par_test_1[1], nevents) 

xini_peak_test = np.append(xini_peak_test_0, xini_peak_test_1)

plt.hist(xini_peak_test, x_nbins, range=[x_min,x_max],histtype="stepfilled", color="blue", alpha=0.25)
plt.title("True measured variable")
plt.ylabel("counts/{} eV".format((x_max-x_min)/x_nbins))
plt.xlabel("Energy [eV]")
plt.yscale("log")
plt.xlim(x_min,x_max)
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

plt.hist(b_test_gen, x_nbins, range=[x_min,x_max],histtype="stepfilled", color="blue", alpha=0.75)
plt.title("Measured variable")
plt.ylabel("counts/{} eV".format((x_max-x_min)/x_nbins))
plt.xlabel("Energy [eV]")
plt.yscale("log")
plt.xlim(x_min,x_max)


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
ZERO_HANDLING = 2
prior = np.ones(len(x_ini[0])) / len(x_ini[0])
bayes = BayesUnfold(x_ini[0], response_hist, b_test[0], ZERO_HANDLING, prior)
niter=20
N=10000
bayes.processing(niter, N)


x = bayes.get_xm()
X = bayes.get_xs()

plt.subplots(2 , 1, figsize=(15, 10))
plt.subplot(2,1,1)
plt.hist(x_ini_gen, bins=x_nbins, range=[x_min, x_max], histtype="stepfilled", color="blue", alpha=0.25, label="xini -> mie aspettative sulla distribuzione del processo", weights=scale_factor*np.ones(len(x_ini_gen)))
plt.hist(x_test_gen, bins=x_nbins, range=[x_min, x_max], histtype="stepfilled", color="red", alpha=0.25, label="xtest -> distribuzione reale ")
plt.plot(bin_centers_b, b_test[0], ".", color="red", label="$b_{test}$ ->  true measured data distribution (xtest+error)")
plt.errorbar(bin_centers_x, x, fmt="k.", yerr=X, label="x unfolded")
plt.xlabel("Energy [eV]")
plt.ylabel("counts/{} eV".format((x_max-x_min)/x_nbins))
plt.legend()
plt.yscale("log")
plt.subplot(2,1,2)
plt.errorbar(bin_centers_x,(x_test[0]-x)/X ,marker='.', linestyle="", color="black", label="(xtest-xunfolded)/std.")
plt.xlabel("Energy [eV]")
plt.legend()
plt.ylabel("Norm. Res.")
plt.savefig("{}eVRMS_binwidth{}_bayes_new.pdf".format(sigma, (x_max-x_min)/x_nbins))
plt.show()



