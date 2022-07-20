import inspect
import numpy as np
from scipy import interpolate
from scipy import linalg
import sys
import matplotlib.pyplot as plt

sys.path.append('/burg/home/as6131/CMB_dist_instrument/sd_foregrounds_optimize/')
sys.path.append('/burg/home/as6131/CMB_dist_instrument/specter_optimization_project/')

from NoiseFunctions import getnoise_nominal
import spectral_distortions as sd
import foregrounds_fisher as fg
ndp = np.float64
#ndp=np.float128
clight=299792458.
class FisherEstimation:
    def __init__(self, fmin=7.5e9, fmax=3.e12, fstep=15.e9, \
                 duration=86.4, bandpass=True, fsky=0.7, mult=1., \
                 priors={'alps':0.1, 'As':0.1}, drop=0, doCO=False, instrument='pixie',\
                  file_prefix='test',freq_bands=np.array([]), Ndet_arr=np.array([]),noisefile=None):

        self.fmin = fmin
        self.fmax = fmax
        self.bandpass_step = 1.e8
        self.fstep = fstep
        self.duration = duration
        self.bandpass = bandpass
        self.fsky = fsky
        self.mult = mult
        self.priors = priors
        self.drop = drop
        self.file_prefix=file_prefix

        if instrument=='specter':

            self.freq_bands=freq_bands
            self.Ndet_arr=Ndet_arr
            self.noisefile=noisefile
            self.center_frequencies, self.noise=self.specter_sensitivity()


        elif instrument=='pixie':
            self.set_frequencies()
            self.noise = self.pixie_sensitivity()

        elif instrument=='firas':

            self.center_frequencies, self.noise = self.firas_sensitivity()
        else:
            print("choose between pixie, firas or specter")
            return

        self.set_signals()

        if doCO:
            self.mask = ~np.isclose(115.27e9, self.center_frequencies, atol=self.fstep/2.)
        else:
            self.mask = np.ones(len(self.center_frequencies), bool)
            #print(self.mask)
        #print(self.center_frequencies)
        #print(self.noise)
        return

    def run_fisher_calculation(self):
        N = len(self.args)
        F = self.calculate_fisher_matrix()
        for k in self.priors.keys():
            if k in self.args and self.priors[k] > 0:
                kindex = np.where(self.args == k)[0][0]
                F[kindex, kindex] += 1. / (self.priors[k] * self.argvals[k])**2
        #print("fisher information matrix after priors & fiducial values")
        #print(F.diagonal())
        normF = np.zeros([N, N], dtype=ndp)
        for k in range(N):
            normF[k, k] = 1. / F[k, k]
        self.cov = ((np.mat(normF, dtype=ndp) * np.mat(F, dtype=ndp)).I * np.mat(normF, dtype=ndp)).astype(ndp)
        #self.cov=np.mat(F, dtype=ndp).I
        #self.cov=np.matmul(np.linalg.inv((np.matmul(normF,F))),normF)
        #self.cov=(linalg.inv(normF.dot(F.T))).dot(normF.T)


        #print("covariance matrix")
        #print(self.cov.diagonal())
        #self.cov = np.mat(F, dtype=ndp).I
        self.F = F
        self.get_errors()

        return

    def get_errors(self):
        self.errors = {}
        for k, arg in enumerate(self.args):
            self.errors[arg] = np.sqrt(self.cov[k,k])
        #print(self.errors)
        return

    def print_errors(self, args=None):
        if not args:
            args = self.args
        for arg in args:
            #print arg, self.errors[arg], self.argvals[arg]/self.errors[arg]
            print(arg, self.argvals[arg]/self.errors[arg])

    def set_signals(self, fncs=None):
        if fncs is None:
            fncs = [sd.DeltaI_mu, sd.DeltaI_reltSZ_2param_yweight, sd.DeltaI_DeltaT,
                    fg.thermal_dust_rad, fg.cib_rad, fg.jens_freefree_rad,
                    fg.jens_synch_rad, fg.spinning_dust, fg.co_rad]
        self.signals = fncs
        self.args, self.p0, self.argvals = self.get_function_args()
        return

    def set_frequencies(self):
        if self.bandpass:
            self.band_frequencies, self.center_frequencies, self.binstep = self.band_averaging_frequencies()
        else:
            self.center_frequencies = np.arange(self.fmin + self.fstep/2., \
                                                self.fmax + self.fstep, self.fstep, dtype=ndp)[self.drop:]
        return

    def band_averaging_frequencies(self):
        #freqs = np.arange(self.fmin + self.bandpass_step/2., self.fmax + self.fstep, self.bandpass_step, dtype=ndp)
        freqs = np.arange(self.fmin + self.bandpass_step/2., self.fmax + self.bandpass_step+self.fmin, self.bandpass_step, dtype=ndp)
        binstep = int(self.fstep / self.bandpass_step)
        #print(int((len(freqs) / binstep) * binstep))
        freqs = freqs[self.drop * binstep : int((len(freqs) / binstep) * binstep)]
        #print(len(freqs))
        centerfreqs = freqs.reshape((int(len(freqs) / binstep), binstep)).mean(axis=1)
        #self.windowfnc = np.sinc((np.arange(binstep)-(binstep/2-1))/float(binstep))
        return freqs, centerfreqs, binstep

    def pixie_sensitivity(self):
        sdata = np.loadtxt('/Users/asabyr/Documents/SecondYearProject/sd_foregrounds/templates/Sensitivities.dat', dtype=ndp)
        fs = sdata[:, 0] * 1e9
        sens = sdata[:, 1]
        template = interpolate.interp1d(np.log10(fs), np.log10(sens), bounds_error=False, fill_value="extrapolate")
        skysr = 4. * np.pi * (180. / np.pi) ** 2 * self.fsky
        if self.bandpass:
            N = len(self.band_frequencies)
            noise = 10. ** template(np.log10(self.band_frequencies)) / np.sqrt(skysr) * np.sqrt(15. / self.duration) * self.mult * 1.e26
            return (noise.reshape((int( N / self.binstep), self.binstep)).mean(axis=1)).astype(ndp)
        else:
            return (10. ** template(np.log10(self.center_frequencies)) / np.sqrt(skysr) * np.sqrt(15. / self.duration) * self.mult * 1.e26).astype(ndp)

    def specter_sensitivity(self):

        center_frequencies, sens=getnoise_nominal(prefix=self.file_prefix, bands=self.freq_bands, dets=self.Ndet_arr, precompute=self.noisefile)
        skysr = 4. * np.pi * (180. / np.pi) ** 2 * self.fsky

        # print((center_frequencies).astype(ndp))
        # print((sens/ np.sqrt(skysr) * np.sqrt(6./self.duration) * self.mult).astype(ndp))
        return (center_frequencies).astype(ndp),(sens/ np.sqrt(skysr) * np.sqrt(6./self.duration) * self.mult).astype(ndp)

    def firas_sensitivity(self):

        sdata=np.loadtxt('/Users/asabyr/Documents/software/sd_foregrounds_optimize/data/firas_sensitivity.txt', dtype=ndp, delimiter=',')
        firas_sigmas=np.loadtxt('/Users/asabyr/Documents/software/sd_foregrounds_optimize/data/firas_monopole_spec_v1.txt')
        fs_sigmas=firas_sigmas[:-2,0]*clight*10**2
        #print(fs_sigmas)
        #fs=firas_sigmas[:-2,0]*clight*10**2
        fs = sdata[:, 0] * 1e9
        #print(fs)
        sens = sdata[:, 1]
        #print("loaded firas sensitivities")
        #template = interpolate.interp1d(np.log10(fs), np.log10(sens), bounds_error=False, fill_value="extrapolate")

        #return fs_sigmas, 10. ** template(np.log10(fs_sigmas))
        return fs, sens


    def get_function_args(self):
        targs = []
        tp0 = []
        for fnc in self.signals:
            argsp = inspect.getargspec(fnc)
            args = argsp[0][1:]
            p0 = argsp[-1]
            targs = np.concatenate([targs, args])
            tp0 = np.concatenate([tp0, p0])
        return targs, tp0, dict(zip(targs, tp0))

    def calculate_fisher_matrix(self):
        N = len(self.p0)
        F = np.zeros([N, N], dtype=ndp)
        for i in range(N):
            dfdpi = self.signal_derivative(self.args[i], self.p0[i])
            #print(dfdpi)
            dfdpi /= self.noise
            #print(dfdpi)
            # if dfdpi[self.mask].any() < 0.:
            #     print(dfdpi)
            #     print(self.args[i])
            #     print(self.p0[i])

            for j in range(N):
                dfdpj = self.signal_derivative(self.args[j], self.p0[j])
                #print(dfdpj)
                dfdpj /= self.noise
                #print(dfdpj)
                # if dfdpj[self.mask].any() < 0.:
                #     print(dfdpj)
                #     print(self.args[j])
                #     print(self.p0[j])
                #F[i, j] = np.dot(dfdpi, dfdpj)
                F[i, j] = np.dot(dfdpi[self.mask], dfdpj[self.mask])
                # if F[i,j] < 0 :
                #     print(F[i,j])
                #     print(dfdpi[self.mask])
                #     print(dfdpj[self.mask])
        #print("fisher information matrix")
        #print(F)
        return F

    def signal_derivative(self, x, x0):
        h = 1.e-4
        zp = 1. + h
        deriv = (self.measure_signal(**{x: x0 * zp}) - self.measure_signal(**{x: x0})) / (h * x0)
        return deriv

    def measure_signal(self, **kwarg):
        if self.bandpass:
            frequencies = self.band_frequencies
        else:
            frequencies = self.center_frequencies

        N = len(frequencies)
        model = np.zeros(N, dtype=ndp)
        for fnc in self.signals:
            argsp = inspect.getargspec(fnc)
            args = argsp[0][1:]
            if len(kwarg) and list(kwarg.keys())[0] in args:
                model += fnc(frequencies, **kwarg)
        if self.bandpass:
            #rmodel = model.reshape((N / self.binstep, self.binstep))
            #total = rmodel * self.windowfnc
            return model.reshape((int(N / self.binstep), self.binstep)).mean(axis=1)
            #return total.mean(axis=1)
        else:
            #plt.loglog(frequencies, model)
            #print(model)
            return model
