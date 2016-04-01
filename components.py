import numpy as np
import math, os
import glob
from scipy import interpolate
from scipy import integrate
from scipy import special
import matplotlib
matplotlib.use('pdf')
matplotlib.rc('font', family='serif', serif='cm10')
matplotlib.rc('text', usetex=True)
fontProperties = {'family':'sans-serif',
    'weight' : 'normal', 'size' : 20}
import matplotlib.pyplot as plt

"""
A model for each component signal in the spectral distortion sky.
The function for each component takes as input a vector of frequency values (in Hz), a parameter (or parameters) describing the amplitude,
and (if necessary) other parameters that specify the SED, which we may want to vary (e.g., dust spectral index).
The output is Delta I, the specific intensity measured with respect to the assumed fiducial blackbody spectrum at T_CMB = 2.726 K.
"""

# constants (MKS units, except electron rest mass)
TCMB = 2.726 #Kelvin
hplanck=6.626068e-34 #MKS
kboltz=1.3806503e-23 #MKS
clight=299792458.0 #MKS
m_elec = 510.999 #keV!

# Delta_T distortion (i.e., this accounts for the difference in the true CMB temperature from our assumed value)
#   N.B. I am only working to lowest order here, we may want to go beyond this (see Eq. 2 of http://arxiv.org/pdf/1306.5751v2.pdf and discussion thereafter; also Sec. 4.1 of that paper)
def DeltaI_DeltaT(freqs, DeltaT_amp): #freqs in Hz, DeltaT_amp dimensionless (DeltaT = (T_CMB_true - T_CMB_assumed)/T_CMB_assumed), DeltaI_DeltaT in W/m^2/Hz/sr
    X = hplanck*freqs/(kboltz*TCMB)
    return DeltaT_amp * X**4.0 * np.exp(X)/(np.exp(X) - 1.0)**2.0 * 2.0*(kboltz*TCMB)**3.0 / (hplanck*clight)**2.0

# y-type distortion (i.e., non-relativistic tSZ) -- see e.g. Eq. 6 of Hill+2015
def DeltaI_y(freqs, y_amp): #freqs in Hz, y_amp dimensionless, DeltaI_y in W/m^2/Hz/sr
    X = hplanck*freqs/(kboltz*TCMB)
    return y_amp * (X / np.tanh(X/2.0) - 4.0) * X**4.0 * np.exp(X)/(np.exp(X) - 1.0)**2.0 * 2.0*(kboltz*TCMB)**3.0 / (hplanck*clight)**2.0

# mu-type distortion -- see e.g. Section 2.3 of 1306.5751
def DeltaI_mu(freqs, mu_amp): #freqs in Hz, mu_amp dimensionless, DeltaI_mu in W/m^2/Hz/sr
    X = hplanck*freqs/(kboltz*TCMB)
    return mu_amp * (X / 2.1923 - 1.0)/X * X**4.0 * np.exp(X)/(np.exp(X) - 1.0)**2.0 * 2.0*(kboltz*TCMB)**3.0 / (hplanck*clight)**2.0

# r-type distortion (first non-mu/non-y eigenmode -- this is only approximately correct for us to use here, but let's stick with it for now)
def DeltaI_r(freqs, r_amp): #freqs in Hz, r_amp dimensionless, DeltaI_r in W/m^2/Hz/sr
    X = hplanck*freqs/(kboltz*TCMB)
    # first r-distortion eigenmode from Jens (Fig. 4 of 1306.5751)
    rfile = np.loadtxt('templates/PCA_mode_1.dat')
    Xr = hplanck*rfile[:,0]*1e9/(kboltz*TCMB) #convert from GHz to Hz
    DeltaIr = rfile[:,1]*1e-18*r_amp #conver to W/m^2/Hz/sr
    # linearly interpolate (set to zero above the highest frequency in Jens's file (his lowest frequency is 30 GHz, so things are OK on that end--put in a crazy value so we catch it if needed))
    return np.interp(X, Xr, DeltaIr, left=-1e6, right=0.0)

# relativistic tSZ distortion ("beyond y") -- see Hill+2015
#   N.B. although this signal in principle requires an infinite number of parameters to be fully specified, in practice PIXIE will only be sensitive to (at best) one, which is
#      the mean tau-weighted ICM electron temperature (kT_moments[0], below).  We'll hold all the others fixed, or perhaps just put reasonable priors on them and marginalize over them.
def DeltaI_reltSZ(freqs, tau_ICM, kT_moments): #freqs in Hz, tau_ICM dimensionless, kT_moments in keV^n, DeltaI_reltSZ in W/m^2/Hz/sr; code uses up to kT_moments[3] (Eq. 8 of Hill+2015 with n=4)
    # convert to Jens's moment definitions -- this formalism follows Chluba+2013 (MNRAS, 430, 3054)
    w1 = (kT_moments[1])/(kT_moments[0])**2 - 1.0
    w2 = (kT_moments[2])/(kT_moments[0])**3 - 3.0*(kT_moments[1])/(kT_moments[0])**2 + 2.0
    w3 = (kT_moments[3])/(kT_moments[0])**4 - 4.0*(kT_moments[2])/(kT_moments[0])**3 + 6.0*(kT_moments[1])/(kT_moments[0])**2 - 3.0
    #w4 not included for now; stop at w3, which corresponds to <(kT)^4> order
    #w4 = (kT_moments[4])/(kT_moments[0])**5 - 5.0*(kT_moments[3])/(kT_moments[0])**4 + 10.0*(kT_moments[2])/(kT_moments[0])**3 - 10.0*(kT_moments[1])/(kT_moments[0])**2 + 4.0
    X = hplanck*freqs/(kboltz*TCMB)
    Xtwid = X*np.cosh(0.5*X)/np.sinh(0.5*X)
    Stwid = X/np.sinh(0.5*X)
    # relativistic expressions from Nozawa+2006
    Y0=Xtwid-4.0 #N.B. the Y0 term is the non-relativistic tSZ effect (pure y), which we have already accounted for in the DeltaI_y function above
    Y1=-10.0+23.5*Xtwid-8.4*Xtwid**2+0.7*Xtwid**3+Stwid**2*(-4.2+1.4*Xtwid)
    Y2=-7.5+127.875*Xtwid-173.6*Xtwid**2.0+65.8*Xtwid**3.0-8.8*Xtwid**4.0+0.3666667*Xtwid**5.0+Stwid**2.0*(-86.8+131.6*Xtwid-48.4*Xtwid**2.0+4.7666667*Xtwid**3.0)+Stwid**4.0*(-8.8+3.11666667*Xtwid)
    Y3=7.5+313.125*Xtwid-1419.6*Xtwid**2.0+1425.3*Xtwid**3.0-531.257142857*Xtwid**4.0+86.1357142857*Xtwid**5.0-6.09523809524*Xtwid**6.0+0.15238095238*Xtwid**7.0+Stwid**2.0*(-709.8+2850.6*Xtwid-2921.91428571*Xtwid**2.0+1119.76428571*Xtwid**3.0-173.714285714*Xtwid**4.0+9.14285714286*Xtwid**5.0)+Stwid**4.0*(-531.257142857+732.153571429*Xtwid-274.285714286*Xtwid**2.0+29.2571428571*Xtwid**3.0)+Stwid**6.0*(-25.9047619048+9.44761904762*Xtwid)
    gfuncrel=Y1*(kT_moments[0]/m_elec)+Y2*(kT_moments[0]/m_elec)**2.0+Y3*(kT_moments[0]/m_elec)**3.0 #third-order (NOTE: no Y0 term because it is already included in DeltaI_y function above)
    ddgfuncrel=2.0*Y1+6.0*Y2*(kT_moments[0]/m_elec)+24.0*Y3*(kT_moments[0]/m_elec)**2.0
    dddgfuncrel=6.0*Y2+24.0*Y3*(kT_moments[0]/m_elec)
    ddddgfuncrel=24.0*Y3
    return X**4.0 * np.exp(X)/(np.exp(X) - 1.0)**2.0 * 2.0*(kboltz*TCMB)**3.0 / (hplanck*clight)**2.0 * ( tau_ICM * (kT_moments[0]/m_elec) * gfuncrel + tau_ICM/2.0 * ddgfuncrel * (kT_moments[0]/m_elec)**2.0 * w1 + tau_ICM/6.0 * dddgfuncrel * (kT_moments[0]/m_elec)**3.0 * w2 + tau_ICM/24.0 * ddddgfuncrel * (kT_moments[0]/m_elec)**4.0 * w3)

### Foreground components from PlanckX2015 ###
# Here we are in brightness tempearture (as a first pass) with unit K Rayleigh Jeans
# I list the free params as well as the priors that planck used N for a gaussian with mean and std
# I've put the best fit Planck values as defaults

# Thermal Dust
# Params Ad, Bd, Td which are amplitude, index, and temperature
# priors: Ad>0, Bd ~ N(1.55, 0.1), Td ~ N(23, 3)
# oh no wait this is the polarized sed...?!
def thermal_dust(freqs, Ad=163., Bd=1.53, Td=21.):
    f0 = 545.e9    #from planck params
    gam = hplanck/(kboltz*Td)   
    return Ad * (freqs/f0)**(Bd+1.) * (np.exp(gam*f0)-1) / (np.exp(gam*freqs)-1)

# Synchrotron (based on Haslam and GALPROP) 
# Params As, alpha : amplitude and shift parameter
# priors: As>0, alpha>0
# oh no wait this is the polarized sed...?!
def synchrotron(freqs, As=20., alpha=None):
    #fs = need an external template from galprop?
    f0 = 408.e6
    #return As * (f0/freqs)**2. * fs(freqs/alpha) / fs(f0/alpha)
    return As * (f0/freqs)**2.

# Free-free 
# Params EM, Te : emission measure (=integrated square electron density along LOS) and electron temp
# priors: logEM ~ uniform, Te ~ N(7000, 500)
# Ok I think this one is at least in intensity (since free free isn't really polarized)
def freefree(freqs, EM=15.e-3, Te=7000.):
    Tef = (Te * 10**-4)**(-3./2.)
    f9 = freqs / (10**9)
    gff = np.log(np.exp(5.960 - np.sqrt(3.)/np.pi * np.log(f9*Tef)) + np.e)
    tau = 0.05468 * Tef * EM * gff / f9**2
    return (1.-np.exp(-tau))*Te*10**6

# AME
# Params Asd, fp : amplitude and peak frequency
# priors: Asd>0, fp ~ N(19, 3), fp>0
# planck has 2 sets of params here
def ame(freqs, Asd=93., fp=19.):
    #fsd = need external template?
    fp0 = 30.e9
    f01 = 22.8e9
    f02 = 41.e9
    f0 = f01
    return Asd * (f0/freqs)**2 #* fsd(freqs*fp0/fp) / fsd(f0*fp0/fp)
    
# SZ
# params Asz>0
# including this as a check but shouldnt it be the same as the y distortion 
def sz(freqs, ysz=1.4e-6):
    X = hplanck*freqs/(kboltz*TCMB)
    gf = (np.exp(X)-1)**2 / (X*X*np.exp(X))
    return (ysz*10**6)*TCMB * ( (X*np.exp(X)+1.)/(np.exp(X)-1.) - 4.) / gf

# Line emission
# this needs more work. should look in paper about CO emission as spectral distortion


