import numpy as np
import os
from scipy import interpolate
### See components for a better description of the signals.
from other_foregrounds import radiance_to_krj as r2k
TCMB = 2.7255 #Kelvin
hplanck = 6.626070150e-34  # MKS
kboltz = 1.380649e-23  # MKS
clight=299792458.0 #MKS
m_elec = 510.999 #keV!
jy = 1.e26

ndp = np.float64
this_dir=os.path.dirname(os.path.abspath(__file__))

def DeltaI_DeltaT(freqs, DeltaT_amp=1.2e-4):
    X = hplanck*freqs/(kboltz*TCMB)
    return (DeltaT_amp * X**4.0 * np.exp(X)/(np.exp(X) - 1.0)**2.0 * 2.0*(kboltz*TCMB)**3.0 / (hplanck*clight)**2.0 * jy).astype(ndp)

def DeltaI_mu(freqs, mu_amp=2.e-8):
    X = hplanck*freqs/(kboltz*TCMB)
    return (mu_amp * (X / 2.1923 - 1.0)/X * X**4.0 * np.exp(X)/(np.exp(X) - 1.0)**2.0 * 2.0*(kboltz*TCMB)**3.0 / (hplanck*clight)**2.0 * jy).astype(ndp)

def DeltaI_reltSZ_2param_yweight(freqs, y_tot=1.77e-6, kT_yweight=1.245):
    tau = y_tot/kT_yweight * m_elec
    X = hplanck*freqs/(kboltz*TCMB)
    Xtwid = X*np.cosh(0.5*X)/np.sinh(0.5*X)
    Stwid = X/np.sinh(0.5*X)
    Y0=Xtwid-4.0
    Y1=-10.0+23.5*Xtwid-8.4*Xtwid**2+0.7*Xtwid**3+Stwid**2*(-4.2+1.4*Xtwid)
    Y2=-7.5+127.875*Xtwid-173.6*Xtwid**2.0+65.8*Xtwid**3.0-8.8*Xtwid**4.0+0.3666667*Xtwid**5.0+Stwid**2.0*(-86.8+131.6*Xtwid-48.4*Xtwid**2.0+4.7666667*Xtwid**3.0)+Stwid**4.0*(-8.8+3.11666667*Xtwid)
    Y3=7.5+313.125*Xtwid-1419.6*Xtwid**2.0+1425.3*Xtwid**3.0-531.257142857*Xtwid**4.0+86.1357142857*Xtwid**5.0-6.09523809524*Xtwid**6.0+0.15238095238*Xtwid**7.0+Stwid**2.0*(-709.8+2850.6*Xtwid-2921.91428571*Xtwid**2.0+1119.76428571*Xtwid**3.0-173.714285714*Xtwid**4.0+9.14285714286*Xtwid**5.0)+Stwid**4.0*(-531.257142857+732.153571429*Xtwid-274.285714286*Xtwid**2.0+29.2571428571*Xtwid**3.0)+Stwid**6.0*(-25.9047619048+9.44761904762*Xtwid)
    gfuncrel_only=Y1*(kT_yweight/m_elec)+Y2*(kT_yweight/m_elec)**2.0+Y3*(kT_yweight/m_elec)**3.0
    return (X**4.0 * np.exp(X)/(np.exp(X) - 1.0)**2.0 * 2.0*(kboltz*TCMB)**3.0 / (hplanck*clight)**2.0 * (y_tot * Y0 + tau * (kT_yweight/m_elec) * gfuncrel_only) * jy).astype(ndp)


def DeltaI_y(freqs, y_tot=1.77e-6):
    X = hplanck*freqs/(kboltz*TCMB)
    return ((y_tot * (X / np.tanh(X/2.0) - 4.0) * X**4.0 * np.exp(X)/(np.exp(X) - 1.0)**2.0 * 2.0*(kboltz*TCMB)**3.0 / (hplanck*clight)**2.0) * jy).astype(ndp)


def blackbody(nu, DT=1.e-3):
    T = DT*TCMB + TCMB
    X = hplanck * nu / (kboltz * T)
    Xcmb = hplanck * nu / (kboltz * TCMB)
    bbT = 2.0 * hplanck * (nu * nu * nu) / (clight ** 2) * (1.0 / (np.exp(X) - 1.0))
    bbTcmb = 2.0 * hplanck * (nu * nu * nu) / (clight ** 2) * (1.0 / (np.exp(Xcmb) - 1.0))
    return ( (bbT - bbTcmb)*jy ).astype(ndp)

################### additions ###################
####### from components.py #######
def recombination(freqs, scale=1.0):
    rdata = np.loadtxt(this_dir+'/templates/recombination/total_spectrum_f.dat')
    fs = rdata[:,0] * 1e9
    recomb = rdata[:,1]
    template = interpolate.interp1d(np.log10(fs), np.log10(recomb), fill_value=np.log10(1e-30), bounds_error=False)
    return scale * 10.0**template(np.log10(freqs))*jy

####### from https://arxiv.org/pdf/2202.02275.pdf #######
def DeltaI_cib(nu, dIcib_amp=1.0):

    nu0, dI0 = np.loadtxt(this_dir+'/templates/CIB_SZ.txt',unpack=True)

    func = interpolate.interp1d(np.log10(nu0), dI0, kind='cubic', fill_value='extrapolate')
    dInew = func(np.log10(nu))

    return (dIcib_amp*dInew*jy).astype(ndp)

####### from components.py #######
# r-type distortion (first non-mu/non-y eigenmode -- this is only approximately correct for us to use here, but let's stick with it for now)
#freqs in Hz, r_amp dimensionless, DeltaI_r in W/m^2/Hz/sr
def DeltaI_r(freqs, r_amp=1.e-6):
    X = hplanck*freqs/(kboltz*TCMB)
    # first r-distortion eigenmode from Jens (Fig. 4 of 1306.5751)
    rfile = np.loadtxt(this_dir+'/templates/PCA_mode_1.dat')
    Xr = hplanck*rfile[:,0]*1e9/(kboltz*TCMB) #convert from GHz to Hz
    DeltaIr = rfile[:,1]*1e-18*r_amp #conver to W/m^2/Hz/sr
    # linearly interpolate (set to zero above the highest frequency in Jens's file (his lowest frequency is 30 GHz, so things are OK on that end--put in a crazy value so we catch it if needed))
    return np.interp(X, Xr, DeltaIr, left=0.0, right=0.0)*jy

def kDeltaI_r(freqs, r_amp=1.e-6):
    return r2k(freqs, DeltaI_r(freqs, r_amp))*jy

### rel correction to SZ including second-order moment based on https://arxiv.org/pdf/1705.01534.pdf ###
def DeltaI_reltSZ_w1(freqs, y_tot=1.77e-6, kT_yweight=1.282, omega=1.152):
    Yorder=3
    #based on Abitbol+2017, Hill+2015, uses Y functions of Nozawa+2006, Itoh+1998
    yIGM_plus_yreion=1.87e-7
    X=hplanck*freqs/(kboltz*TCMB)
    Xtwid=X*np.cosh(0.5*X)/np.sinh(0.5*X)
    Stwid=X/np.sinh(0.5*X)

    #Y functions
    Y0=Xtwid-4.0
    Y1=-10.0+23.5*Xtwid-8.4*Xtwid**2+0.7*Xtwid**3+Stwid**2*(-4.2+1.4*Xtwid)
    Y2=-7.5+127.875*Xtwid-173.6*Xtwid**2.0+65.8*Xtwid**3.0-8.8*Xtwid**4.0+11.0/30.0*Xtwid**5.0\
    +Stwid**2.0*(-86.8+131.6*Xtwid-48.4*Xtwid**2.0+143.0/30.0*Xtwid**3.0)\
    +Stwid**4.0*(-8.8+187.0/60.0*Xtwid)

    Y3=7.5+313.125*Xtwid-1419.6*Xtwid**2.0+1425.3*Xtwid**3.0-18594.0/35.0*Xtwid**4.0+12059.0/140.0*Xtwid**5.0-128.0/21.0*Xtwid**6.0+16.0/105.0*Xtwid**7.0\
    +Stwid**2.0*(-709.8+2850.6*Xtwid-102267.0/35.0*Xtwid**2.0+156767.0/140.0*Xtwid**3.0-1216.0/7.0*Xtwid**4.0+64.0/7.0*Xtwid**5.0)\
    +Stwid**4.0*(-18594.0/35.0+205003.0/280.0*Xtwid-1920.0/7.0*Xtwid**2.0+1024.0/35.0*Xtwid**3.0)\
    +Stwid**6.0*(-544.0/21.0+992.0/105.0*Xtwid)
    #gfuncrel=Y0+Y1*(kT_yweight/const.m_elec)+Y2*(kT_yweight/const.m_elec)**2.0+Y3*(kT_yweight/const.m_elec)**3.0+Y4*(kT_yweight/const.m_elec)**4.0
    #add different y orders
    orders=np.array([Y0,Y1,Y2,Y3])
    gfuncrel=0.0
    for i in range(Yorder+1):
        gfuncrel+=orders[i]*(kT_yweight/m_elec)**i
        # print(f"added {i}")
    if Yorder==0:
        gfuncrel=Y0

    Trelapprox = y_tot * (gfuncrel+(Y2*(kT_yweight/m_elec)**2.0+3*Y3*(kT_yweight/m_elec)**3.0)*omega) * (TCMB*1e6)
    Planckian = X**4.0*np.exp(X)/(np.exp(X) - 1.0)**2.0
    DeltaIrelapprox = Planckian*Trelapprox / (TCMB*1e6)

    return DeltaIrelapprox*2.0*(kboltz*TCMB)**3.0 / (hplanck*clight)**2.0 * jy
