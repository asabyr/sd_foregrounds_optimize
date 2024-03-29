import numpy as np
import os
file_path=os.path.dirname(os.path.abspath(__file__))
print(file_path)
root_path=file_path.replace('/sd_foregrounds_optimize','/bolocalc-space/')
import sys
sys.path.append(root_path+'analyze-bc')
from gen_bolos import GenBolos
import pandas as pd

bc_fp = root_path+'calcBolos.py'
exp_fp = root_path+'Experiments/specter_v1/'
ndp=np.float64
Tcmb=2.7255 #SI
kb=1.380649e-23 #SI
hplanck=6.62607015e-34 #SI
clight=299792458. #SI

def muKtoJypersr(muK,f):
    """
    convert from muK units to Jy/sr based on https://arxiv.org/pdf/1303.5070.pdf & https://arxiv.org/pdf/2010.16405.pdf
    i.e. by taking a derivative of the Planck function

    args:
    muK: sensitivity [muK]
    f: frequency [Hz]

    output:
    sensitivity [Jy/sr]
    """
    x=hplanck*f/(kb*Tcmb)
    factor=2.*hplanck**2/(clight**2*kb*(Tcmb*1.e6)*Tcmb)*1.e26
    Jypersr_muK=factor*(f)**4*np.exp(x)/(np.exp(x)-1.)**2

    return muK*Jypersr_muK

def getnoise_raw(path, prefix,bands, hemt_amps = True, hemt_freq = 10):
    """
    calculate instantenous sensitivity for a set of frequency bands using bolocalc calculator
    (saves the sensitivity in path+prefix+"_raw_sensitivity.txt")

    args:
    path: directory path to the project folder
    prefix: file prefix for bolocalc calculator
    bands: frequency bands [GHz]

    output:
    frequencies [GHz]
    sensitivity [muK-rtSec]

    """

    bolos = GenBolos(bc_fp=bc_fp, exp_fp=exp_fp, band_edges=bands, file_prefix=prefix, hemt_amps=hemt_amps, hemt_freq=hemt_freq)
    sensitivity_dict = bolos.calc_bolos()
    noise_df = pd.DataFrame(sensitivity_dict).T

    freqs=np.array(noise_df['Center Frequency'], dtype=ndp) #GHz
    raw_noise= np.array(noise_df['Detector NET_CMB'], dtype=ndp)#m


    raw_sens_file=path+prefix+"_raw_sensitivity.txt"
    np.savetxt(raw_sens_file,np.column_stack([freqs,raw_noise]))


    return freqs, raw_noise

def getnoise_nominal(prefix, bands, dets, hemt_amps = True, hemt_freq = 10, precompute=False):
    """
    calculate nominal sensitivity for a set of frequency bands using bolocalc calculator
    (assumes 6 months of nominal integration, 100% of the sky)

    args:
    prefix: file prefix for bolocalc calculator
    bands: frequency bands [GHz]
    dets: detector counts
    precompute: sensitivity file (if precalculated, otherwise False)

    output:
    frequencies [Hz]
    sensitivity [Jy/sr]

    """
    nominal_exposure=6. #months
    exposure_sec=365.25*24.*3600.*nominal_exposure/12. #seconds

    if precompute:

        noise=np.loadtxt(precompute, dtype=ndp)
        freqs=noise[:,0]*1.e9 #Hz
        noise_muK_rtsec=noise[:,1] #muK-rtSec

    else:

        bolos = GenBolos(bc_fp=bc_fp, exp_fp=exp_fp, band_edges=bands, file_prefix=prefix, hemt_amps=hemt_amps, hemt_freq=hemt_freq)
        sensitivity_dict = bolos.calc_bolos()
        noise_df = pd.DataFrame(sensitivity_dict).T
        freqs=np.array(noise_df['Center Frequency'], dtype=ndp)*1.e9 #Hz
        noise_muK_rtsec= np.array(noise_df['Detector NET_CMB'], dtype=ndp)#muK-rtSec


    noise_muK_deg=noise_muK_rtsec/np.sqrt(exposure_sec) #muK
    noise_Jysr_deg = muKtoJypersr(noise_muK_deg,freqs) #Jypersr
    noise=noise_Jysr_deg/np.sqrt(dets)

    return freqs, noise
