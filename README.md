# Spectral Distortion Fisher Forecasting

This is a modified version of the fisher forecasting code: [sd_foregrounds](https://github.com/mabitbol/sd_foregrounds). 
If you use this version of the code, please cite the original paper/code [Abitbol+2017](https://arxiv.org/abs/1705.01534) and [Sabyr+2024]()

## Requirements
[numpy](https://numpy.org/),
[scipy](https://scipy.org/),
[bolocalc-space](https://github.com/csierra2/bolocalc-space) (if using bolocalc to compute sensitivity)

## Description:
[example_notebook.ipynb](example_notebook.ipynb) shows how to use this version of the code. 

Additions include: 
  * computing fisher forecasts for some general instrument sensitivity (either computed via bolocalc or an input instantaneous noise in mu-K(rts) units)
  * computing bias on the parameters given some bias in the observed sky monopole
  * easily swapping fiducial values used in the fisher forecast.

## CMB Signals
 * Blackbody CMB
 * Thermal SZ ("y") + relativistic effects
 * mu distortion
 * r-type ("residual", or non-mu/non-y) distortion
 * H and He recombination lines

## Foregrounds
 * Cosmic Infrared Background (dusty galaxies)
 * Galactic dust
 * Galactic synchrotron
 * Extragalactic synchrotron/radio emission
 * Galactic "anomalous microwave emission" (perhaps spinning dust)
 * Galactic free-free emission
 * CO lines
 * tSZ-like distortion in CIB (using a template from [Sabyr, Hill, & Bolliet 2022](https://arxiv.org/abs/2202.02275))


