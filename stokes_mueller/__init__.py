"""
Numerical simulations of the polarization state. 

This package enables modelling of the light polarization state evolution
through an experimental setup. It utilizes the combined Stokes-Mueller
formalism for linear optical phenomena. Its main focus are simulations
of reflective optical elements (metallic mirrors) in the EUV spectral 
region.

Conventions
-----------
angle:
    positive angle of rotation: counter-clockwise
    negative angle of rotation: clockwise
    incidence angle: 
        measured from the normal direction of the reflection surface
    mirror orientation: 
        alpha = 0 for a vertical plane of incidence
    s- and p-state orientation in the laboratory frame of reference: 
        s-state is vertical and p-state is horizontal
helicity:
    helicity by IUPAC: for an observer looking towards the source.
    Left-Circularly Polarization (LCP): counter-clockwise rotation
    Right-Circularly Polarization (RCP): clockwise rotation
Stokes Parameters sign formalism:
    S1:
        positive: Degree of p-state polarization
        negative: Degree of s-state polarization
    S2: 
        positive: Degree of LP in a plane rotated +45° to the p-plane
        negative: Degree of LP in a plane rotated -45° to the p-plane
    S3: 
        positive: Degree of RCP
        negative: Degree of LCP

Main API
--------
plot_setup : Complete polarization state simulation with visualization.
plot_spectrum : Reflective element spectrum-wide characterization.
plot_optimized_geometry : Optical setup optimization with visualization.
fit_data : Experimental data regression.

Dependencies
------------
numpy
sympy
scipy
refractiveindex
matplotlib
"""

from .simulation import plot_spectrum, optimize_geometry, plot_optimized_geometry, plot_setup
from .analysis import get_data, subtract, rectify, stabilize, integrate, fit_data
from .visualization import plot_data

__all__ = [
    "plot_spectrum",
    "optimize_geometry",
    "plot_optimized_geometry",
    "plot_setup",
    "get_data",
    "subtract",
    "rectify",
    "stabilize",
    "integrate",
    "fit_data",
    "plot_data"
]