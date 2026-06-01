"""
Advanced utilities for the simulation module

Contains utility functions that are used in the simulation module, 
but they are not intended for direct end user interaction. 
"""


import numpy as np
from .materials import get_n
from .core import fresnel
from .matrices import numeric as npM


def analyze_reflection(alpha, theta, material, E_eV, vector = (1, 1, 0, 0), source = "auto"):
    """
    Returns the reflectivity, degree of linear polarization and degree of circular polarization 
    of a given reflective setup.

    Parameters
    ----------
    alpha : float
        Orientation of the plane of incidence relative to the vertical direction
    theta : float or list of floats
        Angle of incidence. Single float denotes a single mirror. List or tuple of floats denotes 
        a multiple-reflection polarizer in coplanar geometry of incidence planes of all mirrors.
    material : str
        Reflective surface material.
    E_eV : float
        Energy of the incident beam in electronvolts.
    vector : tuple or list or numpy.ndarray, optional
        incident beam Stokes vector. Default is (1, 1, 0, 0) (completely p-polarized)
    source : str, optional
        Preffered experimental data source for the refractive index from RefractiveIndex database.
        Default is 'auto', which will select the best aviable source for the given beam energy.

    Returns
    -------
    tuple of numpy.float64
        reflectivity, degree of linear polarization, degree of circular polarization

    Examples
    --------
    >>> analyze_reflection(alpha = 45, theta =  72, material = "Au", E_eV = 5) 
    (np.float64(0.4613053743133022), np.float64(0.6270367072506624), np.float64(-0.7789897096626162))

    >>> R, PC, PL = analyze_reflection(alpha = 60, 
    ...                                theta = (50, 70, 70, 50), 
    ...                                material = "Mo", 
    ...                                E_eV = 15.3, 
    ...                                vector = (1, 0.6, 0, 0.8), 
    ...                                source = "Windt"
    ... )
    >>> PC
    np.float64(0.9910009074562234)
    """
    n = get_n(material, E_eV, source, cite = False)
    rs, rp = fresnel(n, theta)

    S_in = np.asarray(vector)
    S_out = npM.reflection(rs, rp, npM.rotation(alpha, S_in))

    R = S_out[0]/S_in[0]
    PL = np.sqrt(S_out[1]**2 + S_out[2]**2)/S_out[0]
    PC = S_out[3]/S_out[0]

    return R, PL, PC


def fom_function(geometry, material, E_eV, incidence, mirrors, vector, figure_of_merit, source):
    """"
    Calculates Figure of merit (fom) as a function of reflective element geometry.

    The main puropose of this function is its implementation into the optimize_geometry() function,
    where the figure of merit is maximized by finding the optimal geometrical parameters.
    The geometry can either be only the element orientation (alpha), or both the orientation 
    and the incidence angle (theta). See 'incidence' parameter for details. 

    Parameters
    ----------
    geometry : float or numpy.ndarray
        geometrical parameters. For 'variable' incidence: numpyp.array(alpha, theta) [°]. 
        For 'fixed' incidence: alpha [°]. This is the optimized parameter in the optimize_geometry()
        function. 
    material : str
        Mirror material chemical symbol (e.g., 'Au').
    E_eV : float
        Energy of the incident beam in electronvolts.
    incidence : float or list of floats or str
        For fixed-incidence geometry: angle of incidence in degrees, can also be a tuple 
            of different incidence angles inside a coplanar polarizer.
        For variable-incidence geometry: 'variable'
    mirrors : int
        number of mirrors of a multiple-reflection polarizer. Only relevant for optimization
        of incidence angle of a multiple-reflection coplanar polarizer. In this case, 
        an identical incidence angle is assumed for all mirrors.
    vector : tuple or list or numpy.ndarray
        incident beam Stokes vector
    figure of merit : str
        chosen figure of merit to be returned by the function.
        'PC2' for P_C**2 (square of degree of circular polarization)
        'RPC2' for R*P_C**2 (reflectivity * square of degree of circular polarization)
    source : str
        Preffered experimental data source for the refractive index from RefractiveIndex database.
        Default is 'auto', which will select the best aviable source for the given beam energy.

    Returns
    -------
    np.float :
        NEGATIVE figure of merit. This makes it convenient for a subsequent minimalization. 

    Examples
    --------
    >>> fom_function(geometry = np.array([42]), material = "Au", E_eV = 60, incidence = 60, mirrors = 1, vector = (1, 1, 0, 0), figure_of_merit = "PC2", source = "auto") 
    np.float64(-0.23770434603854615)
    """
    alpha = geometry[0]
        
    if incidence == "variable":
        if mirrors > 1:
            theta = []
            for i in range(0, mirrors):
                theta.append(geometry[1])
        else:
            theta = geometry[1]
    else:
        theta = incidence

    R, _, PC = analyze_reflection(alpha, theta, material, E_eV, vector, source)

    if figure_of_merit == "RPC2":
        return -R*PC**2
    elif figure_of_merit == "PC2":
        return -PC**2
    else:
        raise ValueError(f"invalid figure of merit: {figure_of_merit}")