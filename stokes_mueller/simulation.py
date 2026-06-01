"""
Simulations of polarization state utilizing the combined Stokes-Mueller formalism

This module enables simulations of beam propagation through optical setups, 
as well as advanced characterization of optical elements and geometry optimization.
Some functions in this module both run the simulation and visualize the results; 
the plotting feature will soon be removed from this module and moved exclusively 
to the visualization module. Purpose of this module will then be only
to run the simulations and return simulation results. 
"""


import numpy as np
import sympy as sp
from scipy.optimize import minimize, shgo
from .materials import get_n
from .core import fresnel, DimensionError
from .matrices import numeric as npM
from .matrices import symbolic as spM
from .utilities import analyze_reflection, fom_function
import matplotlib.pyplot as plt

# Configure matplotlib to use Times New Roman font
plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = ["Times New Roman"]
plt.rcParams["mathtext.fontset"] = "stix"


def plot_spectrum(material = "Au", theta = (70, 50, 70), show = "fractional polarization", min = 5, max = 80, step = 0.1, source = "auto", formalism = "Schmising", preset = "none"):
    """
    Plots the spectrum of optical properties for the chosen material and incidence. 
    
    This function is inspired by a figure in paper by Koide et al. from 1991 (Fig. 3)

    Parameters
    ----------
    material : str
        Mirror material chemical symbol (e.g., 'Au').
    theta : float or list of floats
        Angle of incidence in degrees. Single float denotes a single mirror. List or tuple of floats denotes 
        a multiple-reflection polarizer in coplanar geometry of incidence planes of all mirrors.
    show : str or list of str
        values(s) to be plotted. 
        Options are: 
        'polarization': plots fractional polarization = (1 - rho**2)/(1 + rho**2) 
            and phase shift (delta) = delta_p - delta_s; 
        'fractional polarization': plots (1 - rho**2)/(1 + rho**2); 
        'phase shift': plots delta = delta_p - delta_s; 
        'transmittance': plots T_s = r_s**2 and T_p = r_p**2; 
        'reflection ratio': plots rho = r_p/r_s
    min : float 
        minimum eV of the spectrum
    max : float 
        maximum eV of the spectrum
    step: float 
        step between datapoints in eV
    source : str, optional
        Preffered experimental data source for the refractive index from RefractiveIndex database.
        Default is 'auto', which will select the best available source for the given beam energy.
    formalism : str, optional
        Sign convention for helicity. 'Schmising' (by default) or 'Koide'
    preset: str, optional
        bypass all parameters and reconstrunct calculations from literature. "Koide 3a" or "Koide 3b".
        Default is "none".

    Returns
    -------
    numpy.ndarray
        Array of shape=(rows, cols), where 'rows' is either 2 or 3, depending on the number 
        of calculated metrics, and 'cols' is the number of spectrum datapoints in the eV range.
        First row of the array is the sequence of energies in eV.
        Second (and/or) third row of the array are the calculated metrics at the corresponding eV.
    
    Examples
    --------
    >>> plot_spectrum(material = "Au", theta = (70, 50, 70), show = "fractional polarization", min = 5, max = 80, step = 0.1)
    Refractive index of Au - data source: Hagemann-2
    array([[ 5.        ,  5.1       ,  5.2       , ..., 79.7       ,
            79.8       , 79.9       ],
           [ 0.94757407,  0.94979798,  0.95187137, ...,  0.85940061,
             0.85942953,  0.85945821]], shape=(2, 750))

    >>> plot_spectrum(material = "Au", theta = (70, 50, 70), show = "transmittance", min = 5, max = 80, step = 0.1)                       
    Refractive index of Au - data source: Hagemann-2
    array([[5.00000000e+00, 5.10000000e+00, 5.20000000e+00, ...,
            7.97000000e+01, 7.98000000e+01, 7.99000000e+01],
          [2.47437635e-01, 2.39335656e-01, 2.31522687e-01, ...,
           9.17369506e-03, 9.14328481e-03, 9.11310741e-03],
          [6.66067002e-03, 6.16224576e-03, 5.70881362e-03, ...,
           6.93672961e-04, 6.91220525e-04, 6.88787945e-04]], shape=(3, 750))

    >>> plot_spectrum(preset = "Koide 3a")
    Refractive index of Au - data source: Hagemann-2
    array([[  5.        ,   5.1       ,   5.2       , ...,  79.7       ,
             79.8       ,  79.9       ],
           [155.72855631, 153.24015953, 150.82187968, ...,  36.50183748,
            36.45889784,  36.41602542]], shape=(2, 750))  
    """
    if preset == "Koide 3a":
        material = "Au"
        theta = (80, 70, 80)
        show = "polarization"
        source = "Hagemann-2"
        formalism = "Koide"
    elif preset == "Koide 3b":
        material = "Au"
        theta = (70, 50, 70)
        show = "transmittance"
        source = "Hagemann-2"
        formalism = "Koide"

    E_eV = np.arange(min, max, step)
    # creates n as an array of metrics across the spectrum range
    n = get_n(material, E_eV, source)
    # also an array
    rs, rp = fresnel(n, theta)
    
    if show == "fractional polarization" or show == "polarization":
        rho = abs(rp/rs)
        fractional_polarization = (1 - rho**2)/(1 + rho**2)
        plt.plot(E_eV, 100*fractional_polarization, color = "blue", label = "fractional polarization")
        plt.ylabel("fractional polarization [%]")
        plt.ylim(0, 100)        
        data = np.array([E_eV, fractional_polarization])

    if show == "phase shift" or show == "polarization":
        delta = np.angle(rp/rs)
        # Koide paper uses inverted helicity convention
        if formalism == "Koide": 
            delta = -delta

        delta_deg = np.rad2deg(delta)
        # ensure graph continuity (np. angle returns only in range -180..180)
        for i in range(0, len(delta_deg)): 
            # if there is a sudden drop of more than 180 deg
            if delta_deg[i - 1] > 170 and delta_deg[i] < (delta_deg[i - 1] - 180): 
                delta_deg[i] += 360
            # if there is a sudden jump of more than 180 deg
            if delta_deg[i - 1] < -170 and delta_deg[i] > (delta_deg[i - 1] + 180): 
                delta_deg[i] -= 360
        
        plt.plot(E_eV, delta_deg, color = "red", label = "phase shift")
        plt.ylabel("phase shift [°]")
        data = np.array([E_eV, delta_deg])

    elif show == "transmittance":
        Ts = np.abs(rs)**2
        Tp = np.abs(rp)**2
        plt.plot(E_eV, Ts, color = "blue", label = "Ts")
        plt.plot(E_eV, Tp, color = "red", label = "Tp")
        plt.yscale("log")
        plt.ylim(10**(-4), 1)
        data = np.array([E_eV, Ts, Tp])

    elif show == "reflection ratio":
        rho = np.abs(rp/rs)
        plt.plot(E_eV, rho, color = "blue", label = "rho")
        data = np.array([E_eV, rho])

    plt.title("Spectrum")
    plt.xlabel("energy [eV]")
    plt.xlim(0, 80)
    plt.legend()
    plt.show()

    return(data)


def optimize_geometry(material = "Mo", E_eV = 70, x0 = np.array([45, 78]), incidence = "variable", mirrors = 1, vector = (1, 1, 0, 0), figure_of_merit = "RPC2", optimization = "local", source = "auto", analyze = False):
    """
    Maximizes the figure of merit by optimizing the reflective element (mirror or polarizer) geometry 
    
    The geometry can either be only the element orientation (alpha), or both the orientation 
    and the incidence angle (theta). See 'incidence' parameter for details. Geometry of coplanar 
    multi-reflection polarizers may also be optimized. See 'mirrors' parameter for details. 
    Multiple-reflection polarizers with internal reflections occuring under different incidence angles 
    may be optimized only in regards to alpha (theta must be fixed). 

    Parameters
    ----------
    material : str
        Mirror material chemical symbol (e.g., 'Au').
    E_eV : float
        Energy of the incident beam in electronvolts.
    x0 : numpy.ndarray
        Initial geometry guess in form of np.array([alpha]) for fixed incidence
        or np.array([alpha, theta]) for variable incidence. 
    incidence : float or tuple of floats or str
        For fixed-incidence geometry: angle of incidence in degrees, can also be a tuple 
            of different incidence angles inside a coplanar polarizer.
        For variable-incidence geometry: 'variable'.
        In case of fixed geometry, only element orientation (alpha) will be the optimized parameter.
        If set to 'variable", both orientation (alpha) and incidence (theta) 
        will be the optimized parameters.
    mirrors : int
        number of mirrors of a multiple-reflection coplanar polarizer. Only relevant for optimization
        of incidence angle of a multiple-reflection coplanar polarizer. In this case, 
        an identical incidence angle is assumed for all internal reflections.
    vector : array-like
        incident beam Stokes vector
    figure of merit : str
        chosen figure of merit to be maximized.
        'PC2' for P_C**2 (square of degree of circular polarization)
        'RPC2' for R*P_C**2 (reflectivity * square of degree of circular polarization)
    optimization : str
        Type of algorithm used in optimization. 'local' or 'global'
    source : str, optional
        Preffered experimental data source for the refractive index from RefractiveIndex database.
        Default is 'auto', which will select the best available source for the given beam energy.
    analyze : bool, optional
        If True, the function returns detailed analysis of the optimization process and its result, 
        along with the vizualization of the found maximum location. Default is False.

    Returns
    -------
    list of numpy.float64
        Results of optimization and relevant metrics
        For fixed incidence optimization:
            [
            figure of merit at optimized geometry, 
            reflectivity at optimized geometry, 
            degree of circular polarization at optimized geometry, 
            optimized element orientation in degrees
            ]
        For variable incidence optimization:
            [
            figure of merit at optimized geometry, 
            reflectivity at optimized geometry, 
            degree of circular polarization at optimized geometry, 
            optimized element orientation in degrees,
            optimized incidence in degrees
            ]

    See Also
    --------
    plot_optimized_geometry()
            
    Examples
    --------
    >>> # variable incidence single mirror optimization
    >>> fom_max, R_opt, PC_opt, alpha_opt, theta_opt = optimize_geometry(material = "Mo", E_eV = 70, x0 = np.array([45, 78]), incidence = "variable", mirrors = 1, vector = (1, 1, 0, 0), figure_of_merit = "RPC2")
    >>> fom_max
    np.float64(0.04394809355990873)
    >>> R_opt
    np.float64(0.577308921522008)
    >>> PC_opt
    np.float64(-0.275909011787379)
    >>> alpha_opt
    np.float64(44.02921022239664)
    >>> theta_opt
    np.float64(70.48464887397228) 

    >>> # variable incidence 4-mirror coplanar polarizer optimization
    >>> results = optimize_geometry(material = "Au", E_eV = 40, x0 = np.array([40, 70]), incidence = "variable", mirrors = 4, vector = (1, -1, 0, 0), figure_of_merit = "PC2")              
    >>> figure_of_merit = results[0]
    >>> optimized_alpha = results[3]
    >>> optimized_theta = results[4]
    >>> figure_of_merit
    np.float64(0.9999999999944476)
    >>> optimized_alpha
    np.float64(64.49529669696634)
    >>> optimized_theta
    np.float64(75.51477917537564)

    >>> # fixed incidence 4-mirror coplanar polarizer optimization
    >>> fixed_incidence_results = optimize_geometry(material = "Au", E_eV = 40, x0 = np.array([65]), incidence = 70, mirrors = 4, vector = (1, -1, 0, 0), figure_of_merit = "PC2")          
    >>> optimized_alpha = fixed_incidence_results[3]
    >>> optimized_alpha
    np.float64(52.66987609863281)
    """
    args = (material, E_eV, incidence, mirrors, vector, figure_of_merit, source)
    
    if incidence == "variable":
        if optimization == "local":
            result = minimize(fom_function, x0, args, bounds = [(0, 90), (0, 90)], method = "Nelder-Mead")
        elif optimization == "global":
            result = shgo(fom_function, [(0, 90), (0, 90)], args)
        else: 
            raise ValueError("Invalid 'optimization' argument. Possible values are: 'local' or 'global'.")
        
        amax = result.x[0]
        tmax = result.x[1]
        fmax = -result.fun

        char_max = analyze_reflection(amax, tmax, material, E_eV, vector, source)
        Rmax = char_max[0]
        PCmax = char_max[2]
    
        if analyze == True:
            # just to ensure citation once
            get_n(material, E_eV, source, cite = True)
            print("Cause of algorithm termination:", result.message)
            print("figure of merit =", fmax, "\nalpha =", amax, "\ntheta =", tmax)
            print("Preparing the plot...")
            x = []
            y = []
            z = []
            i = 0

            for a in np.arange(amax - 0.05, amax + 0.0499, 0.001):
                x.append(np.array([a]))
                z.append([-fom_function(np.array([a, tmax - 0.05]), material, E_eV, incidence, mirrors, vector, figure_of_merit, source)])
                for t in np.arange(tmax - 0.049, tmax + 0.0499, 0.001):
                    z[i].append(-fom_function(np.array([a, t]), material, E_eV, incidence, mirrors, vector, figure_of_merit, source))
                i += 1
            for t in np.arange(tmax - 0.05, tmax + 0.05, 0.001):
                y.append(np.array([t]))

            arg_max = np.argmax(z)
            scan_amax = arg_max//100 # index of row (1st array layer)
            scan_tmax = arg_max%100 # index of column (2nd array layer)
            if np.round(x[scan_amax], 10) == np.round(amax, 10) and np.round(y[scan_tmax], 10) == np.round(tmax, 10):
                print("The optimization result sits in the scanned maximum.")
            else:
                delta = np.sqrt((x[scan_amax] - amax)**2 + (y[scan_tmax] - tmax)**2)
                print(f"Optimization result offset from scanned maximum: {delta[0]}°")

        return([fmax, Rmax, PCmax, amax, tmax])
        
    else:
        if optimization == "local":
            result = minimize(fom_function, x0, args, method = "Nelder-Mead")
        elif optimization == "global":
            result = shgo(fom_function, [(0, 90)], args)
        else:
            raise ValueError(f"Invalid 'optimization' argument: {optimization}. Valid arguments are: 'local' or 'global'.")
        amax = result.x[0]
        fmax = -result.fun

        char_max = analyze_reflection(amax, incidence, material, E_eV, vector, source)
        Rmax = char_max[0]
        PCmax = char_max[2]

        if analyze == True:
            get_n(material, E_eV, source, cite = True) # prints refraction index source citation
            print("Cause of algorithm termination:", result.message)
            print("figure of merit =", fmax, "\nalpha =", amax)
            print("Preparing the plot...")
            x = []
            y = []
            for a in np.arange(amax - 0.05, amax + 0.05, 0.001):
                x.append(np.array(a))
                y.append(-fom_function(np.array([a]), material, E_eV, incidence, mirrors, vector, figure_of_merit, source))

            arg_max = (np.argmax(y)) # index of the scanned maimum value on x-axis       
            if np.round(x[arg_max], 10) == np.round(amax, 10):
                print("The optimization result sits in the scanned maximum.")
            else:
                delta = abs(x[arg_max] - amax)
                print(f"Optimization result offset from scanned maximum:{delta}°")

        return([fmax, Rmax, PCmax, amax])


def plot_optimized_geometry(material = "Mo", incidence = "variable", show = "characteristics", mirrors = 4, min = 35, max = 90, step = 0.1, vector = (1, 1, 0, 0), figure_of_merit = "RPC2", optimization = "local", source = "auto"):
    """
    Plots the optimized geometry of a reflective element (mirror, coplanar polarizer)
    as a function of the beam energy.

    The geometry can either be only the element orientation (alpha), or both the orientation 
    and the incidence angle (theta). See 'incidence' parameter for details. Geometry of coplanar 
    multi-reflection polarizers may also be optimized. See 'mirrors' parameter for details. 
    Multiple-reflection polarizers with internal reflections occuring under different incidence angles 
    may be optimized only in regards to alpha (theta must be fixed). 

    If no arguments are passed on, plot_optimized_geometry() by default attempts a reconstruction 
    of the Fig. 1a from the paper by Schmising et al. under https://doi.org/10.14279/depositonce-10039

    Parameters
    ----------
    material : str
        Mirror material chemical symbol (e.g., 'Au').
    incidence : float or tuple of floats or str
        For fixed-incidence geometry: angle of incidence in degrees, can also be a tuple 
            of different incidence angles inside a coplanar polarizer.
        For variable-incidence geometry: 'variable'.
        In case of fixed geometry, only element orientation (alpha) will be the optimized parameter.
        If set to 'variable", both orientation (alpha) and incidence (theta) 
        will be the optimized parameters.
    show : str
        Aspect of the optimization result to be plotted as a function of beam energy
        'characteristics' plots figure of merit, reflectivity and degree of circular polarization
        'geometry' plots optimized geometrical parameters
            element orientation (alpha) for fixed incidence (see incidence parameter)
            alpha and incidence (theta) for variable incidence (see incidence parameter)
    mirrors : int
        number of mirrors of a multiple-reflection coplanar polarizer. Only relevant for optimization
        of incidence angle of a multiple-reflection coplanar polarizer. In this case, 
        an identical incidence angle is assumed for all internal reflections.
    min : float
        lower limit of the spectrum in eV
    max : float
        upper limit of the spectrum in eV
    step : float
        step between datapoints of the spectrum in eV
    vector : array-like
        incident beam Stokes vector
    figure of merit : str
        chosen figure of merit to be maximized.
        'PC2' for P_C**2 (square of degree of circular polarization)
        'RPC2' for R*P_C**2 (reflectivity * square of degree of circular polarization)
    optimization : str
        Type of algorithm used in optimization. 'local' or 'global'
    source : str, optional
        Preffered experimental data source for the refractive index from RefractiveIndex database.
        Default is 'auto', which will select the best available source for the given beam energy.
    
    material: mirror material
    incidence: optional, 'variable' by default. Specifies theta, which is then excluded from the optimization. Can also be a tuple of different angles inside a coplanar polarizer.
    min: minimum eV of the spectrum
    max: maximum eV of the spectrum
    step: step between datapoints in eV
    source: preffered source for the refraction index data
    preset: reconstruction of paper calculations

    See Also
    --------
    optimize_geometry()

    Examples
    --------
    >>> # reconstruction attempt of Fig. 1a from Schmising2017
    >>> plot_optimized_geometry()
    Refractive index of Mo - data source: Windt

    >>> # reconstruction attempt of Fig. 1b from Schmising2017
    >>> plot_optimized_geometry(show = "geometry")
    Refractive index of Mo - data source: Windt
    """

    E_eV = np.arange(min, max, step)
    # prints refraction index citation
    get_n(material, E_eV, source, cite = True) 
    i = 0
    fom = []
    R = []
    PC = []
    alpha = []

    if incidence == "variable":
        theta = []
        
        for epsilon in np.arange(min, max, step):
            if i == 0:
                # initial guess, works the best so far
                x0 = np.array([45, 75]) 
            else:
                x0 = np.array([alpha[i-1], theta[i-1]]) # all further guesses are based on the previous datapoint
            opt = optimize_geometry(material, epsilon, x0, incidence, mirrors, vector, figure_of_merit, optimization, source, analyze = False)
            fom.append(opt[0])
            R.append(opt[1])
            PC.append(opt[2])
            alpha.append(opt[3])
            theta.append(opt[4])
            i += 1
    else:
        for epsilon in np.arange(min, max, step):
            if i == 0:
                x0 = np.array([45]) # initial guess
            else:
                x0 = np.array([alpha[i-1]]) # further guesses based on previous datapoints
            opt = optimize_geometry(material, epsilon, x0, incidence, mirrors, vector, figure_of_merit, optimization, source, analyze = False)
            fom.append(opt[0])
            R.append(opt[1])
            PC.append(opt[2])
            alpha.append(opt[3])
            i += 1
    
    if show == "characteristics":
        if figure_of_merit == "RPC2":
            plt.plot(E_eV, fom, color = "blue", linestyle = "--", label = "R*P_C**2")
        elif figure_of_merit == "PC2":
            plt.plot(E_eV, fom, color = "blue", linestyle = "--", label = "P_C**2")
        else: 
            raise ValueError(f"Invalid figure of merit: {figure_of_merit}. Valid arguments are: 'RPC2' or 'PC2'.")
        plt.plot(E_eV, R, color = "green", linestyle = "-.", label = "R")
        plt.plot(E_eV, PC, color = "red", label = "P_C")
        plt.ylabel("figure value")
    elif show == "geometry":
        plt.plot(E_eV, alpha, color = "red", label = "alpha")        
        if incidence == "variable":
            plt.plot(E_eV, theta, color = "blue", label = "theta")
        plt.ylabel("[°]")
    else:
        raise ValueError(f"Invalid 'show' argument: {show}. Possible values are: 'characteristics' (for figure of merit, R and P_C plot) or 'geometry' (for alpha and theta plot).')")
    plt.title('Spectrum optimization result')
    plt.xlabel('energy [eV]')
    plt.legend()
    plt.show()


def plot_setup(vector, E_eV, element, incidence, orientation, show = "all", source = "auto", simplify_expressions = False, show_title = True, return_data = True, export_data = False):
    """
    Plot simualted data for an optical system as a function of setup geometry. 

    The optical setup may contain an arbitrary number of transmissive or optical elements,
    or their combination. The setup is defined by three parallel lists: see parameters 
    'element', 'incidence', and 'orientation'. The geometrical parameters are orientations 
    of up to 2 elements in the setup. 

    Parameters
    ----------
    vector : array-like
        incident beam Stokes vector
    E_eV : float
        Energy of the incident beam in electronvolts.
    element : list or tuple of str
        sequence of elements IDs':
            'lp' for linear polarizer
            'qwp' for a quarter-wave plate
            'hwp' for a half-wave plate
            material of a mirror - chemical fromula (e.g., 'Au' or 'MgF2')
    incidence : list or tuple of floats
        Incidence angle onto a mirror in degrees. Relevant only for reflective elements 
        (may be set to arbitary number for transmissive elements).  
    orientation : list or tuple of floats or np.ndarrays
        List or tuple of elements orientations. The to-be-parametrized element orientations 
        must be passed on as a numpy.arange (up to 2 parametrized orientation angles 
        in one setup). The contents of this range specify all the element orientations 
        the data will be evaluated for. 
    show: str or tuple of str, optional
        Specifies which data to plot. Options are:
            'intensity' for resulting beam intensity
            'PC' for resulting beam degree of circular polarization
            'PL' for resulting beam degree of linear polarization
            'RPC2' for resulting beam R*PC**2, where R is the overall reflectivity
                and PC is degree of circular polarization
            'ellipticity' for ellipticity epsilon
            'all' plots all abovementioned values
            'stokes' shows Stokes parameters S1, S2 and S3
            or any combination in a tuple.             
        The plots will be shown in separate windows. Default is 'all'
    source : str, optional
        Preffered experimental data source for the refractive index from RefractiveIndex database.
        Default is 'auto', which will select the best available source for the given beam energy.
    simplify_expressions : bool, optional
        Perform SymPy simplification of symbolic expressions of the different metrics.
        Suitable only for smaller and simple setups, gets very computationally demanding 
        for larger optical setups. Default is False
    show_title : bool, optional
        Show the title of the plot. May be set to False when the title is undesirable, 
        e.g., when generating figures for a thesis. Default is True
    return_data : bool, optional
        Option to return the data in a form of array (for 1 parameter) or nested array 
        (for 2 parameters). Default is True
    export_data : bool, optional
        Option to export data into a 'output.csv' file in the current directory.
        Relevant only for a single parameter. If the file is not present,
        it will be automatically created. The structure of the exported data inside the file 
        is the same as for the returned data (see Returns). 

    Returns
    -------
    numpy.ndarray
        Array of the simulation results. Returns only if the number of parameters is equal to 1. 
        First column is the parametrized orientation angle, next columns are values, which were
        specified with the 'show' parameter.

    Notes
    -----
    Uses SymPy syntax, enabling more flexible (but slower) calculations and symbolic expressions. 

    Examples
    --------
    >>> # 1D plot: demonstration of Malus' law (the output is not shown)
    >>> plot_setup(vector = (1, 0, 0, 0), E_eV = 1.5, element = ('lp', 'lp'), incidence = (0, 0), orientation = (0, np.arange(0, 360)), show = 'intensity')

    >>> # 2D plot: reflective setup containing two mirrors with avriable orientation (the output is not shown)
    >>> plot_setup(vector = (1, 1/3, 2/3, 2/3), E_eV = 20.15, element = ('Au', 'Mo'), incidence = (45, 70), orientation = (np.arange(0, 360), np.arange(0, 360)))
    """

    if isinstance(orientation, (list, tuple, np.ndarray)):
        N = len(element)
        if N != len(orientation):
            raise DimensionError(f"Inconsistent input: orientations have dimension of {len(orientation)}, but there are {len(element)} elements.")
        # list is mutable (unlike tuple) - I will later replace some orientations with symbols
        orientation = list(orientation)      
    else:
        orientation = [orientation]
        element = [element]
        incidence = [incidence]
        N = 1

    if len(vector) < 4 or len(vector) > 5:
        raise DimensionError(f"Stokes vector has wrong size: {len(vector)}")
    
    S = sp.Matrix(vector)
    
    # initializing the variable parameters
    a, b = sp.symbols("alpha, beta")  # [°]

    # This will be later used to determine whether a 2D or 3D plot is necessary
    parameters = []    

    # step-by-step passage through the whole setup
    for i in range(0, N):

        # when at the analyzer
        if isinstance(orientation[i], (np.ndarray, range, list, tuple)):
            
            # ndarray is more suitable for future operations
            if not isinstance(orientation[i], np.ndarray):
                orientation[i] = np.array(orientation[i])
                       
            if len(parameters) == 0:
                x = orientation[i]
                orientation[i] = a
            elif len(parameters) == 1:
                y = orientation[i]
                orientation[i] = b
            else:
                raise ValueError("Too many analyzers: maximum of two parameters may be given.")
            parameters.append(orientation[i])

        # transform the Stokes vector according to the current element
        match element[i]:
            case "lp":
                S = spM.rotation(-orientation[i], spM.linear_polarizer(spM.rotation(orientation[i], S)))
            case "qwp":
                S = spM.rotation(-orientation[i], spM.retarder(90, spM.rotation(orientation[i], S)))
            case "hwp":
                S = spM.rotation(-orientation[i], spM.retarder(180, spM.rotation(orientation[i], S)))
            case _: # assume a mirror
                n = get_n(element[i], E_eV, source)
                rs, rp = fresnel(n, incidence[i])
                S = spM.rotation(-orientation[i], spM.reflection(rs, rp, spM.rotation(orientation[i], S))) 
    
    # characterization of the resulting Stokes vector
    PC = (S[3]/S[0])
    PL = (sp.sqrt(S[1]**2 + S[2]**2)/S[0])
    R = S[0]/vector[0]
    RPC2 = R*PC**2
    # ellipticity (Khokhova 2021)
    epsilon = -sp.tan(1/2*sp.atan(S[3]/sp.sqrt(S[1]**2 + S[2]**2)))

    # just evaluate the expressions if there were no parameters given
    if len(parameters) == 0:
        S = S.evalf()
        PC = PC.evalf()
        PL = PL.evalf()
        RPC2 = RPC2.evalf()
        epsilon = epsilon.evalf()
    elif simplify_expressions == True:
        
        print("\nSimplifying the expression for the resulting Stokes vector (1/5).\nIf it takes too long, relaunch the calculation with 'simplify_expressions = False'.")
        S = sp.simplify(S)
        print("\nSimplifying the expression for degree of circular polarization (2/5).\nIf it takes too long, relaunch the calculation with 'simplify_expressions = False'.")
        PC = sp.simplify(PC)
        print("\nSimplifying the expression for degree of linear polarization (3/5).\nIf it takes too long, relaunch the calculation with 'simplify_expressions = False'.")
        PL = sp.simplify(PL)
        print("\nSimplifying the expression for figure of merit (4/5).\nIf it takes too long, relaunch the calculation with 'simplify_expressions = False'.")
        RPC2 = sp.simplify(RPC2)
        print("\nSimplifying the expression for ellipticity (5/5).\nIf it takes too long, relaunch the calculation with 'simplify_expressions = False'.")
        epsilon = sp.simplify(epsilon)

    if len(parameters) == 1:
        print(f"\nSymbol used in expressions:\n\talpha: analyzer orientation from {np.min(x):.2f}° to {np.max(x):.2f}°")
    if len(parameters) == 2:
        print(f"\nSymbols used in expressions:")
        print(f"\talpha: first analyzer position from {np.min(x):.2f}° to {np.max(x):.1f}°")
        print(f"\tbeta: second analyzer position from {np.min(y):.2f}° to {np.max(y):.1f}°")

    print("\nSYMBOLIC EXPRESSIONS\n====================")
    print(f"\nStokes vector at the detector:\n\tS = {S/S[0]}")
    print(f"\nRelative intensity at the detector:\n\tS0 = {S[0]}")
    print(f"\nDegree of circular polarization at the detector:\n\tPC = S[3]/S[0] = {PC}")
    print(f"\nDegree of linear polarization at the detector:\n\tPL = sqrt(S[1]**2 + S[2]**2)/S[0] = {PL}")
    print(f"\nFigure of merit at the detector:\n\tR*PC**2 = {RPC2}")
    print(f"\nEllipticity at the detector:\n\tepsilon = {epsilon}")

    # close all graphs that may have remained in memory
    plt.close("all")

    # create a dictionary of graphs to be shown
    graphs = {}
    if show in ("all", "RPC2") or "RPC2" in show:
        graphs["RPC2 Data"] = {
            "expr": RPC2,
            "title": "Figure of Merit",
            "find_min": False,
            "label": r"$RP_C^2$",
            "color": "orange",
            "cmap": "plasma",
            "contours": "white"
        }
    if show in ("all", "PL") or "PL" in show:
        graphs["PL Data"] = {
            "expr": PL, 
            "title": "Degree of Linear Polarization", 
            "find_min": False,
            "label": r"$P_L$",
            "color": "dodgerblue",
            "cmap": "winter",
            "contours": "white"
        }
    if show in ("all", "PC") or "PC" in show:
        graphs["PC Data"] = {
            "expr": PC,
            "title": "Degree of Circular Polarization", 
            "find_min": True,
            "label": r"$P_C$",
            "color": "limegreen",
            "cmap": "berlin",
            "contours": "white"
        }
    if show in ("all", "ellipticity") or "ellipticity" in show:
        graphs["Ellipticity Data"] = {
            "expr": epsilon,
            "title": "Ellipticity", 
            "find_min": True,
            "label": r"$\epsilon$",
            "color": "limegreen",
            "cmap": "viridis",
            "contours": "white"
        }        
    if show in ("all", "intensity") or "intensity" in show:
        graphs["Intensity Data"] = {
            "expr": S[0],
            "title": "Intensity", 
            "find_min": True,
            "label": r"$S_0$",
            "color": "orange",
            "cmap": "viridis",
            "contours": "white"
        }
    if show == "stokes" or "stokes" in show:
        for i in range(1, 4):
            graphs[f"S_{i} Data"] = {
                "expr": S[i]/S[0],
                "title": f"S_{i}",
                "find_min": True,
                "label": rf"$S_{i}$",
                "color": "dodgerblue",
                "cmap": "viridis",
                "contours": "white"
        }

    # Create a dictionary of lambda functions: I need them in parallel in order to get all values at all maxima (e.g. intensity at max PC)
    lambdas = {}
    for _, config in graphs.items():
        lambdas[config["label"]] = sp.lambdify(parameters, config["expr"], "numpy")
    
    print("\nEXTREMES\n========")
    
    # 2D plot
    if len(parameters) == 1:

        # initiate data variable for 2D plot
        data = [x]
        data_header = ("alpha")

        # Sequential plot of all graphs
        for window_title, config in graphs.items():

            # broadcast_to forces the y-values to have the same dimension as x-values
            # if lambda(x) would be constant, it would return just an integer (which would later break up the plotting functions)
            y = np.broadcast_to(lambdas[config["label"]](x), x.shape)
            data.append(y)
            data_header += f",{config["label"]}"

            fig, ax = plt.subplots(figsize = (6, 3))
            fig.canvas.manager.set_window_title(window_title)

            ax.plot(x, y, color = config["color"], linewidth = 2)
            if show_title:
                ax.set_title(config["title"])
            ax.set_xlabel("analyzer orientation [°]")
            ax.set_ylabel(config["label"])

            # find the extremes (only if the functions are not constant, allclose() solves the floating point error)
            # custom absolute tolerance (atol) ensures data with very small deviations won't be considered constant
            if not np.allclose(y, y[0], atol = 1e-12):
                max_y = np.max(y)
                ix_max = np.where(y == max_y)
                max_x = x[ix_max]
                print(f"\nMaximum {config["title"]} of {max_y} found at:\n\t\N{GREEK SMALL LETTER ALPHA} [°] = {max_x}")

                # Also calculate other quantities from 'show' in this maximum
                if len(graphs) > 1:
                    print("\tOther values at this settings:")
                    for _, other_config in graphs.items():
                        # exclude the currently maximized value
                        if config["label"] != other_config["label"]:
                            other_value = lambdas[other_config["label"]](max_x)
                            print(f"\t\t{other_config["title"]} = {other_value}")

                # if necessary plot the minima as well (= Maximum RCP):
                if config["find_min"]:
                    min_y = np.min(y)
                    ix_min = np.where(y == min_y)
                    min_x = x[ix_min]
                    print(f"Minimum {config["title"]} of {min_y} found at:\n\t\N{GREEK SMALL LETTER ALPHA} [°] = {min_x}")

                    # Also calculate other quantities from 'show' in this minimum
                    if len(graphs) > 1:
                        print("\tOther values at this settings:")
                        for _, other_config in graphs.items():
                            # exclude the currently maximized value
                            if config["label"] != other_config["label"]:
                                other_value = lambdas[other_config["label"]](min_x)
                                print(f"\t\t{other_config["title"]} = {other_value}")

                    ax.plot(
                        min_x, np.broadcast_to(min_y, min_x.shape), 
                        marker = "x", 
                        color = "red", 
                        markersize = 12, 
                        linestyle = "None", 
                        label = f"min {config["label"]} ({min_y:.2f})"
                    )

                # Plot the maxima: Broadcast_to makes sure the y-values array is of the same shape as x-values. 
                ax.plot(
                    max_x, np.broadcast_to(max_y, max_x.shape), 
                    marker = "x", 
                    color = "black", 
                    markersize = 12, 
                    linestyle = "None", 
                    label = f"max {config["label"]} ({max_y:.2f})"
                )
                ax.legend(loc = "lower right")
         
        plt.show()

        # export data
        if export_data:
            data_columns = np.column_stack(data)
            np.savetxt("output.csv", data_columns, delimiter = ",", header = data_header)

        if return_data:
            print(f"\nReturning data in format:\n\t{data_header}")
            return(np.array(data))

    # 3D plots
    elif len(parameters) == 2:
        X, Y = np.meshgrid(x, y)

        # initiate data variable for 3D: the first element is an array of [alpha, beta] values
        data = [np.stack((X, Y), axis = -1)]
        data_header = "[alpha, beta]"

        for window_title, config in graphs.items():
            
            # X meshgrid has the same size as Z
            Z = np.broadcast_to(lambdas[config["label"]](X, Y), X.shape)
            
            # prepare the values to be later returned
            data.append(Z)
            data_header += f", {config["title"]}[alpha, beta]"

            # custom absolute tolerance (atol) ensures data with very small deviations won't be considered constant
            if not np.allclose(Z, Z[0, 0], atol = 1e-12):
                fig, ax = plt.subplots(figsize = (8, 6))
                fig.canvas.manager.set_window_title(window_title)
                if show_title:
                    ax.set_title(config["title"])
                ax.set_xlabel(r"$\alpha$ [°]")
                ax.set_ylabel(r"$\beta$ [°]")            

                max_z = np.max(Z)
                ix, iy = np.where(Z == max_z)
                max_x = X[ix, iy]
                max_y = Y[ix, iy]

                # if reasonable to find the minima as well + ensure color symmetry
                if config["find_min"]:
                    min_z = np.min(Z)
                    ix_min, iy_min = np.where(Z == min_z)
                    min_x = X[ix_min, iy_min]
                    min_y = Y[ix_min, iy_min]
                    print(f"\nMaximum {config["title"]} of {max_z} found at setting(s):\n\t\N{GREEK SMALL LETTER ALPHA} [°] = {max_x}\n\t\N{GREEK SMALL LETTER BETA} [°] =  {max_y}")

                    # Also calculate other quantities from 'show' in this maximum
                    if len(graphs) > 1:
                        print("\tOther values at this settings:")
                        for _, other_config in graphs.items():
                            # exclude the currently maximized value
                            if config["label"] != other_config["label"]:
                                other_value = lambdas[other_config["label"]](max_x, max_y)
                                print(f"\t\t{other_config["title"]} = {other_value}")

                    print(f"Minimum {config["title"]} of {min_z} found at setting(s):\n\t\N{GREEK SMALL LETTER ALPHA} [°] = {min_x}\n\t\N{GREEK SMALL LETTER BETA} [°] =  {min_y}")

                    # Also calculate other quantities from 'show' in this minimum
                    if len(graphs) > 1:
                        print("\tOther values at this settings:")
                        for _, other_config in graphs.items():
                            # exclude the currently maximized value
                            if config["label"] != other_config["label"]:
                                other_value = lambdas[other_config["label"]](min_x, min_y)
                                print(f"\t\t{other_config["title"]} = {other_value}")

                    # ensure color range symmetry in respect to 0 for non-symmetrical data
                    color_range = max(abs(max_z), abs(min_z))
                    symmetric_levels = np.linspace(-color_range, color_range, 33)
                    background = ax.contourf(X, Y, Z, levels = symmetric_levels, cmap = config["cmap"])

                    contours = ax.contour(X, Y, Z, levels = 8, colors =  config["contours"], linewidths = 1)                                    

                    # plot markers only after the contourf and contours (they could otherwise get obstructed)
                    ax.plot(
                        min_x, min_y,
                        marker = "x",
                        color = "red",
                        markersize = 12,
                        linestyle = "None",
                        label = f"min {config["label"]} ({min_z:.2f})"
                    )
                
                else:
                    print(f"\nMaximum {config["title"]} of {max_z} found at setting(s):\n\t\N{GREEK SMALL LETTER ALPHA} [°] = {max_x}\n\t\N{GREEK SMALL LETTER BETA} [°] =  {max_y}")

                    # Also calculate other quantities from 'show' in this maximum
                    if len(graphs) > 1:
                        print("\tOther values at this settings:")
                        for _, other_config in graphs.items():
                            # exclude the currently maximized value
                            if config["label"] != other_config["label"]:
                                other_value = lambdas[other_config["label"]](max_x, max_y)
                                print(f"\t\t{other_config["title"]} = {other_value}")

                    background = ax.contourf(X, Y, Z, levels = 32, cmap = config["cmap"])
                    contours = ax.contour(X, Y, Z, levels = 3, colors =  config["contours"], linewidths = 1)

                # mark the maxima
                ax.plot(
                    max_x, max_y, 
                    marker = "x", 
                    color = "black", 
                    markersize = 12, 
                    linestyle = "None", 
                    label = f"max {config["label"]} ({max_z:.2f})"
                )

                ax.clabel(contours, inline = True, fontsize = 10, colors = config["contours"])
                plt.colorbar(background, ax = ax, label = config["label"])
                ax.legend(loc = "upper right")

            else: 
                print(f"\n{config["title"]} is not plotted, its value is constant.")

        plt.show()