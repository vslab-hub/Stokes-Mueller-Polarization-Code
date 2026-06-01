"""
Data analysis and regression tools for the stokes_mueller package

This module contains functions created specifically 
for the XUV polarizer data from ELI Beamlines Facility analysis.
It also enables regression of Stokes parameters from intensity data.
"""

import numpy as np
import sympy as sp
from pathlib import Path
from scipy.optimize import curve_fit
from .core import fresnel
from .materials import get_n
from .matrices import symbolic as spM


def get_data(folder_path) -> dict: 
    """
    Imports experimental data from the specified folder.

    Converts each .npy file into an np.ndarray and organizes 
    the arrays into a dictionary according to the corresponding
    orientation angle. 
    The filenames must contain '(...)_Angle_XX_(...).npy',
    where XX is the corresponding orientation angle.

    Parameters
    ----------
    folder_path: str
        Relative or absolute path to the folder containing data 
        in .npy files.
    
    Returns
    -------
    dict : 
        Dictionary containing the extracted data. The keyword
        to each array is its corresponding orientation angle
        {angle: data_array} 
    """
    folder = Path(folder_path)
    data_dict = {}

    # for each relevant (.npy) file extract the angle from its name
    for file_path in folder.glob("*.npy"):
        filename = file_path.stem

        # chop up the name
        _, keyword, after = filename.partition("_Angle_")

        # if keyword found
        if keyword:
            # chop out the rest after "_"
            angle_str = after.split("_")[0]
            angle = float(angle_str)

            # add a new keyword to the data dictionary
            data_dict[angle] = np.load(file_path)
    
    return data_dict


def subtract(exp_data, bcg_data) -> dict:
    """
    Subtracts background from experimental data. 

    Parameters
    ----------
    exp_data : dict
        Raw experimental data organized into an dictionary as
        {angle: data_array}.
    bcg_data : dict
        Background data organized into an dictionary as
        {angle: data_array}.

    Returns
    -------
    dict :
        Dictionary containing the subtracted data. The keyword
        to each array is its corresponding orientation angle
        {angle: data_array}

    See Also
    --------
    get_data :
        Function for data import in the structure compatible
        with the subtract function.

    Examples
    --------
    >>> # importing experimental data and subtracting the background
    >>> bcg_data = get_data("ELI_Data/HH13_bcg_pol_scan_fine")
    >>> raw_data = get_data("ELI_Data/HH13_9216_pol_scan_fine")
    >>> subtracted_data = subtract(exp_data, bcg_data)
    >>> # visualization of the result
    >>> plot_data(subtracted_data)
    """
    clean_data = {}

    for angle in exp_data:
        clean_data[angle] = np.clip(exp_data[angle] - bcg_data[angle], a_min = 0, a_max=None)

    return clean_data


def rectify(data_dict) -> dict:
    """
    Rectifies the data. 

    Clips the negative values in data (sets them all to zero).

    Parameters
    ----------
    data_dict : dict
        The data organized into a dictionary as {angle: data_array},
        which will be rectified.
    
    Returns
    -------
    dict :
        Rectified data in a form of dict {angle: data_array}.

    See Also
    --------
    get_data :
        Function for data import in the structure compatible
        with the rectify function.
    subtract :
        Function for the subtraction of background noise from raw data. 
        This step is usually performed prior to rectification.
        Data structure compatible.

    Examples
    --------
    >>> # importing experimental data
    >>> bcg_data = get_data("ELI_Data/HH13_bcg_pol_scan_fine")
    >>> raw_data = get_data("ELI_Data/HH13_9216_pol_scan_fine")
    >>> # subtracting the background
    >>> subtracted_data = subtract(exp_data, bcg_data)
    >>> # rectification
    >>> rect_data = rectify(subtracted_data)
    >>> # visualization
    >>> plot_data(rect_data)
    """
    rectified_data = {}

    for angle in data_dict:
        rectified_data[angle] = np.clip(data_dict[angle], a_min = 0, a_max=None)

    return rectified_data


def stabilize(data, movement_box, focus_box, start_angle = 270, direction = "clockwise"):
    """
    Compensates for spot 'wobble' in the image, assumes an elliptical path.
    
    The strategy is to parametrize the movement of a small window that would
    move perfectly along with the spot and therefore compensate for the wobble. 
    This parametrization is based on several observations (see Parameters) 
    the user should be able to make from a visualization of the averaged data 
    over all angles (see Examples). 

    Parameters
    ----------
    data : dict
        the experimental data in a form of a dictionary with structure
        angle: data_array
    movement_box : tuple of int
        A 4-element tuple defining the boundaries of the movement box. 
        The movement box is the smallest rectangle that confines 
        all movement of the spot within the image across all angles. 
        * x_min : int
        * y_min : int
        * x_max : int
        * y_max : int
    focus_box : tuple of int
        A 2-element tuple defining the size of the window that should track
        the spot within the image. 
        width : int
        height : int
    start_angle : float
        starting position of the window on the elliptical track in degrees
        (0 is for righmost position) for orientation angle = 0
        (or minimum angle from the dict).
    direction : str
        direction of window movement on the ellipsis track: 'clockwise'
        or 'counter clocwise'

    Returns
    -------
    dict :
        stabilized data in a form of dictionary with structure
        {angle: data_array}. The data_array has size of focus_box.

    See Also
    --------
    get_data, subtract, rectify :
        Functions recommended for import and processing of raw data
        before performing stabilization.
    visualize.plot_data :
        Function recommended for qiuck visualizations during
        the process of finding the optimal stabilization parameters.

    Examples
    -----
    >>> import numpy as np
    >>> # importing experimental data
    >>> bcg_data = get_data("ELI_Data/HH13_bcg_pol_scan_fine")
    >>> raw_data = get_data("ELI_Data/HH13_9216_pol_scan_fine")
    >>> # subtracting the background
    >>> subtracted_data = subtract(exp_data, bcg_data)
    >>> # rectification
    >>> rect_data = rectify(subtracted_data)
    >>> # visualizing several different datapoints at various angles
    >>> # this enables: 
    >>> # (1) correct choice of the size of the focus_box,
    >>> # (2) identification of the initial position of the spot
    >>> # (3) identification of the spot movement direction
    >>> plot(data[0])
    >>> plot(data[90])
    >>> plot(data[180])
    >>> plot(data[270])
    >>> # following paramters were chosen based on the visualizations:
    >>> height = 240
    >>> width = 245
    >>> start_angle = 270
    >>> direction = 'clockwise'
    >>> # averaging the data over all angles to visualize the movement box
    >>> sum_arr = np.zeros((400, 450))
    >>> for angle in range(-15, 376, 5):
    ...     sum_arr += rect_data[angle]
    >>> avg_arr = sum_arr/78
    >>> # visualizing the averaged array for the identification of remaining parameters
    >>> plot(avg_arr)
    >>> # The movement_box is chosen such that it will contain all the meaningful datapoints
    >>> x_min, x_max = 84, 382
    >>> y_min, y_max = 88, 380
    >>> # finally performing the stabilization itself
    >>> stabilized_data = stabilize(
    ...     rect_data, 
    ...     [x_min, y_min, x_max, y_max], 
    ...     [width, height], 
    ...     start_angle, 
    ...     direction
    ... )
    >>> # averaging the stabilized data to evaluate if the process was successful
    >>> stabilized_sum = np.zeros((height, width))
    >>> for angle in range(0, 360, 5):
    ...     stabilized_sum += stabilized_data[angle]
    >>> stabilized_avg = stabilized_sum/78
    >>> plot(stabilized_avg)
    >>> # If the stabilization was successful, the averaged image should be sharp and full of detail
    >>> # If it isn't, try fine-tuning the stabilization parameters
    >>> # and plot the intermediate results each time.
    >>> # Good luck!
    """

    # dict for data cropped inside the movement box
    boxed_data = {}
    # dict for the stabilized images
    stabilized_data = {}

    start_angle_rad = np.deg2rad(start_angle)

    width = focus_box[0]
    height = focus_box[1]

    # bounds of the movement box
    min_x = movement_box[0]
    min_y = movement_box[1]
    max_x = movement_box[2]
    max_y = movement_box[3]

    # ellipsis trace of the window corner params: 
    # a = horizontal axis
    a = (max_x - min_x - width)/2
    # b =vertical axis
    b = (max_y - min_y - height)/2
    # centre of the ellipsis: C = [m, n]
    m = (min_x + max_x - width)/2
    n = (min_y + max_y - height)/2
    
    if direction in ("clockwise", "Clockwise", "c", "C"): 
        factor = -1
    elif direction in ("counter clockwise", "cc"):
        factor = 1
    else:
        raise ValueError("Invalid 'direction' option")

    for angle in data:
        angle_rad = np.deg2rad(factor*angle)

        # confine data into the movement box
        boxed_data[angle] = data[angle][min_y:max_y, min_x:max_x]

        # calculate the current position of the window corner inside the whole data array
        corner_x = np.rint(m + a*np.cos(angle_rad + start_angle_rad))
        corner_y = np.rint(n + b*np.sin(angle_rad + start_angle_rad))

        row_start = int(corner_y)
        row_stop = int(row_start + height)
        col_start = int(corner_x)
        col_stop = int(col_start + width)

        # stabilized_data[angle] = boxed_data[angle][int(corner_y):int(corner_y + height), int(corner_x):int(corner_x + width)]
        stabilized_data[angle] = data[angle][row_start:row_stop, col_start:col_stop]
        

    return stabilized_data


def integrate(data_array, row_start, row_end, col_start, col_end):
    """
    Sums all the values within the specified region of the array.

    Parameters
    ----------
    data_array : numpy.ndarray
        2D array containing the data to be integrated. 
    row_start : int
        The upper limit of the integration region within the data array.
    row_end : int
        The lower limit of the integration region within the data array.
    col_start : int
        The left limit of the integration region within the data array.
    col_end : int
        The right limit of the integration region within the data array.

    Returns
    -------
    float :
        Sum of all the values of the data array within the specified 
        integration region

    See Also
    -------
    get_data, subtract, rectify, stabilize:
        Functions recommended for import and processing of raw data
        before performing integration.

    Examples
    --------
    >>> # importing experimental data
    >>> bcg_data = get_data("ELI_Data/HH13_bcg_pol_scan_fine")
    >>> raw_data = get_data("ELI_Data/HH13_9216_pol_scan_fine")
    >>> # subtracting the background
    >>> subtracted_data = subtract(exp_data, bcg_data)
    >>> # rectification
    >>> data = rectify(subtracted_data)
    >>> # choosing one orientation for the integration
    >>> # visualization enables precise choice of the integration region
    >>> plot(data[67])
    >>> # defining the integration region
    >>> col_start, col_end = 145, 185
    >>> row_start, row_end = 85, 96
    >>> # integrating
    >>> int67 = integrate(data, row_start, row_end, col_start, col_end)
    >>> print(int67)
    """
    integration_region = data_array[row_start:row_end, col_start:col_end]
    region_sum = integration_region.sum()

    return(region_sum)


def fit_data(data, E_eV, element, incidence, orientation, source = "auto", normalize = False):
    """
    Returns the Stokes vector of the incident light by curve-fitting 
    the provided data to the optical setup simulation.

    Parameters
    ----------
    data : array-like of float
        2-membered array of the experimental results to be regressed.
        * angle : float
            Orientation of the variable-orientation element (analyzer)
            in degrees. 
        * intensity : float
            Intensity measured at the detector. Arbitrary linear scale. 
    E_eV : float
        Energy of the incident beam in electronvolts.
    element : list or tuple of str
        sequence of elements IDs':
            'lp' for linear polarizer
            'qwp' for a quarter-wave plate
            'hwp' for a half-wave plate
            material of a mirror - chemical sumbol (e.g., 'Au')
    incidence : list or tuple of floats
        Incidence angle onto a mirror in degrees. Relevant only for reflective elements 
        (may be set to arbitary number for transmissive elements).  
    orientation : array-like of float or str
        Array of elements orientations. The element whose orientation 
        is the parameter of the experimental data must be passed on as a str,
        e.g., as 'alpha'. This parameter identification will be used in the
        symbolic expressions of the funciton output. 
    source : str, optional
        Preffered experimental data source for the refractive index from RefractiveIndex database.
        Default is 'auto', which will select the best aviable source for the given beam energy.    
    normalize : bool, optional
        Whether or not to return the normalized Stokes vector and the regression
        covariance matrix. When the Stokes vector is not normalized, its S_0 
        is on the scale corresponding to the detected intensity. Default is False.

    Returns
    -------
    np.ndarray :
        incident beam Stokes vector
    np.ndarray :
        covariance matrix of regressed Stokes parameters

    Notes
    -----
    Assumes perfectly polarized incident light. 
    Assumes analyzer being insensitive towards degree of circular polarization, 
    therefore deduces S3 from other Stokes parameters.

    Examples
    --------
    >>> import numpy as np
    >>> # experimental data import and processing
    >>> background_data = get_data("ELI_Data/HH13_bcg_pol_scan_fine")
    >>> raw_data = get_data("ELI_Data/HH13_9216_pol_scan_fine")
    >>> subtracted_data = subtract_data(raw_data, background_data)
    >>> data = rectify(subtracted_data)
    >>> # defining the integration region
    >>> col_start, col_end = 150, 380
    >>> row_start, row_end = 90, 310
    >>> # getting the intensity profile by integration of the data
    >>> angles = []
    >>> exp_int = []
    >>> for orientation in data:
    ...     angles.append(orientation)
    ...     exp_int.append(integrate(data[orientation], row_start, row_end, col_start, col_end))
    >>> # performing the regression itself
    >>> S_regressed, cov = fit_data(
    ...     [angles, exp_int],
    ...     E_eV = 20.15,
    ...     element = ["Au"],
    ...     incidence = [(62.5, 62.5, 62.5, 62.5)],
    ...     ['alpha']),
    ...     normalize = True
    ... )
    >>> # printing the regression results
    >>> print(S_regressed)
    >>> print(cov)

    """
    N = len(element)
    alpha = list(orientation)
    xdata = data[0]
    ydata = data[1]

    # initialize symbolic stokes params.
    s0, s1, s2, s3 = sp.symbols("s0, s1, s2, s3")
    S = sp.Matrix([s0, s1, s2, s3])

    # passage through the whole setup
    for i in range(0, N):

        # the orientation angle passed on as a string is assumed to be the analyzer orientation
        if isinstance(alpha[i], str):
            a = sp.symbols(alpha[i])
            alpha[i] = a
        
        match element[i]:
            case "lp":
                S = spM.rotation(-alpha[i], spM.linear_polarizer(spM.rotation(alpha[i], S)))
            case "qwp":
                S = spM.rotation(-alpha[i], spM.retarder(90, spM.rotation(alpha[i], S)))
            case "hwp":
                S = spM.rotation(-alpha[i], spM.retarder(180, spM.rotation(alpha[i], S)))
            case _:
                n = get_n(element[i], E_eV, source)
                rs, rp = fresnel(n, incidence[i])
                S = spM.rotation(-alpha[i], spM.reflection(rs, rp, spM.rotation(alpha[i], S)))

    S[0] = sp.simplify(S[0])
    print(f"\nIntensity as a function of {a}:\n\tS[0] = {S[0]}")

    # convert the expression of the resulting intensity into a lambda function
    if S[0].has(s3):
        intensity = sp.lambdify([a, s0, s1, s2, s3], S[0], "numpy")
    else:
        intensity = sp.lambdify([a, s0, s1, s2], S[0], "numpy")

    # initial guess on the same order of magnitude as the data
    intensity_guess = np.max(ydata)
    optS, pcov = curve_fit(intensity, xdata, ydata, p0 = (intensity_guess, 0, 0))

    # get standard uncertainties from the covariance matrix
    sigma = []
    for element in np.diagonal(pcov):
        sigma.append(np.sqrt(element))

    # deduce S3 if not present in the intenisty expression
    if not S[0].has(s3):
        optS = np.append(optS, np.sqrt(optS[0]**2 - optS[1]**2 - optS[2]**2))
        sigma.append(1/np.abs(optS[3])*np.sqrt((2*np.abs(optS[0]))**2 + (2*np.abs(optS[1]))**2 + (2*np.abs(optS[2]))**2))

    print(f"\nRegressed Stokes vector before normalization:\n\t{optS}")
    print(f"\nCovariance matrix before S normalization:\n{pcov}")
    print(f"\nStandard uncertainties of Stokes parameters before S normalization:\n{sigma}")

    # normalize the covariance matrix, Stokes vector and uncertainties
    pcov_norm = pcov/(optS[0]**2)
    optS_norm = optS/optS[0]
    sigma_norm = sigma/optS[0]

    print(f"\nRegressed Stokes vector:\n\t{optS_norm}")
    print(f"\nCovariance matrix:\n{pcov_norm}")
    print(f"\nStandard uncertainties of Stokes parameters:\n{sigma_norm}")

    if normalize:
        return np.array(optS_norm), np.array(pcov_norm)
    else:
        return np.array(optS), np.array(pcov)