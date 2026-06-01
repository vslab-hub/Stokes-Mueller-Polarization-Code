"""
Data and simulation results visualization module for the stokes_mueller 
package.

This module enables plots of experimental and simulated data 
specifically created for outputs of the stokes_mueller package functions.
"""

import numpy as np
import matplotlib.pyplot as plt

# Configure matplotlib to use Times New Roman font
plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = ["Times New Roman"]
plt.rcParams["mathtext.fontset"] = "stix"


def plot_data(data, angles = "all"):
    """
    Standard plot for experimental data.

    Compatible with the outputs of functions of the stokes.analysis
    module. 

    Parameters
    ----------
    data : numpy.ndarray or dict
        The values to be visualized. May be a single array
        or a dictionary organized as {angle: array}. 
        In the case of dict, the arrays to be plotted are specified
        by the 'angles' parameter.
    angles : str or array-like or float, optional
        Which items of the data dictionary to vizualize.
        'all' for all items of the dictionary.
        Array of values for only certain items of the dict.
        A single value to plot only the corresponding array.

    See Also
    --------
    get_data, subtract, rectify, stabilize :
        Output of these functions has a format compatible with
        the plot_data function. 

    Examples
    --------
    >>> # importing experimental data and subtracting the background
    >>> bcg_data = get_data("ELI_Data/HH13_bcg_pol_scan_fine")
    >>> raw_data = get_data("ELI_Data/HH13_9216_pol_scan_fine")
    >>> subtracted_data = subtract(exp_data, bcg_data)
    >>> # visualization of the result
    >>> plot_data(subtracted_data)
    """
    if isinstance(data, np.ndarray):
        plt.imshow(data, cmap = "viridis", origin = "lower")
        plt.colorbar(label = "intensity")
        # plt.title("Data plot")
        plt.xlabel("x")
        plt.ylabel("y")

    elif isinstance(data, dict):
        # identify the range of angles to generate plots for
        if isinstance(angles, (tuple, list, np.ndarray)):
            angles_range = angles
        elif isinstance(angles, int):
            angles_range = [angles]
        elif angles == "all":
            angles_range = list(data.keys())
        else:
            raise ValueError(f"invalid 'angles' input: Is {type(angles)}, but must be array-like or the default string 'all'")
        
        for angle in angles_range:
            plt.figure()
            plt.imshow(data[angle], cmap = "viridis", origin = "lower")
            plt.colorbar(label = "intensity")
            plt.title(f"Data plot for {angle}°")
    
    else:
        raise ValueError(f"invalid data. Must be of type 'np.ndarray' or 'dict', but is '{type(data)}'.")
    
    plt.show()