"""
Functions representing Mueller matrices implemented with NumPy library

These matrices are mostly used by functions contained in the package 
when symbolic expressions are unnecessary and fast computational speed
is prioritized.
"""

import numpy as np
from ..core import reflection_params


def inversion(matrix) -> np.ndarray:
    """
    Returns matrix after inversion.

    Parameters
    ----------
    matrix : numpy.ndarray
        matrix to be inverted

    Returns
    -------
    numpy.ndarray
        inverted matrix

    Notes
    -----
    Utilizes NumPy library to optimize performance.
    """
    M = np.asarray(matrix)
    Minv = np.linalg.inv(M)
    return(Minv)


def rotation(degrees, vector) -> np.ndarray:
    """
    Returns Stokes vector after coordinate system rotation.

    Parameters
    ----------
    degrees : float
        angle of rotation [°]
    vector : array-like
        incident Stokes vector

    Returns
    -------
    numpy.ndarray
        Stokes vector in the rotated coordinate system
        
    See Also
    --------
    stokes.matrices.symbolic.rotation :
        identical transformation, but implemented with SymPy library

    Notes
    -----
    Utilizes NumPy library to optimize performance.
    """
    a = np.deg2rad(degrees)
    s0 = np.asarray(vector)

    M = np.array([[1,  0,           0,           0],
                  [0,  np.cos(2*a), np.sin(2*a), 0],
                  [0, -np.sin(2*a), np.cos(2*a), 0],
                  [0,  0,           0,           1]])
    
    return M@s0


def linear_polarizer(vector) -> np.ndarray:
    """
    Returns Stokes vector after passing through an ideal linear polarizer (e.g., Nicol prism).
    
    Parameters
    ----------
    vector : array-like
        incident Stokes vector (in the linear polarizer coordinate system)

    Returns
    -------
    numpy.ndarray
        Stokes vector exiting the linear polarizer (in the element's coordinate system)
    
    See Also
    --------
    rotation :
        Function for the Stokes vector coordinate system rotation.
    stokes.matrices.symbolic.linear_polarizer :
        identical transformation, but implemented with SymPy library

    Notes
    -----
    Utilizes NumPy library to optimize performance.
    """
    s0 = np.asarray(vector)

    M = 0.5*np.array([[1, 1, 0, 0],
                      [1, 1, 0, 0],
                      [0, 0, 0, 0],
                      [0, 0, 0, 0]])
    
    return M@s0


def retarder(degrees, vector) -> np.ndarray:
    """
    Returns Stokes vector after passing through an optical retarder (a wave plate).

    Introduces a phase shift between the two orthogonal components of electric field.
    May represent a QWP, HWP or FWP depending on the phase shift.

    Parameters
    ----------
    degrees : float
        phase shift induced by the optical retarder[°]
    vector : array-like
        incident Stokes vector (in the optical retarder coordinate system)

    Returns
    -------
    numpy.ndarray
        Stokes vector exiting the optical retarder (in the element's coordinate system)

    See Also
    --------
    rotation :
        Function for the Stokes vector coordinate system rotation.
    stokes.matrices.symbolic.retarder :
        identical transformation, but implemented with SymPy library

    Notes
    -----
    Utilizes NumPy library to optimize performance.
    """
    a = np.deg2rad(degrees)
    s0 = np.asarray(vector)

    M = np.array([[1, 0, 0,          0        ],
                  [0, 1, 0,          0        ],
                  [0, 0, np.cos(a), -np.sin(a)],
                  [0, 0, np.sin(a),  np.cos(a)]])
    
    s = M@s0
    return s


def reflection(rs, rp, vector):
    """
    Returns Stokes vector after reflection.

    Parameters
    ----------
    rs : complex
        The complex reflection coefficient of s-state component.
    rp : complex
        The complex reflection coefficient of p-state component.
    vector : array-like
        The incident Stokes vector.

    Returns
    ------
    numpy.ndarray
        The Stokes vector after the reflection (in the element's 
        coordinate system).

    See Also
    --------
    rotation :
        Function for the Stokes vector coordinate system rotation.
    stokes.matrices.symbolic.reflection :
        Identical transformation, but implemented with SymPy library.

    Notes
    -----
    Utilizes NumPy library to optimize performance.
    The Mueller matrix is based on the following publications:
    Shmising et al.: https://doi.org/10.14279/depositonce-10039,
    Westerveld et al.: https://doi.org/10.1364/AO.24.002256.
    """
    s0 = np.asarray(vector)
    psi, delta = reflection_params(rs, rp)

    M = 0.5*(abs(rs)**2 + abs(rp)**2)*np.array([[1,              -np.cos(2*psi), 0,                            0                          ],
                                                [-np.cos(2*psi), 1,              0,                            0                          ],
                                                [0,              0,              np.sin(2*psi)*np.cos(delta),  np.sin(2*psi)*np.sin(delta)],
                                                [0,              0,              -np.sin(2*psi)*np.sin(delta), np.sin(2*psi)*np.cos(delta)]])
    
    return M@s0