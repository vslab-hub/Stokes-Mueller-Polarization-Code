"""
Functions representing Mueller matrices implemented with SymPy library

These matrices are mostly used by functions contained in the package 
when symbolic expressions will be created and computational speed 
is not a priority.
"""

import sympy as sp
from ..core import reflection_params


def rotation(degrees, vector):
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
    sympy.Matrix :
        Stokes vector in the rotated coordinate system
        
    See Also
    --------
    stokes.matrices.numeric.rotation :
        identical transformation, but implemented with NumPy library

    Notes
    -----
    Utilized Sympy library, which enables the use of symbolic parameters.
    """
    # couldn't use np.deg2rad, because "degrees" may be a symbol
    a = degrees*sp.pi/180  # rad
    S = sp.Matrix(vector)

    M = sp.Matrix([[1,  0,           0,           0],
                   [0,  sp.cos(2*a), sp.sin(2*a), 0],
                   [0, -sp.sin(2*a), sp.cos(2*a), 0],
                   [0,  0,           0,           1]])
    
    return(M*S)


def linear_polarizer(vector):
    """
    Returns Stokes vector after passing through an ideal linear polarizer (e.g., Nicol prism).
    
    Parameters
    ----------
    vector : array-like
        incident Stokes vector (in the linear polarizer coordinate system)

    Returns
    -------
    sympy.Matrix
        Stokes vector exiting the linear polarizer (in the element's coordinate system)
    
    See Also
    --------
    rotation :
        Function for the Stokes vector coordinate system rotation.
    stokes.matrices.numeric.linear_polarizer :
        identical transformation, but implemented with NumPy library

    Notes
    -----
    Utilized Sympy library, which enables the use of symbolic parameters.
    """
    S = sp.Matrix(vector) # works for tuple, list, np.ndarray

    M = 0.5*sp.Matrix([[1, 1, 0, 0],
                       [1, 1, 0, 0],
                       [0, 0, 0, 0],
                       [0, 0, 0, 0]])
    
    return(M*S)


def retarder(degrees, vector):
    """
    Returns Stokes vector after passing through an optical retarder (a wave plate).

    Introduces a phase shift between the two orthogonal components of electric field.
    May represent a QWP, HWP or FWP depending on the phase shift.

    Parameters
    ----------
    degrees : float
        phase shift induced by the optical retarder [°]
    vector : array-like
        incident Stokes vector (in the optical retarder coordinate system)

    Returns
    -------
    sympy.Matrix
        Stokes vector exiting the optical retarder (in the element's coordinate system)

    See Also
    --------
    rotation :
        Function for the Stokes vector coordinate system rotation.
    stokes.matrices.numeric.retarder :
        identical transformation, but implemented with NumPy library

    Notes
    -----
    Utilizes Sympy library, which enables the use of symbolic parameters.
    """
    a = degrees*sp.pi/180  # rad
    S = sp.Matrix(vector)  # works for tuple, list, np.ndarray

    M = sp.Matrix([[1, 0, 0,          0        ],
                   [0, 1, 0,          0        ],
                   [0, 0, sp.cos(a), -sp.sin(a)],
                   [0, 0, sp.sin(a),  sp.cos(a)]])
    
    return(M*S)


def reflection(rs, rp, vector):
    """
    Returns Stokes vector after reflection.

    Parameters
    ----------
    rs : complex
        complex reflection coefficient of s-state component
    rp : complex
        complex reflection coefficient of p-state component
    vector : array-like
        incident Stokes vector    Returns
    
    Returns
    ------
    sympy.Matrix
        Stokes vector after the reflection (in the element's coordinate system)

    See Also
    --------
    rotation :
        Function for the Stokes vector coordinate system rotation.
    stokes.matrices.numeric.reflection :
        identical transformation, but implemented with NumPy library

    Notes
    -----
    Utilizes Sympy library, which enables the use of symbolic parameters.
    The Mueller matrix is based on the following publications:
    Shmising et al.: https://doi.org/10.14279/depositonce-10039
    Westerveld et al.: https://doi.org/10.1364/AO.24.002256
    """
    S = sp.Matrix(vector)
    psi, delta = reflection_params(rs, rp)

    M = 0.5*(abs(rs)**2 + abs(rp)**2)*sp.Matrix([[1,              -sp.cos(2*psi), 0,                            0                          ],
                                                 [-sp.cos(2*psi), 1,              0,                            0                          ],
                                                 [0,              0,              sp.sin(2*psi)*sp.cos(delta),  sp.sin(2*psi)*sp.sin(delta)],
                                                 [0,              0,              -sp.sin(2*psi)*sp.sin(delta), sp.sin(2*psi)*sp.cos(delta)]]) # (Schmising 2017), (Westerveld 1985)

    return(M*S)