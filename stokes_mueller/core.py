"""
Basic Internal utilities of the stokes_mueller package.

This module contains unit convertors and physics formulas implementation. 
"""

import numpy as np
from scipy import constants as con


class DimensionError(Exception):
    """Raised in case of inconsistent or incorrect argument dimensions"""
    pass


def eV2nm(E_eV):
    """
    Converts energy in eV to wavelength in nm.

    Parameters
    ----------
    E_eV : float
        Energy of the light in eV.

    Returns
    -------
    float :
        Wavelength of the light in nm.

    See Also
    --------
    nm2eV :
        Reverse transformation.
    """
    lambda_nm = 10**9*con.Planck*con.c/con.electron_volt/E_eV
    return lambda_nm


def nm2eV(lambda_nm):
    """
    Converts wavelength in nm to energy in eV.

    Parameters
    ----------
    lambda_nm :
        Wavelength of the light in nm.

    Returns
    -------
    float :
        Energy of the beam in eV.

    See Also
    --------
    eV2nm :
        Reverse transformation.
    """
    E_eV = 10**9*con.Planck*con.c/con.electron_volt/lambda_nm
    return E_eV


def fresnel(n2, theta, n1 = 1 + 0j):
    """
    Calculates complex reflection coefficients utilizing Fresnel equations.
    
    Parameters
    ----------
    n2 : complex
        The transmitting medium's index of refraction, generally
        a complex number. 
    theta : float
        The angle of incidence in degrees. For coplanar polarizer geometry 
        assign as a array-like. 
    n1 : complex
        The incident medium's index of refraction. Default is 1 + 0j 
        (vacuum).

    Returns
    -------
    rs : complex
        The complex reflection coefficient for s-polarization component. 
    rp
        The complex reflection coefficient for p-polarization component.
    
    Notes
    -----
    if only one of the input parameters would be in a form of list, 
    I could have used just the non-array expression, and assume Python 
    would make rs and rp arrays automatically. However, this would work 
    only as long as EITHER only theta OR only n2 were arrays. 
    Once the input would be arrays of BOTH theta (coplanar multiple- 
    -reflection polarizer) AND n2 as well (simulating part of a spectrum),
    the intermediate arrays would acquire higher dimensions (matrices 
    would form) and Python would return an error after attempting certain 
    operations, e.g., sin of a matrix. The safest way is therefore to do 
    the array calculations in a controlled iterative manner.
    """
    # incidence angle
    theta1 = np.deg2rad(theta)

    # for a coplanar analyzer
    if isinstance(theta1, (list, tuple, np.ndarray)): 
        Rs = 1
        Rp = 1
        # initializing arrays
        theta2 = [] 
        cos_theta1 = []
        cos_theta2 = []
        rs = []
        rp = []
        # calculating only single reflection at a time
        for i in range(0, len(theta1)):
            # refraction angle (Snell's Law)
            theta2.append(np.arcsin(n1/n2*np.sin(theta1[i]))) 

            cos_theta1.append(np.cos(theta1[i]))
            cos_theta2.append(np.cos(theta2[i]))

            # Fresnel eqs. (Hecht: p. 124-125)
            rs.append((n1*cos_theta1[i] - n2*cos_theta2[i])/(n1*cos_theta1[i] + n2*cos_theta2[i]))
            rp.append((n2*cos_theta1[i] - n1*cos_theta2[i])/(n1*cos_theta2[i] + n2*cos_theta1[i]))

            Rs *= rs[i]
            Rp *= rp[i]
        return Rs, Rp
    
    # for a single reflection
    else:
        # refraction angle (Snell's Law)
        theta2 = np.arcsin(n1/n2*np.sin(theta1)) 

        cos_theta1 = np.cos(theta1)
        cos_theta2 = np.cos(theta2)

        # Fresnel eqs. (Hecht: p. 124-125)
        rs = (n1*cos_theta1 - n2*cos_theta2)/(n1*cos_theta1 + n2*cos_theta2)
        rp = (n2*cos_theta1 - n1*cos_theta2)/(n1*cos_theta2 + n2*cos_theta1)
        return rs, rp
    

def reflection_params(rs = -0.9 - 0.28j, rp = -0.33 + 0.7j):
    """
    Calculates the reflection parameters Psi and Delta.

    Parameters
    ----------
    rs : complex
        The complex reflection coefficient for s-polarization component.
    rp : complex 
        The complex reflection coefficient for p-polarization component.

    Returns
    -------
    Psi : float
        The reflection ratio angle in radians.
    Delta : float
        The phase difference between p- and s-polarization.
    """
    rho = abs(rp/rs) # reflection ratio (Koide: p. 637)

    psi = np.arctan(rho) # (Koide: p. 637)
    delta = np.angle(rp/rs)

    if delta < 0:
        delta += 2*np.pi
    
    return psi, delta