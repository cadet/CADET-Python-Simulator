import warnings

import numpy as np

from CADETPythonSimulator.exception import CADETPythonSimError


def calculate_residual_volume_cstr(
        V: float,
        V_dot: float,
        Q_in: float,
        Q_out: float
        ) -> float:
    """
    Calculate the residual equations of the volume of a CSTR.

    Parameters
    ----------
    V : float
        Volume within the CSTR
    V_dot : float
        Volume change rate of the CSTR
    Q_in : float
        Volume entering the Unit
    Q_out : float
        Volume leaving the Unit

    Returns
    -------
    float
        Residual of the Flow equation of the CSTR with dimensions like the input

    """
    if V < 0:
        raise CADETPythonSimError("V can't be less then zero")

    return V_dot - Q_in + Q_out

def calculate_residual_concentration_cstr(
        c: np.ndarray,
        c_dot: np.ndarray,
        V: float,
        V_dot: float,
        Q_in: float,
        Q_out: float,
        c_in: np.ndarray
        ) -> np.ndarray:
    """
    Calculate the residual equations of the concentration of a CSTR.

    Parameters
    ----------
    c : np.ndarray
        Concentration
    c_dot : np.ndarray
        Changing of the concentration
    V : float
        Volume within the CSTR
    V_dot : float
        Volume change rate of the CSTR
    Q_in : float
        Volume entering the Unit
    Q_out : float
        Volume leaving the Unit
    c_in : np.ndarray
        Incomming concentration

    """
    if V < 0:
        raise CADETPythonSimError("V can't be less then zero")

    return c_dot * V + V_dot * c - Q_in * c_in + Q_out * c

def calculate_residual_visc_cstr():
    """Calculate the residual of the Viscosity equation of the CSTR."""
    warnings.warn("Viscosity of CSTR not yet implemented")

    return 0

def calculate_residual_cake_vol_def(
        V: float,
        rejection: np.ndarray,
        densities: np.ndarray,
        molecular_weights: np.ndarray,
        c_in: np.ndarray,
        V_C: float
        ) -> float:
    """
    Residual equation for the Cake Volume.

    Parameters
    ----------
    V : float
        Volume of liquid pressured into the filter
    rejection : np.ndarray
        Rejection of the filter
    densities : np.ndarray
        densities of suspended material
    molecular_weights : np.ndarray
        molecular weights of the components
    c_in : np.array
        Incoming Concentration
    V_C : float
        Cake Volume

    """
    return -V_C + V * np.sum(rejection * c_in * molecular_weights / densities)

def calculate_residual_press_easy_def(
        V_dot_Perm: float,
        V_C: float,
        deltap: float,
        A: float,
        mu: float,
        Rm: float,
        alpha: float
        ) -> float:
    """
    Calculate the residual equations.

    Calculates the residual equation for a dead end filtration equation
    for the pressure in the easy model.

    Parameters
    ----------
    V_dot_Perm : np.ndarray
        FLow of the Permeate through the membrane and Cake
    V_C : float
        Volume of the Cake
    deltap : float
        Pressure drop in this unit
    A : float
        Filtration area
    mu : float
        dynamic Viscosity
    Rm : float
        resistance of the medium
    alpha : float
        Specific cake resistance

    """
    hyd_resistance = (Rm + alpha*V_C/A) * mu

    return deltap * A - V_dot_Perm* hyd_resistance

def calculate_residual_visc_def():
    """Calculate the residual of the Viscosity equation of the CSTR."""
    warnings.warn("Viscosity of def not yet implemented")

    return 0
