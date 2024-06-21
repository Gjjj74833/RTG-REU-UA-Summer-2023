# -*- coding: utf-8 -*-
"""
Created on Wed Apr 17 17:42:06 2024

@author: Yihan Liu
"""

def pierson_moskowitz_spectrum(U19_5, zeta, eta, t, random_phases):
    """
    This function generates the Pierson-Moskowitz spectrum for a given wind speed U10 and frequency f.
    
    parameters
    ----------
    U19_5 : float
        the average wind speed at 19.5m above the sea surface
    zeta : float
        the x component to evaluate
    eta : float
        the y component to evaluate. (Note: the coordinate system here is different
                                      from the Betti model. The downward is negative
                                      in this case)
    t: float
        the time to evaluate.
    random_phase : Numpy Array
        the random phase to generate wave. Should be in [0, 2*pi)

    Returns
    -------
    wave_eta : float
        The wave elevation
    [v_x, v_y, a_x, a_y]: list
        The wave velocity and acceleration in x and y direction
    """
    g = 9.81  # gravitational constant
    alpha = 0.0081  # Phillips' constant

    f_pm = 0.14*(g/U19_5)  # peak frequency
    
    N = 400
    
    cutof_f = 3*f_pm # Cutoff frequency
    
    f = np.linspace(0.1, cutof_f, N) # Array
    omega = 2*np.pi*f # Array
    delta_f = f[1] - f[0] # Array

    S_pm = (alpha*g**2/((2*np.pi)**4*f**5))*np.exp(-(5/4)*(f_pm/f)**4) # Array
    
    a = np.sqrt(2*S_pm*delta_f)
    k = omega**2/g    
    
    # Generate random phases all at once
    
    
    # Perform the calculations in a vectorized manner
    sin_component = np.sin(omega*t - k*zeta + random_phases)
    cos_component = np.cos(omega*t - k*zeta + random_phases)
    exp_component = np.exp(k*eta)
    
    wave_eta = np.sum(a * sin_component)
    
    v_x = np.sum(omega * a * exp_component * sin_component)
    v_y = np.sum(omega * a * exp_component * cos_component)
    
    a_x = np.sum((omega**2) * a * exp_component * cos_component)
    a_y = -np.sum((omega**2) * a * exp_component * sin_component)

    return wave_eta, [v_x, v_y, a_x, a_y]