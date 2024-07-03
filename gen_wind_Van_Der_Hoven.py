# -*- coding: utf-8 -*-
"""
Created on Thu May  9 21:26:25 2024

Wind generator

@author: Yihan Liu
"""
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import splev
import pickle

def von_der_Haven_spectrum(frequency):
    """
    return Van der Hoven’s spectral

    Parameters
    ----------
    frequency : double
        frequency in cycle per hour

    Returns
    -------
    TYPE
        DESCRIPTION.

    """
    with open('Van_der_Hoven_spectral_fit.pkl', 'rb') as f:
        tck = pickle.load(f)
    
    log_frequency = np.log10(frequency)

    spectrum_value = splev(log_frequency, tck)
    
    return spectrum_value / frequency
    

    
def generate_frequencies(N = 9, k_range = (-3, 1), base=10):
    """
    Generate discrete frequencies for the simulation.

    Parameters
    ----------
    N : int
        Number of frequencies per decade
    k_range : tuple
        Range of decades (min_k, max_k)

    Returns
    -------
    list
        List of frequencies (in cycles/hour)
    """
    frequencies = []
    for k in range(k_range[0], k_range[1] + 1):
        frequencies.extend([i * (base**k) for i in range(1, N+1)])
    return frequencies

def generate_frequencies(N=9, k_range=(-3, 2), base=10):
    """
    Generate discrete frequencies for the simulation.

    Parameters
    ----------
    N : int
        Number of frequencies per decade
    k_range : tuple
        Range of decades (min_k, max_k)

    Returns
    -------
    list
        List of frequencies (in cycles/hour)
    """
    frequencies = []
    for k in range(k_range[0], k_range[1] + 1):
        frequencies.extend([i * (base**k) for i in range(1, N + 1)])
    return frequencies

def get_median_long_component(T_s1, T_F, v_bar, white_noise):
    """
    Simulate the medium- and long-term wind speed component.

    Parameters
    ----------
    T_s1 : float
        Sampling period for medium-long term component (seconds)
    T_F : float
        Total simulation time (seconds)
    v_bar : float
        Mean wind speed
    white_noise : np.array
        white noise should be uniform distribution from -pi to pi 
        with length equal to the length of frequency array. 
        Here this length is 31

    Returns
    -------
    np.array
        Simulated wind speeds over time.
    """
    frequencies = generate_frequencies()[:31]
    omegas = np.array(frequencies) * 2 * np.pi   # Convert cycles/hour to rad/hour
    amplitudes = []
    #phases = np.random.uniform(-np.pi, np.pi, size=len(omegas))
    
    # Calculate amplitudes using the Van der Hoven spectrum
    for i in range(len(omegas) - 1):
        Svv_i = von_der_Haven_spectrum(frequencies[i])
        Svv_ip1 = von_der_Haven_spectrum(frequencies[i + 1])
        amplitude = (2/np.pi)*np.sqrt(0.5*(Svv_i + Svv_ip1)*(omegas[i+1] - omegas[i]))
    
        amplitudes.append(amplitude)
    
    amplitudes = np.array(amplitudes)
    
    wind_size = int(np.ceil(T_F / T_s1))
    wind_speeds = np.full(wind_size, v_bar, dtype=float)

    
    # for each time step
    for t in range(wind_size):
        # at time t, for each frequency
        for i in range(len(amplitudes)):
            wind_speeds[t] += amplitudes[i]*np.cos(omegas[i]*t + white_noise[i])
    
    return wind_speeds



def gen_turbulence(v_bar, L, k_sigma_v, T_s, N_t, white_noise, 
                   delta_omega = 0.002, M = 5000, N = 100):
    """
    Generate turbulencec component for wind speed

    Parameters
    ----------
    v_bar : int
        Average wind speed
    L : int
        Turbulence length
    k_sigma_v : float
        Slope parameter
    T_s : int
        Time step
    N_t : int
        Number of time steps
    white_noise : np.array
        the white noise with mean = 0 and std = 1, has length N_t

    Returns
    -------
    Array of wind speed with turbulence

    """
    
    # Step 1: Update the current values of the parameters in 
    # the turbulence component model
    T_F = L / v_bar
    sigma_v = k_sigma_v * v_bar
    K_F = np.sqrt((2 * np.pi * T_F)/(4.20654 * T_s))
    
    # Step 2: Calculate the discrete impulse response of the filter
    delta_omega = 0.002 # Frequency step size
    M = 5000 # Number of frequency points
    N = 100 # Numerical parameters for convolution integration, divide 
            # finite integral from 0 to t to N regions
    
    # Discrete frequency domain P(omega)
    P = np.zeros(M + 1)
    for r in range(M + 1):
        P[r] = np.real(K_F / (1 + 1j * r * delta_omega * T_F)**(5/6))
    
    # Discrete impulse response h(k) === h(T_s*k), k range from 0 to N
    h = np.zeros(N + 1)
    for k in range(N + 1):
        h[k] = T_s * delta_omega * (2/np.pi) * np.sum(P * np.cos(k * np.arange(M + 1) * T_s * delta_omega))
    
    # Step 3: Generate the turbulence component in the interval using convolution
    v_t = np.zeros(N_t)
    
    # Zero-pad the white noise 
    white_noise_padded = np.pad(white_noise, (0, N), 'constant')
    
    for m in range(N_t):
        v_t[m] = T_s * np.sum(h * white_noise_padded[m : m + N + 1])
    
    return v_bar + sigma_v * v_t
    
def generate_wind(v_bar, L, k_sigma_v, T_s, T_s1, T_F, white_noise_ml, white_noise_turb):
    """
    Generate wind speed with turbulence for each average wind speed in the array.

    Parameters
    ----------
    v_bar : float
        Mean wind speed
    L : float
        Turbulence length scale
    k_sigma_v : float
        Slope parameter
    T_s : int
        Time step (sampling period)
    T_s1 : int
        Duration for each average wind speed
    T_F : int
        Total simulation time
    white_noise_ml : np.array
        White noise array for medium-long term component
    white_noise_turb : np.array
        White noise array for turbulence component

    Returns
    -------
    np.array
        Large array of wind speeds with turbulence
    """
    # Generate medium-long term component
    v_ml = get_median_long_component(T_s1, T_F, v_bar, white_noise_ml)
    
    large_wind_speed_array = []
    
    for i in range(len(v_ml)):
        N_t = int(T_s1 / T_s)
        white_noise_segment = white_noise_turb[i * N_t : (i + 1)* N_t]
        wind_with_turbulence = gen_turbulence(v_ml[i], L, k_sigma_v, T_s, N_t, white_noise_segment)
        large_wind_speed_array.extend(wind_with_turbulence)
    
    return np.array(large_wind_speed_array), v_ml
    


# Parameters
T_s1 = 180  # Sampling period (seconds)
T_total = 3000  # Total simulation time (seconds)
v_bar = 20  # Mean wind speed (m/s)

L = 180  # Turbulence length scale in meters
k_sigma_v = 0.13 # Slope parameter
T_s = 1  # Time step in seconds

# Generate white noise for each segment
import random
seeds = [random.randint(0, 9999999) for _ in range(3)]
#seeds = [6668902, 7679329, 3083145]
# Print the list of random numbers
print(seeds)
# generate medium long component noise use the first seed
state_before = np.random.get_state()
np.random.seed(seeds[0])
white_noise_ml = np.random.uniform(-np.pi, np.pi, 31) 
np.random.set_state(state_before)

# generate turbulence noise use the second seed
state_before = np.random.get_state()
np.random.seed(seeds[1])
white_noise_turb = np.random.normal(0, 1, int(np.ceil(T_total / T_s1) * T_s1))  # For turbulence component
np.random.set_state(state_before)

#white_noise_ml = np.random.uniform(-np.pi, np.pi, 31)  # For phase in medium-long term
#white_noise_turb = np.random.normal(0, 1, int(np.ceil(T_total / T_s1) * T_s1))  # For turbulence component

# Generate large array of wind speeds with turbulence
wind_speeds, v_ml = generate_wind(v_bar, L, k_sigma_v, T_s, T_s1, T_total, white_noise_ml, white_noise_turb)

# Plot the simulated wind speeds
plt.plot(np.arange(0, len(wind_speeds), T_s), wind_speeds, 'b-', linewidth=0.5, label='Wind Speed')
plt.xlabel('Time (s)')
plt.ylabel('Wind Speed (m/s)')
plt.title('Wind Speed')
plt.grid(True)
plt.show()





