# -*- coding: utf-8 -*-
"""
Betti model implementation

@author: Yihan Liu
@version (2023-06-24)
"""

import numpy as np
import matplotlib.pyplot as plt
import os
import bisect
import time
import random
from multiprocessing import Pool


# The model

def process_rotor_performance(input_file = "Cp_Ct.NREL5MW.txt"):
    """
    This function will read the power coefficient surface from a text file generated
    by AeroDyn v15 and store the power coefficient in a 2D list

    Parameters
    ----------
    input_file : String, optional
        The file name of the pwer coefficient

    Returns
    -------
    C_p : 2D list
        The power coefficient. col: pitch angle, row: TSR value
    C_t : 2D list
        The thrust coefficient. col: pitch angle, row: TSR value
    pitch_angles : list
        The pitch angle corresponding to the col of C_p
    TSR_values : list
        The TSR values corresponding to the row of C_p

    """
    
    pitch_angles = []
    TSR_values = []

    with open(input_file, 'r') as file:
        lines = file.readlines()

        # Extract pitch angle vector
        pitch_angles_line = lines[4]
        # Extract TSR value vector
        TSR_values_line = lines[6]
        
        pitch_angles = [float(num_str) for num_str in pitch_angles_line.split()]
        TSR_values = [float(num_str) for num_str in TSR_values_line.split()]
        
        C_p = []
        for i in range(12, 12 + len(TSR_values)):
            Cp_row = [float(num_str) for num_str in lines[i].split()]
            C_p.append(Cp_row)
            
        C_t = []
        for i in range(16 + len(TSR_values), 16 + len(TSR_values) + len(TSR_values)):
            Ct_row = [float(num_str) for num_str in lines[i].split()]
            C_t.append(Ct_row)

    return C_p, C_t, pitch_angles, TSR_values



    

def CpCtCq(TSR, beta, performance):
    """
    Find the power coefficient based on the given TSR value and pitch angle

    Parameters
    ----------
    TSR : Tip speed ratio
    beta : blade pitch angle
    performance: The rotor performance generated by processing process_rotor_performance()

    Returns
    -------
    C_p: float
        power coefficient
    C_t: float
        thrust coefficient
    """
    beta = np.rad2deg(beta)

    C_p = performance[0] 
    C_t = performance[1]
    pitch_list = performance[2] 
    TSR_list = performance[3]
    
    # Find the closed pitch and TSR value in the list
    pitch_index = bisect.bisect_left(pitch_list, beta)
    TSR_index = bisect.bisect_left(TSR_list, TSR)
    
    # Correct the index if it's out of bounds or if the previous value is closer
    if pitch_index != 0 and (pitch_index == len(pitch_list) or abs(beta - pitch_list[pitch_index - 1]) < abs(beta - pitch_list[pitch_index])):
        pitch_index -= 1
    if TSR_index != 0 and (TSR_index == len(TSR_list) or abs(TSR - TSR_list[TSR_index - 1]) < abs(TSR - TSR_list[TSR_index])):
        TSR_index -= 1
    
    # Get the C_p value at the index 
    return C_p[TSR_index][pitch_index], C_t[TSR_index][pitch_index]


def drvCpCtCq(omega_R, v_in, beta):
    """
    Compute the power coefficient use AeroDyn v15 driver. Based on input 
    requirements for AeroDyn v15, we take the rotor speed and relative 
    wind velocity as parameters instead of TSR. there's no need to compute 
    TSR when use this function to calculate Cp and Ct.

    Parameters
    ----------
    omega_R : float
        The rotor speed in rad/s
    v_in : float
        relative wind speed
    beta : float
        blade pitch angle in rad

    Returns
    -------
    float
        the power coefficient
        the thrust coefficient

    """

    
    # Convert rad/s to rpm since the AeroDyn driver takes rpm as parameter
    omega_rpm = omega_R*(60 / (2*np.pi))
    beta_deg = beta*(180/np.pi)
    
    omega_Rstr = str(omega_rpm)
    v_instr = str(v_in)
    beta_str = str(beta_deg)
    
    # Replace the path to the AeroDyn exe and drv file based on your file location
    path_exe = "C:/Users/ghhh7/AeroDyn_v15/bin/AeroDyn_Driver_x64.exe"
    path_drv = "C:/Users/ghhh7/AeroDyn_v15/TLPmodel/5MW_TLP_DLL_WTurb_WavesIrr_WavesMulti/5MW_TLP_DLL_WTurb_WavesIrr_Aero.dvr"
    
    # Update the driver file with desired input case to analysis
    with open(path_drv, 'r') as file:
        lines = file.readlines() 

    # Replace line 22 for input
    lines[21] = v_instr + " 0.00E+00 " + omega_Rstr + " " + beta_str + " 0.00E+00 0.1 0.1\n"

    # Open the file in write mode and overwrite it with the new content
    with open(path_drv, 'w') as file:
        file.writelines(lines)
    
    # Execute the driver 
    os.system(path_exe + " " + path_drv)
    
    # Read output file
    with open("LTP.1.out", 'r') as file:
        lines = file.readlines()
        data = lines[8]
        data_list = data.split()
    
    return float(data_list[4]), float(data_list[6])


def genWind(v_w, end_time, time_step):
    """
    Use Turbsim to generate a wind with turbulence.

    Parameters
    ----------
    v_w : float
        the average wind speed
    end_time : float
        the time to analysis. Should be consistent with the model driver
    time_step : float
        the time step to analysis. Should be consistent with the model driver

    Returns
    -------
    horSpd : list
        A list of horizontal wind speed computed at each time step

    """
    if end_time < 600:
        end_time = 600
        
    # Generate seeds for random wind model
    seed1 = np.random.randint(-2147483648, 2147483648)
    seed2 = np.random.randint(-2147483648, 2147483648)
    seed = [seed1, seed2]
    
    # Replace the path to the Turbsim exe and inp file based on your file location
    path_exe = "C:/Users/ghhh7/Turbsim/TurbSim.exe"
    path_inp = "C:/Users/ghhh7/Turbsim/myModel/vonKarm_15.inp"
    
    # Open the inp file and overwrite with given parameters
    with open(path_inp, 'r') as file:
        lines = file.readlines()
        
    # Overwrite with new seeds
    for i in [3, 4]:
        line = lines[i].split()
        line[0] = str(seed[i - 3])
        lines[i] = ' '.join(line) + '\n'
    
    # Overwrite "AnalysisTime" and "UsableTime"
    for i in [20, 21]:
        line = lines[i].split()
        line[0] = str(end_time)
        lines[i] = ' '.join(line) + '\n'
    
    # Overwrite the "TimeStep "
    line = lines[19].split()
    line[0] = str(time_step)
    lines[19] = ' '.join(line) + '\n'
    
    # Overwrite the average reference wind velocity
    line = lines[36].split()
    line[0] = str(v_w)
    lines[36] = ' '.join(line) + '\n'
    
    # Update the input file
    with open(path_inp, 'w') as file:
        file.writelines(lines)
    
    # Run the Turbsim to generate wind
    os.system(path_exe + " " + path_inp)
    
    # Read the output file
    path_hh = "C:/Users/ghhh7/Turbsim/myModel/vonKarm_15.hh"
    
    with open(path_hh, 'r') as file:
        lines = file.readlines()
    
    # Skip the header
    data = lines[8:]
    
    horSpd = []

    for line in data:
        columns = line.split()
        horSpd.append(float(columns[1]))  

    return horSpd



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
    '''
    wave_eta = 0
    v_x = 0
    v_y = 0
    a_x = 0
    a_y = 0
    
    for i in range(N):
        random_phase = 2*np.pi*random.random()
        wave_eta += a[i]*np.sin(omega[i]*t - k[i]*zeta + random_phase)
        v_x = omega[i]*a[i]*np.exp(k[i]*eta)*np.sin(omega[i]*t - k[i]*zeta + random_phase)
        v_y = omega[i]*a[i]*np.exp(k[i]*eta)*np.cos(omega[i]*t - k[i]*zeta + random_phase)
        a_x = omega[i]**2*a[i]*np.exp(k[i]*eta)*np.cos(omega[i]*t - k[i]*zeta + random_phase)
        a_y = -omega[i]**2*a[i]*np.exp(k[i]*eta)*np.sin(omega[i]*t - k[i]*zeta + random_phase)

    return wave_eta, [v_x, v_y, a_x, a_y]
    '''


def structure(x_1, beta, omega_R, t, Cp_type, performance, v_w, v_aveg, random_phases):
    """
    The structure of the Betti model

    Parameters
    ----------
    x_1 : np.array
        The state vector: [zeta v_zeta eta v_eta alpha omega]^T
    beta : float
        The blade pitch angle
    omega_R : double
        Rotor speed
    t : float
        Time
    Cp_type : int
        The mode to compute the power and thrust coefficient. 
        (0: read file; 1: use AeroDyn v15)
    performance: list
        Used when Cp_type = 0. The rotor performance parameter pass to CpCtCq(TSR, beta, performance)
    v_w: float
        The wind speed with turbulent
    v_aveg: float
        The average wind speed used to compute wave
    random_phase: Numpy Array
        The random parameter used to compute wave

    Returns
    -------
    np.linalg.inv(E) @ F: Numpy Array
        The derivative for the state vector
    v_in : float
        The relative wind speed
    Cp : float
        The power coefficient

    """
    
    zeta = x_1[0] # surge (x) position
    v_zeta = x_1[1] # surge velocity
    eta = x_1[2] # heave (y) position
    v_eta = x_1[3] # heave velocity
    alpha = x_1[4] # pitch position
    omega = x_1[5] # pitch velocity    
    
    g = 9.80665  # (m/s^2) gravity acceleration
    rho_w = 1025  # (kg/m^3) water density

    # Coefficient matrix E
    # Constants and parameters
    M_N = 240000  # (kg) Mass of nacelle
    M_P = 110000  # (kg) Mass of blades and hub
    M_S = 8947870  # (kg) Mass of "structure" (tower and floater)
    m_x = 11127000  # (kg) Added mass in horizontal direction
    m_y = 1504400  # (kg) Added mass in vertical direction

    d_Nh = -1.8  # (m) Horizontal distance between BS and BN
    d_Nv = 126.9003  # (m) Vertical distance between BS and BN
    d_Ph = 5.4305  # (m) Horizontal distance between BS and BP
    d_Pv = 127.5879  # (m) Vertical distance between BS and BP

    J_S = 3.4917*10**9 # (kg*m^2) "Structure" moment of inertia
    J_N = 2607890  # (kg*m^2) Nacelle moment of inertia
    J_P = 50365000  # (kg*m^2) Blades, hub and low speed shaft moment of inertia

    M_X = M_S + m_x + M_N + M_P
    M_Y = M_S + m_y + M_N + M_P
    
    d_N = np.sqrt(d_Nh**2 + d_Nv**2)
    d_P = np.sqrt(d_Ph**2 + d_Pv**2)

    M_d = M_N*d_N + M_P*d_P
    J_TOT = J_S + J_N + J_P + M_N*d_N**2 + M_P*d_P**2

    E = np.array([[1, 0, 0, 0, 0, 0],
         [0, M_X, 0, 0, 0, M_d*np.cos(alpha)],
         [0, 0, 1, 0, 0, 0],
         [0, 0, 0, M_Y, 0, M_d*np.sin(alpha)],
         [0, 0, 0, 0, 1, 0],
         [0, M_d*np.cos(alpha), 0, M_d*np.sin(alpha), 0, J_TOT]]) 

    #####################################################################
    # Force vector F
    
    h = 200  # (m) Depth of water
    h_pt = 47.89  # (m) Height of the floating structure
    r_g = 9  # (m) Radius of floater
    d_Sbott = 10.3397  # (m) Vertical distance between BS and floater bottom
    r_tb = 3  # (m) Maximum radius of the tower
    d_t = 10.3397  # (m) Vertical distance between BS and hooks of tie rods
    l_a = 27  # (m) Distance between the hooks of tie rods
    l_0 = 151.73  # (m) Rest length of tie rods
    
    K_T1 = 2*(1.5/l_0)*10**9  # (N/m) Spring constant of lateral tie rods
    K_T2 = 2*(1.5/l_0)*10**9  # (N/m) Spring constant of lateral tie rods
    K_T3 = 4*(1.5/l_0)*10**9  # (N/m) Spring constant of central tie rod

    d_T = 75.7843 # (m) Vertical distance between BS and BT
    rho = 1.225 # (kg/m^3) Density of air
    C_dN = 1 # (-) Nacelle drag coefficient
    A_N = 9.62 # (m^2) Nacelle area
    C_dT = 1 # (-) tower drag coefficient
    '''
    H_delta = np.array([[-2613.44, 810.13],
                        [810.13, 1744.28]]) # (-) Coefficient for computing deltaFA
    F_delta = np.array([-22790.37, -279533.43]) # (-) Coefficient for computing deltaFA
    C_delta = 10207305.54 # (-) Coefficient for computing deltaFA
    '''
    A = 12469 # (m^2) Rotor area
    n_dg= 2 # （-） Number of floater sub-cylinders
    C_dgper = 1 # (-) Perpendicular cylinder drag coefficient
    C_dgpar = 0.006 # (-) Parallel cylinder drag coefficient
    C_dgb = 1.9 # (-) Floater bottom drag coefficient
    R = 63 # (m) Radius of rotor
    den_l = 116.027 # (kg/m) the mass density of the mooring lines
    dia_l = 0.127 # (m) the diameter of the mooring lines
    h_T = 87.6 # (m) the height of the tower
    D_T = 4.935 # (m) the main diameter of the tower

    # Weight Forces
    Qwe_zeta = 0
    Qwe_eta = (M_N + M_P + M_S)*g
    Qwe_alpha = ((M_N*d_Nv + M_P*d_Pv)*np.sin(alpha) + (M_N*d_Nh + M_P*d_Ph )*np.cos(alpha))*g

    # Buoyancy Forces
    h_w = h  # Assume the sea bed is flat and horizontal
    h_sub = min(h_w - h + eta + d_Sbott, h_pt)
    
    d_G = eta - h_sub/2
    V_g = h_sub*np.pi*r_g**2 + max((h_w - h + eta + d_Sbott) - h_pt, 0)*np.pi*r_tb**2

    Qb_zeta = 0
    Qb_eta = -rho_w*V_g*g
    Qb_alpha = -rho_w*V_g*g*d_G*np.sin(alpha)
    
    # Tie Rod Force
    
    D_x = l_a

    l_1 = np.sqrt((h - eta - l_a*np.sin(alpha) - d_t*np.cos(alpha))**2 
                  + (D_x - zeta - l_a*np.cos(alpha) + d_t*np.sin(alpha))**2)
    l_2 = np.sqrt((h - eta + l_a*np.sin(alpha) - d_t*np.cos(alpha))**2 
                  + (D_x + zeta - l_a*np.cos(alpha) - d_t*np.sin(alpha))**2)
    l_3 = np.sqrt((h - eta - d_t*np.cos(alpha))**2 + (zeta - d_t*np.sin(alpha))**2)

    f_1 = max(0, K_T1*(l_1 - l_0))
    f_2 = max(0, K_T2*(l_2 - l_0))
    f_3 = max(0, K_T3*(l_3 - l_0))

    theta_1 = np.arctan((D_x - zeta - l_a*np.cos(alpha) + d_t*np.sin(alpha))
                        /(h - eta - l_a*np.sin(alpha) - d_t*np.cos(alpha)))
    theta_2 = np.arctan((D_x + zeta - l_a*np.cos(alpha) - d_t*np.sin(alpha))
                        /(h - eta + l_a*np.sin(alpha) - d_t*np.cos(alpha)))
    theta_3 = np.arctan((zeta - d_t*np.sin(alpha))/(h - eta - d_t*np.cos(alpha)))

    v_tir = (0.5*dia_l)**2*np.pi
    w_tir = den_l*g
    b_tir = rho_w*g*v_tir
    lambda_tir = w_tir - b_tir

    Qt_zeta = f_1*np.sin(theta_1) - f_2*np.sin(theta_2) - f_3*np.sin(theta_3)
    Qt_eta = f_1*np.cos(theta_1) + f_2*np.cos(theta_2) + f_3*np.cos(theta_3) + 4*lambda_tir*l_0
    Qt_alpha = (f_1*(l_a*np.cos(theta_1 + alpha) - d_t*np.sin(theta_1 + alpha)) 
                - f_2*(l_a*np.cos(theta_2 - alpha) - d_t*np.sin(theta_2 - alpha)) 
                + f_3*d_t*np.sin(theta_3 - alpha) + lambda_tir*l_0
                *(l_a*np.cos(alpha) - d_t*np.sin(alpha)) 
                - lambda_tir*l_0*(l_a*np.cos(alpha) 
                + d_t*np.sin(alpha)) - 2*lambda_tir*l_0*d_t*np.sin(alpha))

    # Wind Force
    v_in = v_w + v_zeta + d_P*omega*np.cos(alpha)

    TSR = (omega_R*R)/v_in

    Cp = 0
    Ct = 0
    
    if Cp_type == 0:
        Cp = CpCtCq(TSR, beta, performance)[0]
        Ct = CpCtCq(TSR, beta, performance)[1]
    else:
        Cp = drvCpCtCq(omega_R, v_in, beta)[0]
        Ct = drvCpCtCq(omega_R, v_in, beta)[1]
    
    '''
    v_root = np.roots([1, v_in, v_in**2, (1 - 2*Cp)*v_in**3])
    
    v_out = None
    for i in v_root:
        if np.isreal(i):
            v_out = np.real(i)
            break

    
    deltaFA = (np.array([v_in, beta]) @ H_delta @ np.array([v_in, beta]).T 
        + F_delta @ np.array([v_in, beta]).T + C_delta)
    

    
    FA = 0.5*rho*A*(v_in**2 - v_out**2) + deltaFA
    '''
    
    FA = 0.5*rho*A*Ct*v_in**2
    FAN = 0.5*rho*C_dN*A_N*np.cos(alpha)*(v_w + v_zeta + d_N*omega*np.cos(alpha))**2
    FAT = 0.5*rho*C_dT*h_T*D_T*np.cos(alpha)*(v_w + v_zeta + d_T*omega*np.cos(alpha))**2
    
    Qwi_zeta = -(FA + FAN + FAT)
    Qwi_eta = 0
    Qwi_alpha = (-FA*(d_Pv*np.cos(alpha) - d_Ph*np.sin(alpha))
                 -FAN*(d_Nv*np.cos(alpha) - d_Nh*np.sin(alpha))
                 -FAT*d_T*np.cos(alpha))
    
    # Wave and Drag Forces
    h_pg = np.zeros(n_dg)
    v_per = np.zeros(n_dg) # v_perpendicular relative velocity between water and immersed body
    v_par = np.zeros(n_dg) # v_parallel relative velocity between water and immersed body
    a_per = np.zeros(n_dg) # a_perpendicular acceleration of water
    tempQh_zeta = np.zeros(n_dg)
    tempQh_eta = np.zeros(n_dg)
    tempQwa_zeta = np.zeros(n_dg)
    tempQwa_eta = np.zeros(n_dg)
    Qh_zeta = 0
    Qh_eta = 0
    Qwa_zeta = 0
    Qwa_eta = 0
    Qh_alpha = 0
    Qwa_alpha = 0
    
    v_x = [0, 0]
    v_y = [0, 0]
    a_x = [0, 0]
    a_y = [0, 0]
    height = [0, 0]
    
    for i in range(n_dg):
        
        h_pg[i] = (i + 1 - 0.5)*h_sub/n_dg
        height[i] = -(h_sub - h_pg[i])
        
        wave = pierson_moskowitz_spectrum(v_aveg, zeta, height[i], t, random_phases)[1]
        
        v_x[i] = wave[0]
        v_y[i] = wave[1]
        a_x[i] = wave[2]
        a_y[i] = wave[3]
        
        v_per[i] =  ((v_zeta + (h_pg[i] - d_Sbott)*omega*np.cos(alpha) - v_x[i])*np.cos(alpha)
                     + (v_eta + (h_pg[i] - d_Sbott)*omega*np.sin(alpha) - v_y[i])*np.sin(alpha))
        v_par[i] =  ((v_zeta + (h_pg[i] - d_Sbott)*omega*np.cos(alpha) - v_x[i])*np.sin(-alpha)
                    + (v_eta + (h_pg[i] - d_Sbott)*omega*np.sin(alpha) - v_y[i])*np.cos(alpha))
        a_per[i] = a_x[i]*np.cos(alpha) + a_y[i]*np.sin(alpha)
        
        tempQh_zeta[i] = (-0.5*C_dgper*rho_w*2*r_g*(h_sub/n_dg)*  np.abs(v_per[i])*v_per[i]*np.cos(alpha)
                        - 0.5*C_dgpar*rho_w*np.pi*2*r_g*(h_sub/n_dg)*  np.abs(v_par[i])*v_par[i]*np.sin(alpha))
        tempQh_eta[i] = (-0.5*C_dgper*rho_w*2*r_g*(h_sub/n_dg)* np.abs(v_per[i])*v_per[i]*np.sin(alpha)
                         - 0.5*C_dgpar*rho_w*np.pi*2*r_g*(h_sub/n_dg)* np.abs(v_par[i])*v_par[i]*np.cos(alpha))
        tempQwa_zeta[i] = (rho_w*V_g + m_x)*a_per[i]*np.cos(alpha)/n_dg
        tempQwa_eta[i] =  (rho_w*V_g + m_x)*a_per[i]*np.sin(alpha)/n_dg
        
        Qh_zeta += tempQh_zeta[i] 
        Qh_eta += tempQh_eta[i] 
        Qwa_zeta += tempQwa_zeta[i]
        Qwa_eta += tempQwa_eta[i]
        Qh_alpha += (tempQh_zeta[i]*(h_pg[i] - d_Sbott)*np.cos(alpha)
                    + tempQh_eta[i]*(h_pg[i] - d_Sbott)*np.sin(alpha))
        Qwa_alpha += (tempQwa_zeta[i]*(h_pg[i] - d_Sbott)*np.cos(alpha)
                    + tempQwa_eta[i]*(h_pg[i] - d_Sbott)*np.sin(alpha))
    
    Qh_zeta -= 0.5*C_dgb*rho_w*np.pi*r_g**2*np.abs(v_par[0])*v_par[0]*np.sin(alpha)
    Qh_eta -= 0.5*C_dgb*rho_w*np.pi*r_g**2*np.abs(v_par[0])*v_par[0]*np.cos(alpha)

    # net force in x DOF
    Q_zeta = Qwe_zeta + Qb_zeta + Qt_zeta + Qh_zeta + Qwa_zeta + Qwi_zeta + Qh_zeta# 
    # net force in y DOF
    Q_eta = Qwe_eta + Qb_eta + Qt_eta + Qh_eta + Qwa_eta + Qwi_eta + Qh_eta
    # net torque in pitch DOF
    Q_alpha = Qwe_alpha + Qb_alpha + Qt_alpha + Qh_alpha + Qwa_alpha + Qh_alpha + Qwi_alpha

    F = np.array([v_zeta, 
                  Q_zeta + M_d*omega**2*np.sin(alpha), 
                  v_eta, 
                  Q_eta - M_d*omega**2*np.cos(alpha), 
                  omega, 
                  Q_alpha])

    return np.linalg.inv(E) @ F, v_in, Cp



def WindTurbine(omega_R, v_in, beta, T_E, t, Cp):
    """
    The drivetrain model 

    Parameters
    ----------
    omega_R : float
        The rotor speed
    v_in : float
        The relative wind speed
    beta : float
        The blade pitch angle
    T_E : float
        The generator torque
    t : float
        Time
    Cp : float
        The power coefficient

    Returns
    -------
    domega_R: float
        The derivative of rotor speed

    """
    
    # Constants and parameters
    J_G = 534.116 # (kg*m^2) Total inertia of electric generator and high speed shaft
    J_R = 35444067 # (kg*m^2) Total inertia of blades, hub and low speed shaft
    rho = 1.225 # (kg/m^3) Density of air
    A = 12469 # (m^2) Rotor area
    eta_G = 97 # (-) Speed ratio between high and low speed shafts
    
    tildeJ_R = eta_G**2*J_G + J_R
    tildeT_E = eta_G*T_E
    
    P_wind = 0.5*rho*A*v_in**3

    P_A = P_wind*Cp

    T_A = P_A/omega_R
    domega_R = (1/tildeJ_R)*(T_A - tildeT_E)
    
    return domega_R
    

def Betti(x, t, beta, T_E, Cp_type, performance, v_w, v_aveg, random_phases):
    """
    Combine the WindTurbine model and structure model
    
    Parameters
    ----------
    x : np.array
        the state vector: [zeta, v_zeta, eta, v_eta, alpha, omega, omega_R]^T
    t : float
        time
    beta : float
        blade pitch angle
    T_E : float
        generator torque
    Cp_type : int
        The mode to compute the power and thrust coefficient. 
        (0: read file; 1: use AeroDyn v15)
    performance: list
        Used when Cp_type = 0. The rotor performance parameter pass to CpCtCq(TSR, beta, performance)
    v_w: float
        The wind speed with turbulent
    v_aveg: float
        The average wind speed used to compute wave
    random_phase: Numpy Array
        The random parameter used to compute wave

    Returns
    -------
    dxdt : Numpy Array
        The derivative of the state vector

    """
    x1 = x[:6]
    omega_R = x[6]
    
    dx1dt, v_in, Cp = structure(x1, beta, omega_R, t, Cp_type, performance, v_w, v_aveg, random_phases)
    dx2dt = WindTurbine(omega_R, v_in, beta, T_E, t, Cp)
    dxdt = np.append(dx1dt, dx2dt)
    
    return dxdt



def rk4(Betti, x0, t0, tf, dt, beta, T_E, Cp_type, performance, v_w, v_wind):
    """
    Solve the system of ODEs dx/dt = Betti(x, t) using the fourth-order Runge-Kutta method.

    Parameters:
    Betti : function
        The function to be integrated.
    x0 : np.array
        Initial conditions.
    t0 : float
        Initial time.
    tf : float
        Final time.
    dt : float
        Time step.
    beta : float
        blade pitch angle
    T_E : float
        generator torque
    Cp_type : int
        The mode to compute the power and thrust coefficient. 
        (0: read file; 1: use AeroDyn v15)
    performance: list
        Used when Cp_type = 0. The rotor performance parameter pass to CpCtCq(TSR, beta, performance)
    v_w: float
        The average wind speed
    wind: wind_mutiprocessing
        Used to for simulaton mutiprocessing. Its field containing the wind speed turbulent
        for all simulations
    
    Returns:
    t, x, v_wind[:len(t)], wave_eta
    np.array, np.array, np.array, np.raay
        Time points and corresponding values of state, wind velocities, sea surface elevation
        Each row is a state vector 
    """
    
    n = int((tf - t0) / dt) + 1
    t = np.linspace(t0, tf, n)
    x = np.empty((n, len(x0)))
    x[0] = x0
    
    #v_wind = genWind(v_w, tf, dt)
    #v_wind = wind.genWind_mutiprocessing()
    
    random_phases = 2 * np.pi * np.random.rand(400)

    count = 0

    for i in range(n - 1):
        k1 = Betti(x[i], t[i], beta, T_E, Cp_type, performance, v_wind[i], v_w, random_phases)
        k2 = Betti(x[i] + 0.5 * dt * k1, t[i] + 0.5 * dt, beta, T_E, Cp_type, performance, v_wind[i], v_w, random_phases)
        k3 = Betti(x[i] + 0.5 * dt * k2, t[i] + 0.5 * dt, beta, T_E, Cp_type, performance, v_wind[i], v_w, random_phases)
        k4 = Betti(x[i] + dt * k3, t[i] + dt, beta, T_E, Cp_type, performance, v_wind[i], v_w, random_phases)
        x[i + 1] = x[i] + dt * (k1 + 2*k2 + 2*k3 + k4) / 6
        
        # Convert pitch anlge, velocity to deg and deg/s, rotor speed to rpm
        x[i][4] = np.rad2deg(x[i][4])
        x[i][5] = np.rad2deg(x[i][5])
        x[i][6] = (60 / (2*np.pi)) * x[i][6]
        x[i][2] = x[i][2] - 40.612
        x[i][0] = - x[i][0]
        
        count += 1
        #print(count)

    x[-1][4] = np.rad2deg(x[-1][4])
    x[-1][5] = np.rad2deg(x[-1][5])
    x[-1][6] = (60 / (2*np.pi)) * x[-1][6]
    x[-1][2] = x[-1][2] - 40.612
    x[-1][0] = - x[-1][0]            

    # Output wave elevation at zeta = 0
    wave_eta = []
    for i in t:
        wave_eta.append(pierson_moskowitz_spectrum(v_w, 0, 0, i, random_phases)[0])

    return t, x, v_wind[:len(t)], wave_eta



def main(end_time, v_w, v_wind, time_step = 0.01, Cp_type = 0):
    """
    Cp computation method

    Parameters
    ----------
    Cp_type : TYPE, optional
        DESCRIPTION. The default is 0.
        0: read the power coefficient file. Fast but not very accurate
        1: run the AeroDyn 15 driver, very accurate and very slow

    Returns
    -------
    t: np.array
        The time array
    x: 2D array:
        The state at each time.The row of x corresponding to each time step.
        The column is each state [surge, surge_velocity, heave, heave_velocity, pitch, pitch_rate, rotor_speed]
    v_wind: list
        The wind speed at each time step
    wave_eta: list
        The wave elevation at surge = 0 for each time step
    """
    '''
    CPU_start = time.process_time()
    start = time.time()
    '''
    
    performance = process_rotor_performance()
    
    start_time = 0
    
    # modify this to change initial condition
    #[zeta, v_zeta, eta, v_eta, alpha, omega, omega_R]
    x0 = np.array([0, 0, 40.612 - 3, 0, 0, 0, 1])

    # modify this to change run time and step size
    #[Betti, x0 (initial condition), start time, end time, time step, beta, T_E]
    t, x, v_wind, wave_eta = rk4(Betti, x0, start_time, end_time, time_step, 0.4, 43000, Cp_type, performance, v_w, v_wind)
    
    # return the output to be ploted
    return t, x, v_wind, wave_eta
    '''
    state_names = ['Surge (m)', 'Surge Velocity (m/s)', 'Heave (m)', 'Heave Velocity (m/s)', 
                   'Pitch Angle (deg)', 'Pitch Rate (deg/s)', 'Rotor speed (rpm)']
    
    
    # Plotting the wind velocity
    plt.figure() 
    plt.plot(t, v_wind)
    plt.xlabel('Time')
    plt.ylabel('Wind Speed')
    plt.title('Time evolution of Wind Speed')
    plt.grid(True)
    plt.xlim(end_time -1000, end_time)
    plt.savefig('wind_speed.png', dpi=600)  
    plt.show()
    
    # Plotting the sea surface elevation
    plt.figure()
    plt.plot(t, wave_eta)
    plt.title('Time evolution of Sea Surface Elevation at position x')
    plt.xlabel('Time (s)')
    plt.ylabel('Elevation (m)')
    plt.grid(True)
    plt.xlim(end_time - 1000, end_time)
    plt.savefig('wave.png', dpi=600) 
    plt.show()
    
    for i in range(x.shape[1]):
        plt.figure()  # create a new figure for each state
        plt.plot(t, x[:, i])
        plt.xlabel('Time')
        plt.ylabel(f'{state_names[i]}')
        plt.title(f'Time evolution of {state_names[i]}')
        plt.grid(True)
        plt.xlim(end_time - 1000, end_time)
        safe_filename = state_names[i].replace('/', '_')  
        plt.savefig(f'{safe_filename}.png', dpi=600)  
        plt.show()
        
    CPU_end = time.process_time()
    end = time.time()
    
    print("CPU time: ", CPU_end - CPU_start, "seconds")
    print("time: ", end - start, "seconds")
    '''


def run_simulation(params):
    return main(*params)


def run_simulations_parallel(n_simulations, params):
    
    vWind = []
    for i in range(n_simulations):
        vWind.append(genWind(v_w, params[0], 0.01))
        
    with Pool() as p:
        
        all_params = [params + [vWind[i]] for i in range(n_simulations)]
        
        results = p.map(run_simulation, all_params)
    

    color = (0, 0, 1, 0.2)

    
    # Plotting the wind velocity
    plt.figure() 
    for i, result in enumerate(results):
        t, _, v_wind, _ = result
        plt.plot(t, v_wind, color=color)
    plt.xlabel('Time')
    plt.ylabel('Wind Speed')
    plt.title('Time evolution of Wind Speed')
    plt.grid(True)
    plt.xlim(0, end_time)
    plt.savefig('wind_speed.png', dpi=2000)  
    plt.show()

    
    # Plotting the sea surface elevation
    plt.figure()
    for i, result in enumerate(results):
        t, _, _, wave_eta = result
        plt.plot(t, wave_eta, color=color)
    plt.title('Time evolution of Sea Surface Elevation at position x')
    plt.xlabel('Time (s)')
    plt.ylabel('Elevation (m)')
    plt.grid(True)
    plt.xlim(0, end_time)
    plt.savefig('wave.png', dpi=2000) 
    plt.show()

    '''
    end_time = params[0]

    state_names = ['Surge (m)', 'Surge Velocity (m/s)', 'Heave (m)', 'Heave Velocity (m/s)', 
                   'Pitch Angle (deg)', 'Pitch Rate (deg/s)', 'Rotor speed (rpm)']
    
    for state_index in range(len(state_names)):
        plt.figure()  # create a new figure for each state
        for i, result in enumerate(results):
            t, x, _, _ = result
            plt.plot(t, x[:, state_index], color=color)
        plt.xlabel('Time')
        plt.ylabel(f'{state_names[state_index]}')
        plt.title(f'Time evolution of {state_names[state_index]}')
        plt.grid(True)
        plt.xlim(0, end_time)
        safe_filename = state_names[state_index].replace('/', '_')  
        plt.savefig(f'{safe_filename}.png', dpi=2000)  
        plt.show()
    '''
    return results


def plot_quantiles(results, end_time):
    
    t = results[0][0]
    
    # Only take the states part to analyze
    state = np.stack([t[1] for t in results], axis=2)
    
    # Get the central 75%
    percentile_87_5 = np.percentile(state, 87.5, axis=2)
    percentile_12_5 = np.percentile(state, 12.5, axis=2)
    
    # Get the central 25%
    percentile_62_5 = np.percentile(state, 62.5, axis=2)
    percentile_37_5 = np.percentile(state, 37.5, axis=2)
    
    # Get the median (50%)
    percentile_50 = np.percentile(state, 50, axis=2)
    
    state_names = ['Surge (m)', 'Surge Velocity (m/s)', 'Heave (m)', 'Heave Velocity (m/s)', 
                   'Pitch Angle (deg)', 'Pitch Rate (deg/s)', 'Rotor speed (rpm)']
    
    start_time = 0
    
    if end_time > 1000:
        start_time = end_time - 1000
    
    for i in range(7):
        plt.figure()
        plt.fill_between(t, percentile_12_5[:, i], percentile_87_5[:, i], color='b', alpha=0.3)
        plt.fill_between(t, percentile_37_5[:, i], percentile_62_5[:, i], color='b', alpha=1)
        plt.plot(t, percentile_50[:, i], color='r', linewidth=1) 
        plt.xlabel('Time')
        plt.ylabel(f'{state_names[i]}')
        plt.title(f'Time evolution of {state_names[i]}')
        plt.grid(True)
        plt.xlim(start_time, end_time)
        safe_filename = state_names[i].replace('/', '_')  
        plt.savefig(f'{safe_filename}.png', dpi=2000)  
        plt.show()
        plt.close()
        
        plt.figure()
        plt.fill_between(t, percentile_12_5[:, i], percentile_87_5[:, i], color='b', alpha=0.3)
        plt.fill_between(t, percentile_37_5[:, i], percentile_62_5[:, i], color='b', alpha=1)
        plt.plot(t, percentile_50[:, i], color='r', linewidth=1) 
        plt.xlabel('Time')
        plt.ylabel(f'{state_names[i]}')
        plt.title(f'Time evolution of {state_names[i]}')
        plt.grid(True)
        plt.xlim(end_time - 30, end_time)
        safe_filename = state_names[i].replace('/', '_')  
        short = ' 30s'
        plt.savefig(f'{safe_filename + short}.png', dpi=2000)  
        plt.show()
        plt.close()
        
'''

class wind_mutiprocessing:
    
    def __init__(self, v_w, end_time, n_simulations, time_step=0.01 ):
        """
        This function generate wind for n simulations before fun the model.
        It will store the wind generated for each simulation in a 2D list.
        """
        self.index = 0
        self.vWind = []
        for i in range(n_simulations):
            self.vWind.append(genWind(v_w, end_time, time_step))
            
    
    def genWind_mutiprocessing(self):
        """
        Everytime this function get called, it will return on wind in vWind_parallel
        and remove that wind from vWind_parallel.

        Parameters
        ----------
        v_w : TYPE
            DESCRIPTION.
        end_time : TYPE
            DESCRIPTION.
        time_step : TYPE
            DESCRIPTION.
        n_simulations : TYPE
            DESCRIPTION.

        Returns
        -------
        None.

        """
        wind = self.vWind[self.index]
        self.index += 1
        return wind

'''
    


########################################
###############################################################################
        


if __name__ == '__main__':
    
    v_w = 20
    end_time = 700
    n_simulations = 50

    params = [end_time, v_w]
    
    results = run_simulations_parallel(n_simulations, params)
    plot_quantiles(results, params[0])
    


#performance = process_rotor_performance()

#print(CpCtCq(-10.08, -5, performance), drvCpCtCq(1.2566, 12, 0))




