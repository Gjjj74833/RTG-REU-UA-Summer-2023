# -*- coding: utf-8 -*-
"""
Betti model implementation

@author: Yihan Liu
@version (2023-06-24)
"""

import numpy as np
import matplotlib.pyplot as plt
import os


# The model

def process_rotor_performance(input_file = "Cp_Ct_Cq.NREL5MW.txt"):
    """
    This function will read the power coefficient surface from a text file generated
    by simulation and store the power coefficient in a 2D list

    Parameters
    ----------
    input_file : String, optional
        The file name of the pwer coefficient

    Returns
    -------
    C_p : 2D list
        The power coefficient. col: pitch angle, row: TSR value
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
            c_row = [float(num_str) for num_str in lines[i].split()]
            C_p.append(c_row)

    return C_p, pitch_angles, TSR_values

    

def C_p(TSR, beta, performance):
    """
    Find the power coefficient based on the given TSR value and pitch angle

    Parameters
    ----------
    TSR : Tip speed ratio
    beta : blade pitch angle

    Returns
    -------
    corresponding power coefficient
    """
    beta = np.rad2deg(beta)

    C_p = performance[0] 
    pitch_list = performance[1] 
    TSR_list = performance[2]
    
    # Find the closed pitch and TSR value in the list
    pitch_close = min(pitch_list, key=lambda x: abs(x - beta))
    TSR_close = min(TSR_list, key=lambda x: abs(x - TSR) )
    
    # Get the index of the closest pitch and TSR value
    pitch_index = pitch_list.index(pitch_close)
    TSR_index = TSR_list.index(TSR_close)
    
    # Get the C_p value at the index 
    return C_p[TSR_index][pitch_index]


def Cp(omega_R, v_in, beta):
    """
    Compute the power coefficient. Based on input requirements for 
    AeroDyn v15, we take the rotor speed and relative wind velocity as
    parameters instead of TSR. there's no need to compute TSR when use 
    this function to calculate Cp.

    Parameters
    ----------
    omega_R : float
        The rotor speed in rad/s
    v_in : float
        relative wind speed
    beta : float
        blade pitch angle

    Returns
    -------
    float
        the power coefficient

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
    
    return float(data_list[4])



def structure(x_1, beta, omega_R, t, Cp_type, performance):
    """
    The structure of the Betti model

    Parameters
    ----------
    x_1 : np.array
        x_1 = [zeta v_zeta eta v_eta alpha omega]^T
    v_w : float
        the wind speed
    d_wave : 2D np.array
        the wave model
    beta : float
        blade pitch angle
    omega_R : double
        rotor speed
    t : float
        time

    Returns
    -------
    TYPE
        DESCRIPTION.
    v_in : TYPE
        DESCRIPTION.

    """
    """
    x_1 = [zeta v_zeta eta v_eta alpha omega]^T
    v_w = wind speed
    beta = blade pitch angle
    """
    # For test, consider constant wind and no wave
    v_w = 23
    
    zeta = x_1[0] # surge (x) position
    v_zeta = x_1[1] # surge velocity
    eta = x_1[2] # heave (y) position
    v_eta = x_1[3] # heave velocity
    alpha = x_1[4] # pitch position
    omega = x_1[5] # pitch velocity
    
    v_x = [0, 0]
    v_y = [0, 0]
    a_x = [0, 0]
    a_y = [0, 0]
    

    
    
    
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
    H_delta = np.array([[-2613.44, 810.13],
                        [810.13, 1744.28]]) # (-) Coefficient for computing deltaFA
    F_delta = np.array([-22790.37, -279533.43]) # (-) Coefficient for computing deltaFA
    C_delta = 10207305.54 # (-) Coefficient for computing deltaFA
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

    if Cp_type == 0:
        v_root = np.roots([1, v_in, v_in**2, (1 - 2*C_p(TSR, beta, performance))*v_in**3])
    else:
        v_root = np.roots([1, v_in, v_in**2, (1 - 2*Cp(omega_R, v_in, beta))*v_in**3])
    v_out = None
    for i in v_root:
        if np.isreal(i):
            v_out = np.real(i)
            break

    
    deltaFA = (np.array([v_in, beta]) @ H_delta @ np.array([v_in, beta]).T 
        + F_delta @ np.array([v_in, beta]).T + C_delta)
    

    
    FA = 0.5*rho*A*(v_in**2 - v_out**2) + deltaFA
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
    
    for i in range(n_dg):
        h_pg[i] = (i + 1 - 0.5)*h_sub/n_dg
        
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

    return np.linalg.inv(E) @ F, v_in



def WindTurbine(omega_R, v_in, beta, T_E, t, Cp_type, performance):
    # Constants and parameters
    J_G = 534.116 # (kg*m^2) Total inertia of electric generator and high speed shaft
    J_R = 35444067 # (kg*m^2) Total inertia of blades, hub and low speed shaft
    rho = 1.225 # (kg/m^3) Density of air
    A = 12469 # (m^2) Rotor area
    R = 63 # (m) Radius of rotor
    eta_G = 97 # (-) Speed ratio between high and low speed shafts
    
    tildeJ_R = eta_G**2*J_G + J_R
    tildeT_E = eta_G*T_E
    
    TSR = (omega_R*R)/v_in
    
    P_wind = 0.5*rho*A*v_in**3
    if Cp_type == 0:
        P_A = P_wind*C_p(TSR, beta, performance)
    else:
        P_A = P_wind*Cp(omega_R, np.abs(v_in), beta)
    
    T_A = P_A/omega_R
    domega_R = 1/(tildeJ_R)*(T_A - tildeT_E)
    
    return domega_R

'''
def wave():
    
    def S_PM(f):
        
        alpha_PM = 0.0081
        g = 9.80665  # (m/s^2) gravity acceleration
        
        S_PM = ((alpha_PM*g**2)/((2*np.pi)**4*f**5))*np.e**(-1.25*(f/f_PM)**(-4))
'''

def Betti(x, t, beta, T_E, Cp_type, performance):
    """
    Combine the WindTurbine model and structure model
    
    Parameters
    ----------
    x : np.array
        x_1 = [zeta, v_zeta, eta, v_eta, alpha, omega, omega_R]^T
    t : double
        time
    beta : float
        blade pitch angle
    omega_R : double
        rotor speed
    T_E : float
        the generator torque

    Returns
    -------
    TYPE
        DESCRIPTION.
    v_in : TYPE
        DESCRIPTION.

    """
    x1 = x[:6]
    omega_R = x[6]
    
    dx1dt, v_in = structure(x1, beta, omega_R, t, Cp_type, performance)
    dx2dt = WindTurbine(omega_R, v_in, beta, T_E, t, Cp_type, performance)
    dxdt = np.append(dx1dt, dx2dt)
    
    return dxdt



def rk4(Betti, x0, t0, tf, dt, beta, T_E, Cp_type, performance):
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
        Blade pitch angle.
    T_E : float
        The generator torque.
    Cp_type: int
        Cp computation method 
        0: read file
        1: run AeroDyn v15 driver
        
    Returns:
    np.array, np.array
        Time points and corresponding values of x.
        Each row is a state vector 
    """
    
    n = int((tf - t0) / dt) + 1
    t = np.linspace(t0, tf, n)
    x = np.empty((n, len(x0)))
    x[0] = x0

    count = 0

    for i in range(n - 1):
        k1 = Betti(x[i], t[i], beta, T_E, Cp_type, performance)
        k2 = Betti(x[i] + 0.5 * dt * k1, t[i] + 0.5 * dt, beta, T_E, Cp_type, performance)
        k3 = Betti(x[i] + 0.5 * dt * k2, t[i] + 0.5 * dt, beta, T_E, Cp_type, performance)
        k4 = Betti(x[i] + dt * k3, t[i] + dt, beta, T_E, Cp_type, performance)
        x[i + 1] = x[i] + dt * (k1 + 2*k2 + 2*k3 + k4) / 6
        
        # Convert pitch anlge, velocity to deg and deg/s, rotor speed to rpm
        x[i][4] = np.rad2deg(x[i][4])
        x[i][5] = np.rad2deg(x[i][5])
        x[i][6] = (60 / (2*np.pi)) * x[i][6]
        if i == n - 1:
            x[i + 1][4] = np.rad2deg(x[i + 1][4])
            x[i + 1][5] = np.rad2deg(x[i + 1][5])
            x[i + 1][6] = (60 / (2*np.pi)) * x[i + 1][6]
    
        count += 1
        print(count)

    return t, x

def main(end_time, time_step, Cp_type = 0):
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
    None.

    """
    
    performance = process_rotor_performance()
    
    start_time = 0
    
    # modify this to change initial condition
    #[zeta, v_zeta, eta, v_eta, alpha, omega, omega_R]
    x0 = np.array([0, 0, 0, 0, 0, 0, 1])

    # modify this to change run time and step size
    #[Betti, x0 (initial condition), start time, end time, time step, beta, T_E]
    t, x = rk4(Betti, x0, start_time, end_time, time_step, 0.5, 43000, Cp_type, performance)

    state_names = ['Surge (m)', 'Surge Velocity (m/s)', 'Heave (m)', 'Heave Velocity (m/s)', 
                   'Pitch Angle (deg)', 'Pitch Rate (deg/s)', 'Rotor speed (rpm)']

    for i in range(x.shape[1]):
        plt.figure()  # create a new figure for each state
        plt.plot(t, x[:, i])
        plt.xlabel('Time')
        plt.ylabel(f'{state_names[i]}')
        plt.title(f'Time evolution of {state_names[i]}')
        plt.grid(True)
        plt.xlim(0, end_time)
        plt.show()
        
        

###############################################################################
###############################################################################
        
    
main(1000, 0.01)







