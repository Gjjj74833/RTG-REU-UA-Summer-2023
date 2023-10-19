# -*- coding: utf-8 -*-
"""
Created on Wed Sep 20 00:40:27 2023

@author: Yihan Liu
"""
import numpy as np
import matplotlib.pyplot as plt
beta = np.linspace(0, np.pi/4)

eta_G = 97 # (-) Speed ratio between high and low speed shafts
J_G = 534.116 # (kg*m^2) Total inertia of electric generator and high speed shaft
J_R = 35444067 # (kg*m^2) Total inertia of blades, hub and low speed shaft
tildeJ_R = eta_G**2*J_G + J_R

rated_omega_R = 1.26711 # The rated rotor speed is 12.1 rpm
#rated_omega_R = 1.571
zeta_phi = 0.7
omega_phin = 0.6
beta_k = 0.1099965
dpdbeta_0 = -25.52*10**6

GK = 1/(1+(beta/beta_k))

K_p = 0.0765*(2*tildeJ_R*rated_omega_R*zeta_phi*omega_phin*GK)/(eta_G*(-dpdbeta_0))
K_i = 0.013*(tildeJ_R*rated_omega_R*omega_phin**2*GK)/(eta_G*(-dpdbeta_0))

beta = np.rad2deg(beta)

# Create a figure and a 2x1 grid of subplots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 3.5))

# First subplot for K_p
ax1.plot(beta, K_p, label='Kp', color='blue')
ax1.set_ylabel("Kp")
ax1.set_xlim(0, 45)
ax1.grid(True)
ax1.set_title("Control Gain Kp Over Blade Pitch Angle")

# Second subplot for K_i
ax2.plot(beta, K_i, label='Ki', color='red')
ax2.set_xlabel("Blade pitch angle (deg)")
ax2.set_ylabel("Ki")
ax2.set_xlim(0, 45)
ax2.grid(True)
ax2.set_title("Control Gain Ki Over Blade Pitch Angle")

plt.tight_layout()
plt.savefig('KiKp.png', dpi=300)
plt.show()
plt.close()