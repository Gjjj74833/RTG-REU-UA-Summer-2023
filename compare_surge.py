# -*- coding: utf-8 -*-
"""
Created on Mon Nov 27 21:03:44 2023

@author: Yihan Liu
"""

import numpy as np
import matplotlib.pyplot as plt

t0 = 0
tf = 3000
dt = 0.01

n = int((tf - t0) / dt) + 1
surge = np.load('steady_surge.npy')[:,0]
surge_low = np.load('steady_surge_low.npy')[:,0]
t = np.linspace(t0, tf, n)


plt.figure(figsize=(6.4, 2.4))  # create a new figure for each state
plt.plot(t, surge, label='20 m/s')
plt.plot(t, surge_low, label='11 m/s')
plt.xlabel('Time (s)')
plt.ylabel('Surge (m)')
plt.title('Surge at Different Wind Speed')
plt.legend()
plt.grid(True)
plt.xlim(0, 3000)
plt.ylim(0, 5)
plt.savefig('Surge (m).png')
plt.show()
plt.close()