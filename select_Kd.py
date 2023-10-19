# -*- coding: utf-8 -*-
"""
Created on Mon Sep 18 20:59:58 2023

@author: ghhh7
"""
import numpy as np
import matplotlib.pyplot as plt

k_d = np.linspace(0.02, 1, 200)

data = np.load('kd_stat.npy')

std = data[:,2]

print(min(std))

plt.figure()
plt.plot(k_d, std)
plt.xlabel("Derivative gain kd")
plt.ylabel("Standard Deviation of Roter Speed")
plt.title("Standard Deviation of Controlled Rotor Speed Respect to Derivative Gain")
plt.xlim(0.02, 1)
plt.grid(True)
plt.tight_layout()
plt.show()
plt.savefig('std_kd.png', dpi=300)
plt.close()

plt.figure()
plt.plot(k_d, std)

# Find the minimum value and its corresponding k_d
min_std = np.min(std)
min_kd_idx = np.argmin(std)
min_kd = k_d[min_kd_idx]

# Adding a vertical line at the minimum point
plt.axvline(x=min_kd, color='r', linestyle='--')
# Adding a horizontal line at the minimum point
plt.axhline(y=min_std, color='r', linestyle='--')

# Label the minimum value - this will put text on the plot at the specified position
plt.text(min_kd, min_std, f'Minimum\nkd={min_kd:.2f}, std={min_std:.2f}', color='black', 
         ha='right', # horizontal alignment can be 'center', 'right'
         va='bottom', # vertical alignment can be 'center', 'top'
         bbox=dict(facecolor='white', alpha=0.8)) # this puts a white background around the text for readability

# Possibly add an arrow to the text
# plt.annotate(f'Minimum\nkd={min_kd:.2f}, std={min_std:.2f}', xy=(min_kd, min_std),
#              xytext=(min_kd, min_std + some_value), # some_value depends on your data range
#              arrowprops=dict(facecolor='black', shrink=0.05))

plt.xlabel("Derivative gain kd")
plt.ylabel("Standard Deviation of Rotor Speed")
plt.title("Standard Deviation of Controlled Rotor Speed Respect to Derivative Gain")
plt.xlim(0.02, 1)
plt.grid(True)
plt.tight_layout()
plt.show()

# Save the figure. This should come right after `show()` before closing, as some backends will close the figure automatically after show().
plt.savefig('std_kd.png', dpi=300)
plt.close()

