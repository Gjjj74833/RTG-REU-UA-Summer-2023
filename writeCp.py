# -*- coding: utf-8 -*-
"""
Created on Sat Jul 15 14:09:31 2023
Write the power coefficient file

@author: Yihan Liu 
"""

import numpy as np
import os
from Betti import AeroCp

def write(pitch_step = 0.1, omega_step = 0.01, v_w = 10):
    
    with open('Cp_Ct_Cq.NREL5MW.txt', 'w') as file:
        
        R = 63 # (m) Radius of rotor 
        
        # Create a list including all pitch angles to compute
        pitch_angle = np.arange(-5, 45.1, pitch_step).tolist()
        pitch_str = ' '.join("{:.2f}".format(num) for num in pitch_angle)
        
        # Create a list including all rotor speed to compute
        rotor_speed = np.arange(0, 2.619, omega_step).tolist()
        TSR = []
        for i in rotor_speed:
            TSR.append((i*R)/v_w)
        TSR_str = ' '.join("{:.6f}".format(num) for num in TSR)
        
        # write header
        file.write("# ----- Rotor performance tables for the NREL-5MW wind turbine ----- \n")
        file.write("# ------------ Written on Jan-13-22 using the ROSCO toolbox ------------ \n")
        file.write(" \n")
        file.write("# Pitch angle vector, 36 entries - x axis (matrix columns) (deg) \n")
        file.write(pitch_str + "\n")
        file.write("# TSR vector, 26 entries - y axis (matrix rows) (-) \n")
        file.write(TSR_str + "\n")
        file.write("# Wind speed vector - z axis (m/s) \n")
        file.write(str(v_w) + "\n")
        file.write(" \n")
        file.write("# Power coefficient \n")
        file.write(" \n")
        
        # Run the AeroDyn v15 driver and write the power coefficient by row
        count = [0, 0]
        # For each row (rotor speed value)
        for i in rotor_speed:
            line = []
            # For each column (pitch angle value)
            for j in pitch_angle:
                Cp = AeroCp(i, v_w, np.deg2rad(j))
                line.append(Cp)
                count[1] += 1
                print(count)
            # Write the line of data
            line_str = ' '.join("{:.6f}".format(num) for num in line)
            file.write(line_str + "\n")
            
            count[0] += 1
            count[1] = 0
                            
            

write()
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        