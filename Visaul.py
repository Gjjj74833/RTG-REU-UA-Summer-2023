# -*- coding: utf-8 -*-
"""
Created on Sat Jul  8 17:03:11 2023
OpenFAST results visualization
@author: Yihan Liu
"""

import numpy as np
import matplotlib.pyplot as plt


def readOutFiles(input_file = "5MW_TLP_DLL_WTurb_WavesIrr_WavesMulti.out"):
    """
    Read the .out file produced by OpenFAST. Return a dictionary
    maps the parameter name including units to its data value
    i.e: key: "Time (s)", value = list of time value

    Parameters
    ----------
    input_file : String
        The .out file name

    Returns
    -------
    data_dict : dictionary
        the dictionary maps the parameter and data

    """
    
    header = []
    data = []
    data_list = []

    
    with open(input_file, 'r') as file:
        lines = file.readlines()
        
        # Extract the header parameters
        header_str = lines[6]
        header = header_str.split()
        
        # Extract the units
        units_str = lines[7]
        units = units_str.split()
        
        # Map the parameters with units
        header_units_map = {h: u for h, u in zip(header, units)}
        
        # Extract the data
        for line in lines[9:]:  # assuming data starts from the 8th line
            row = [float(col) for col in line.split() if col.strip() != '']
            data_list.append(row)
            
        
        data = np.array(data_list).T.tolist()
        
        data_dict = dict(zip(header, data))
        
    return data_dict, header_units_map



def visual(data_dict, header_units_map, key, key_x = "Time"):
    """
    Plot the data which key is the parameter string. The function will use
    the data in the data_dict in the key and plot in y-axis of a graph. 
    Key_x is the variable on the x-axis. By default, it is time

    Parameters
    ----------
    key : String
        The parameter name, which should be a key in data_dict
    data_dict : dictionary
        The dictionary stores data, and maps the parameter with its data

    Returns
    -------
    None.

    """
    
    # Ensure the key exists in the dictionary
    if key not in data_dict:
        print(f"The key '{key}' was not found in the data dictionary.")
        return
    if key_x not in data_dict:
        print(f"The key '{key_x}' was not found in the data dictionary.")
        return

    # Get the x and y data for the plot
    x_data = data_dict[key_x]
    y_data = data_dict[key]

    # Create the plot
    plt.figure(figsize=(10,6))
    plt.plot(x_data, y_data)
    plt.xlabel(key_x)
    plt.ylabel(key + " " + header_units_map[key])
    plt.title(f'{key} vs. {key_x}')
    plt.xlim(min(x_data), max(x_data))
    plt.grid(True)
    plt.show()
    
    
###############################################################################
###############################################################################
    
data_dict, unit_dict = readOutFiles()

visual(data_dict, unit_dict, "GenTq")


    
    

