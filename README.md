# RTG-REU-UA-Summer-2023

There are two drivers required to execute AeorDyn v15 which
calculate the power coefficient:
AeroDyn_v15\bin\AeroDyn_Driver_x64.exe
AeroDyn_v15\TLPmodel\5MW_TLP_DLL_WTurb_WavesIrr_WavesMulti\5MW_TLP_DLL_WTurb_WavesIrr_Aero.drv

Replace the path in Betti.py, Cp() based on the location on your computer to run.

The input files for AeroDyn v15 has a slight difference to OpenFAST and already been modified properly. Direct use files for OpenFAST as input for AeroDyn won't work.

Visaul.py is used to visualize the result produced by OpenFAST and might be helpful to compare the Betti model with OpenFAST in the future. Detailed user guide are provided in documentations of functions inside the code.
