"""
LineProfile_Calc_V2.py
L. Matrecito, lmm8709@rit.edu
Fri Jun 16 11:11:40 2023

This code is a new version of LineProfile_Calc.py. It contains a the same
analyzation, but presents it more efficiently and user-friendly. For more
information, see LineProfile_Calc.py.
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate as integrate
import pandas as pd

# Results from Fitting-Script.py are saved as plateID-MJD-fiberID_"".npy files
# Source is just the plateID-MJD-fiberID.
source = '0390-51900-0587'

# Path of stored PyQSOFit fit results
path = 'Fit Results/Line Complex Properties/'+source+'/'+'Fit Data/'


# -----------------------------------------------------------------------------

# IMPORTANT

# Remember to make a directory within the plateID-MJD-fiberID directory in the
# with Line Profile Plots as the folder name PRIOR to running the code
# or else it will not work because it will not be able to find that directory.
# A portion of this code will create those directories automatically
# and is currently a WIP!

# Path for saving results from this code
path2 = 'Fit Results/Line Complex Properties/'+source+'/'+'Line Profile Plots/'
    

# -----------------------------------------------------------------------------

# Obtaining line profile data result components from Fitting-Script.py
bl = np.load(path+source+'_BLData.npy')
bl_profile = np.load(path+source+'_BLSpectrum.npy')
nl = np.load(path+source+'_NLData.npy')
data = np.load(path+source+'_Data.npy')             # Flux from SDSS .fits file
data_contFeII_sub = np.load(path+source+'_DataCFeII.npy')  
wavelength = np.load(path+source+'_Wavelength.npy')

# Converting .npy files into 2D dataframes to make velocity shift calculations
# MUCH easier
# BL Data
bl_matrix = np.vstack((wavelength, bl)).T
bl_data = pd.DataFrame(bl_matrix, columns=['Wavelength','Flux'])
# NL Data
nl_matrix = np.vstack((wavelength, nl)).T
nl_data = pd.DataFrame(nl_matrix, columns=['Wavelength','Flux'])


# -----------------------------------------------------------------------------

# Defining functions for calculating velocity shifts

# Part 1: finds the peak velocity shift by obtaining the corresponding
# wavelength of the max flux value of the BL profile. User specifies start and
# end of line complex in addition to the narrow line vacuum wavelength being 
# analyzed 

def calc_pvs(data, start_wave, end_wave, vac_wave):
    b_data = bl_data.loc[(data['Wavelength'] >= start_wave) & (data['Wavelength'] <= end_wave)]
    
    b_wave = b_data['Wavelength']
    b_flux = b_data['Flux']
    
    peak_b_flux = b_flux.max()
    
    
    peak_b_wave = b_data.loc[b_data['Flux'] == peak_b_flux]['Wavelength'].iloc[0]
    peak_b = int(peak_b_wave['Wavelength'])
    
    



