"""
LineProfile_Calc.py
L. Matrecito, lmm8709@rit.edu
Sun Feb 12 15:34:39 2023

This code can analyzes the fitted broad-line (BL) compnents of several line 
complexes from the results of Fitting-Script.py for PyQSOFit. It calculates the 
peak, centroid, line center at 80% of the area (C80, from Whittle 1985) 
velocity shifts. In addition, it calculates characteristic line profile shape 
using the area parameters proposed by Whittle 1985. Please look through the 
code as there are important notes (separated by blocks, titled IMPORTANT) PRIOR 
to running.
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate as integrate
import pandas as pd
#import os

# Results from Fitting-Script.py are saved as plateID-MJD-fiberID_"".npy files
# Source is just the plateID-MJD-fiberID.
source = '0545-52202-0238'

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
    
# Creating directories Line Profile Plots
#line_plots = '/Fit Results/Line Complex Properties/'+source+'/Line Profile Plots/'
#if not os.path.exists(line_plots):
#    os.makedirs(line+plots, mode=0o777)


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

# H_alpha Line Complex

# If H_alpha is not present in your spectrum (z > 0.40), the velocity shift
# calculations can be skipped by answering 'no' to "bha_vs"

bha_vs = 'no'
if(bha_vs == 'yes'):
    # Section 1: finds the peak velocity shift by obtaining the corresponding
    # wavelength of the max flux value of the BL profile
    # ---
    # BLs - change according to length of wings
    bha_data = bl_data.loc[(bl_data['Wavelength'] >= 6300) & \
                           (bl_data['Wavelength'] <= 6860)]
    bha_wave = bha_data['Wavelength']
    bha_flux = bha_data['Flux']
    # Finding peak 
    peak_bha_flux = bha_flux.max()
    peak_bha_wave = bha_data.loc[bha_data['Flux'] == peak_bha_flux]
    peak_bha = int(peak_bha_wave['Wavelength'])
    # ---
    # NLs 
    nha_data = nl_data.loc[(nl_data['Wavelength'] >= 6400) & \
                           (nl_data['Wavelength'] <=6800)]
    nha_flux = nha_data['Flux']
    # Finding peak: should be ~ vacuum wavelength and will be used for 
    # calculating all three types of velocity shifts
    peak_nha_flux = nha_flux.max()
    peak_nha_wave = nha_data.loc[nha_data['Flux'] == peak_nha_flux]
    peak_nha = int(peak_nha_wave['Wavelength'])
    #peak_nha = 6564.61 # use when H_alpha is not fitted due to heavily blended spectra
    # ---
    # Peak velocity shift 
    bha_pvs = ((peak_bha**2 - peak_nha**2) / (peak_bha**2 + peak_nha**2)) * 299792
    
    # Section 2: finds the centroid velocity shift by normalizing the BL 
    # profile and finding the cumulative distribution function (CDF) to obtain 
    # the corresponding wavelength at 50% of the CDF
    # ---
    # Normalizing BL profile and finding CDF
    bha_flux_rescaled = (bha_flux - bha_flux.min()) / (bha_flux.max() - bha_flux.min())
    bha_flux_area = integrate.trapezoid(bha_flux_rescaled, bha_wave)
    bha_flux_new = bha_flux_rescaled/bha_flux_area   
    bha_flux_norm = bha_flux_new / np.sum(bha_flux_new)
    bha_cdf = np.cumsum(bha_flux_norm)
    # ---
    # Finding centroid
    bha_ctr = np.interp(0.5, bha_cdf, bha_wave)
    # Ploting CDF 
    plot_bha_cdf = 'yes'
    if(plot_bha_cdf == 'yes'):
        fig1 = plt.figure(figsize=(16,10))
        plt.plot(bha_wave, bha_cdf, linewidth=2, c='#DB1D1A')
        plt.title(source+r': H$\alpha$ CDF', fontsize=30)
        plt.ylabel('Probability', fontsize=20)
        plt.xlabel(r'Wavelength ($\rm \AA$)', fontsize=20)
        plt.xticks(fontsize=20)
        plt.yticks(fontsize=20)
        plt.tick_params(which='major', length=16, width=2)
        plt.tick_params(which='minor', length=8, width=1)
        plt.savefig(path2+source+'_BHA_CDF.pdf')
        plt.axvline(bha_ctr, c='grey', linestyle='--', label='Centroid')
        plt.text(bha_ctr+5, 0.2, r'H$\alpha$ BL Centroid = {:.2f} $\AA$'.format(bha_ctr))
        plt.legend(fontsize=14)
    # ---
    # Centroid velocity shift
    bha_cvs = ((bha_ctr**2 - peak_nha**2) / (bha_ctr**2 + peak_nha**2)) * 299792 

    # Section 3: finding C80 shift by removing 10% of the area at each end of
    # the BL profile and obtaining the corresponding wavelength at the 
    # center.
    # ---
    bha_10 = np.interp(0.10, bha_cdf, bha_wave)
    bha_90 = np.interp(0.90, bha_cdf, bha_wave)
    bha_C80_data = bl_data.loc[(bl_data['Wavelength'] >= bha_10) & \
                               (bl_data['Wavelength'] <= bha_90)]
    bha_C80_wave = bha_C80_data['Wavelength']
    bha_C80_flux = bha_C80_data['Flux']
    # Normalizing BL profile and finding CDF
    bha_C80_flux_rescaled = (bha_C80_flux - (bha_C80_flux.min())) / \
        (bha_C80_flux.max() - bha_C80_flux.min())
    bha_C80_area = integrate.trapezoid(bha_C80_flux_rescaled, bha_C80_wave)
    bha_C80_flux_new = bha_C80_flux_rescaled/bha_C80_area
    bha_C80_flux_norm = bha_C80_flux_new / np.sum(bha_C80_flux_new)
    bha_cdf_C80 = np.cumsum(bha_C80_flux_norm)
    # ---
    # Finding line center
    bha_C80_ctr = np.interp(0.5, bha_cdf_C80, bha_C80_wave)
    # C80 velocity shift
    bha_C80_vs = ((bha_C80_ctr**2 - peak_nha**2) / (bha_C80_ctr**2 + peak_nha**2)) * 299792 
    
    # Section 4: finding the Whittle 1985 line profile parameters; i.e. inter-
    # percentile velocity width (IPV), asymmetry (A), shift (S), and kurtosis
    # (K)
    # ---
    # Calculating FWHM
    bha_FWHM_calc = peak_bha_flux / 2
    bha_FWHM_range = bha_data.loc[(bha_data['Flux'] >= bha_FWHM_calc)]
    bha_FWHM_wave_min = bha_FWHM_range['Wavelength'].iloc[0]
    bha_FWHM_wave_max = bha_FWHM_range['Wavelength'].iloc[-1]
    bha_FWHM_wave = bha_FWHM_wave_max - bha_FWHM_wave_min    
    bha_FWHM = bha_FWHM_wave / peak_nha * 299792
    # ---
    # Calculating line parameters    
    bha_a = bha_C80_ctr - bha_10
    bha_b = (bha_C80_ctr - bha_90) * (-1)
    bha_IPV = (bha_a + bha_b) / (peak_nha) * 299792
    bha_A = (bha_a - bha_b) / (bha_a + bha_b)                          # asymmetry     
    bha_K = (1.397*bha_FWHM) / ((bha_a + bha_b) / (bha_C80_ctr) * 299792) # kurtosis
    
    # Section 5: Plotting EVERYTHING onto one plot
    # ---
    fig2 = plt.figure(figsize=(16,10))
    plt.plot(wavelength, nl, c='#0B6E4F', linewidth=2.25, label='NL Profile')
    plt.plot(wavelength, bl, c='#CB2C2A', linewidth=2.25, label='BL Profile')
    plt.plot(wavelength, bl_profile, c='b', linewidth=2.25, label='Cleaned BL Profile')
    plt.plot(wavelength, data_contFeII_sub, c='k', linewidth=2.25, label='Data - Continuum - FeII')
    # Plotting corresponding NL peak wavelength and the three wavelengths of 
    # the calculated velocity shifts as vertical lines 
    plt.axvline(peak_nha, c='#146746', linestyle=':', linewidth=1.75, label='NL Peak')
    plt.axvline(peak_bha, c='#CB2C2A', linestyle=':', linewidth=1.75, label='BL Peak')
    plt.axvline(bha_ctr, c='#629FD0', linestyle='--', linewidth=1.75, label='BL Centroid')
    plt.axvline(bha_C80_ctr, c='#9B469B', linestyle='--', linewidth=1.75, label='BL C80')
    # Plot details
    plt.xlabel(r'$\rm Rest \, Wavelength$ ($\rm \AA$)', fontsize=16)
    plt.xlim(6300, 6900)                                  # match with bha_data
    plt.ylabel(r'$\rm f_{\lambda}$ ($\rm 10^{-17} erg\;s^{-1}\;cm^{-2}\;\AA^{-1}$)', fontsize=16)
    plt.title(source+r': H$\alpha$ Line Complex', fontsize=25)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.tick_params(which='major', length=12, width=1)
    plt.legend(fontsize=12)
    plt.savefig(path2+source+'_BHA_ProfileCalc.pdf')
    

# -----------------------------------------------------------------------------

# H_beta Line Complex

# If H_beta is not present in your spectrum (z > 0.89), the velocity shift
# calculations can be skipped by answering 'no' to "bhb_vs"

bhb_vs = 'yes'
if(bhb_vs == 'yes'):
    # Section 1: finds the peak velocity shift by obtaining the corresponding
    # wavelength of the max flux value of the BL profile
    # ---
    # BLs - change according to length of wings
    bhb_data = bl_data.loc[(bl_data['Wavelength'] >= 4600) & \
                           (bl_data['Wavelength'] <= 5100)]
    bhb_wave = bhb_data['Wavelength']
    bhb_flux = bhb_data['Flux']
    # Finding peak 
    peak_bhb_flux = bhb_flux.max()
    peak_bhb_wave = bhb_data.loc[bhb_data['Flux'] == peak_bhb_flux]
    peak_bhb = int(peak_bhb_wave['Wavelength'])
    # ---
    # NLs 
    nhb_data = nl_data.loc[(nl_data['Wavelength'] >= 4650) & \
                           (nl_data['Wavelength'] <= 4900)]
    nhb_flux = nhb_data['Flux']
    # Finding peak: should be ~ vacuum wavelength and will be used for 
    # calculating all three types of velocity shifts
    peak_nhb_flux = nhb_flux.max()
    peak_nhb_wave = nhb_data.loc[nhb_data['Flux'] == peak_nhb_flux]
    peak_nhb = int(peak_nhb_wave['Wavelength'])
    #peak_nhb = 4862.68 # use when H_beta is not fitted due to heavily blended spectra
    # ---
    # Peak velocity shift 
    bhb_pvs = ((peak_bhb**2 - peak_nhb**2) / (peak_bhb**2 + peak_nhb**2)) * 299792
    
    # Section 2: finds the centroid velocity shift by normalizing the BL 
    # profile and finding the cumulative distribution function (CDF) to obtain 
    # the corresponding wavelength at 50% of the CDF
    # ---
    # Normalizing BL profile and finding CDF
    bhb_flux_rescaled = (bhb_flux - bhb_flux.min()) / (bhb_flux.max() - bhb_flux.min())
    bhb_flux_area = integrate.trapezoid(bhb_flux_rescaled, bhb_wave)
    bhb_flux_new = bhb_flux_rescaled/bhb_flux_area   
    bhb_flux_norm = bhb_flux_new / np.sum(bhb_flux_new)
    bhb_cdf = np.cumsum(bhb_flux_norm)
    # ---
    # Finding centroid
    bhb_ctr = np.interp(0.5, bhb_cdf, bhb_wave)
    # Ploting CDF 
    plot_bhb_cdf = 'yes'
    if(plot_bhb_cdf == 'yes'):
        fig4 = plt.figure(figsize=(16,10))
        plt.plot(bhb_wave, bhb_cdf, linewidth=2, c='#DB1D1A')
        plt.title(source+r': H$\beta$ CDF', fontsize=30)
        plt.ylabel('Probability', fontsize=20)
        plt.xlabel(r'Wavelength ($\rm \AA$)', fontsize=20)
        plt.xticks(fontsize=20)
        plt.yticks(fontsize=20)
        plt.tick_params(which='major', length=16, width=2)
        plt.tick_params(which='minor', length=8, width=1)
        plt.savefig(path2+source+'_BHB_CDF.pdf')
        plt.axvline(bhb_ctr, c='grey', linestyle='--', label='Centroid')
        plt.text(bhb_ctr+5, 0.2, r'H$\beta$ BL Centroid = {:.2f} $\AA$'.format(bhb_ctr))
        plt.legend(fontsize=14)
    # ---
    # Centroid velocity shift
    bhb_cvs = ((bhb_ctr**2 - peak_nhb**2) / (bhb_ctr**2 + peak_nhb**2)) * 299792 
    
    # Section 3: finding C80 shift by removing 10% of the area at each end of
    # the BL profile and obtaining the corresponding wavelength at the 
    # center.
    # ---
    bhb_10 = np.interp(0.10, bhb_cdf, bhb_wave)
    bhb_90 = np.interp(0.90, bhb_cdf, bhb_wave)
    bhb_C80_data = bl_data.loc[(bl_data['Wavelength'] >= bhb_10) & \
                               (bl_data['Wavelength'] <= bhb_90)]
    bhb_C80_wave = bhb_C80_data['Wavelength']
    bhb_C80_flux = bhb_C80_data['Flux']
    # Normalizing BL profile and finding CDF
    bhb_C80_flux_rescaled = (bhb_C80_flux - (bhb_C80_flux.min())) / \
        (bhb_C80_flux.max() - bhb_C80_flux.min())
    bhb_C80_area = integrate.trapezoid(bhb_C80_flux_rescaled, bhb_C80_wave)
    bhb_C80_flux_new = bhb_C80_flux_rescaled/bhb_C80_area
    bhb_C80_flux_norm = bhb_C80_flux_new / np.sum(bhb_C80_flux_new)
    bhb_cdf_C80 = np.cumsum(bhb_C80_flux_norm)
    # ---
    # Finding line center
    bhb_C80_ctr = np.interp(0.5, bhb_cdf_C80, bhb_C80_wave)
    # C80 velocity shift
    bhb_C80_vs = ((bhb_C80_ctr**2 - peak_nhb**2) / (bhb_C80_ctr**2 + peak_nhb**2)) * 299792 
    
    # Section 4: finding the Whittle 1985 line profile parameters; i.e. inter-
    # percentile velocity width (IPV), asymmetry (A), shift (S), and kurtosis
    # (K)
    # ---
    # Calculating FWHM
    bhb_FWHM_calc = peak_bhb_flux / 2
    bhb_FWHM_range = bhb_data.loc[(bhb_data['Flux'] >= bhb_FWHM_calc)]
    bhb_FWHM_wave_min = bhb_FWHM_range['Wavelength'].iloc[0]
    bhb_FWHM_wave_max = bhb_FWHM_range['Wavelength'].iloc[-1]
    bhb_FWHM_wave = bhb_FWHM_wave_max - bhb_FWHM_wave_min    
    bhb_FWHM = bhb_FWHM_wave / peak_nhb * 299792
    # ---
    # Calculating line parameters    
    bhb_a = bhb_C80_ctr - bhb_10
    bhb_b = (bhb_C80_ctr - bhb_90) * (-1)
    bhb_IPV = (bhb_a + bhb_b) / (peak_nhb) * 299792
    bhb_A = (bhb_a - bhb_b) / (bhb_a + bhb_b)                          # asymmetry
    bhb_K = (1.397*bhb_FWHM) / ((bhb_a + bhb_b) / (bhb_C80_ctr) * 299792) # kurtosis
    
    # Section 5: Plotting EVERYTHING onto one plot
    # ---
    fig5 = plt.figure(figsize=(16,10))
    plt.plot(wavelength, nl, c='#0B6E4F', linewidth=2.25, label='NL Profile')
    plt.plot(wavelength, bl, c='#CB2C2A', linewidth=2.25, label='BL Profile')
    plt.plot(wavelength, bl_profile, c='b', linewidth=2.25, label='Cleaned BL Profile')
    plt.plot(wavelength, data_contFeII_sub, c='k', linewidth=2.25, label='Data - Continuum - FeII')
    # Plotting corresponding NL peak wavelength and the three wavelengths of 
    # the calculated velocity shifts as vertical lines 
    plt.axvline(peak_nhb, c='#146746', linestyle=':', linewidth=1.75, label='NL Peak')
    plt.axvline(peak_bhb, c='#CB2C2A', linestyle=':', linewidth=1.75, label='BL Peak')
    plt.axvline(bhb_ctr, c='#629FD0', linestyle='--', linewidth=1.75, label='BL Centroid')
    plt.axvline(bhb_C80_ctr, c='#9B469B', linestyle='--', linewidth=1.75, label='BL C80')
    # Plot details
    plt.xlabel(r'$\rm Rest \, Wavelength$ ($\rm \AA$)', fontsize=16)
    plt.xlim(4600, 5100)                                  # match with bhb_data
    plt.ylabel(r'$\rm f_{\lambda}$ ($\rm 10^{-17} erg\;s^{-1}\;cm^{-2}\;\AA^{-1}$)', fontsize=16)
    plt.title(source+r': H$\beta$ Line Complex', fontsize=25)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.tick_params(which='major', length=12, width=1)
    plt.legend(fontsize=12)
    plt.savefig(path2+source+'_BHB_ProfileCalc.pdf')
    


# -----------------------------------------------------------------------------

# MgII Line Complex

# If MgII is not present in your spectrum (z < 0.36), the velocity shift
# calculations can be skipped by answering 'no' to "mg2_vs"

mg2_vs = 'yes'
if(mg2_vs == 'yes'):
    # Section 1: finds the peak velocity shift by obtaining the corresponding
    # wavelength of the max flux value of the BL profile
    # ---
    # BLs - change according to length of wings
    bmg_data = bl_data.loc[(bl_data['Wavelength'] >= 2700) & \
                           (bl_data['Wavelength'] <= 2900)]
    bmg_wave = bmg_data['Wavelength']
    bmg_flux = bmg_data['Flux']
    # Finding peak 
    peak_bmg_flux = bmg_flux.max()
    peak_bmg_wave = bmg_data.loc[bmg_data['Flux'] == peak_bmg_flux]
    peak_bmg = int(peak_bmg_wave['Wavelength'])
    # ---
    # NLs 
    nmg_data = nl_data.loc[(nl_data['Wavelength'] >= 2700) & \
                           (nl_data['Wavelength'] <= 2900)]
    nmg_flux = nmg_data['Flux']
    # Finding peak: should be ~ vacuum wavelength and will be used for 
    # calculating all three types of velocity shifts
    peak_nmg_flux = nmg_flux.max()
    peak_nmg_wave = nmg_data.loc[nmg_data['Flux'] == peak_nmg_flux]
    peak_nmg = int(peak_nmg_wave['Wavelength'])
    peak_nmg = 2798.75 # use when MgII is not fitted due to heavily blended spectra
    # ---
    # Peak velocity shift 
    bmg_pvs = ((peak_bmg**2 - peak_nmg**2) / (peak_bmg**2 + peak_nmg**2)) * 299792
    
    # Section 2: finds the centroid velocity shift by normalizing the BL 
    # profile and finding the cumulative distribution function (CDF) to obtain 
    # the corresponding wavelength at 50% of the CDF
    # ---
    # Normalizing BL profile and finding CDF
    bmg_flux_rescaled = (bmg_flux - bmg_flux.min()) / (bmg_flux.max() - bmg_flux.min())
    bmg_flux_area = integrate.trapezoid(bmg_flux_rescaled, bmg_wave)
    bmg_flux_new = bmg_flux_rescaled/bmg_flux_area   
    bmg_flux_norm = bmg_flux_new / np.sum(bmg_flux_new)
    bmg_cdf = np.cumsum(bmg_flux_norm)
    # ---
    # Finding centroid
    bmg_ctr = np.interp(0.5, bmg_cdf, bmg_wave)
    # Ploting CDF 
    plot_bmg_cdf = 'yes'
    if(plot_bmg_cdf == 'yes'):
        fig4 = plt.figure(figsize=(16,10))
        plt.plot(bmg_wave, bmg_cdf, linewidth=2, c='#DB1D1A')
        plt.title(source+': MgII CDF', fontsize=30)
        plt.ylabel('Probability', fontsize=20)
        plt.xlabel(r'Wavelength ($\rm \AA$)', fontsize=20)
        plt.xticks(fontsize=20)
        plt.yticks(fontsize=20)
        plt.tick_params(which='major', length=16, width=2)
        plt.tick_params(which='minor', length=8, width=1)
        plt.savefig(path2+source+'_BMG_CDF.pdf')
        plt.axvline(bmg_ctr, c='grey', linestyle='--', label='Centroid')
        plt.text(bmg_ctr+5, 0.2, r'MgII BL Centroid = {:.2f} $\AA$'.format(bmg_ctr))
        plt.legend(fontsize=14)
    # ---
    # Centroid velocity shift
    bmg_cvs = ((bmg_ctr**2 - peak_nmg**2) / (bmg_ctr**2 + peak_nmg**2)) * 299792 
    
    # Section 3: finding C80 shift by removing 10% of the area at each end of
    # the BL profile and obtaining the corresponding wavelength at the 
    # center.
    # ---
    bmg_10 = np.interp(0.10, bmg_cdf, bmg_wave)
    bmg_90 = np.interp(0.90, bmg_cdf, bmg_wave)
    bmg_C80_data = bl_data.loc[(bl_data['Wavelength'] >= bmg_10) & \
                               (bl_data['Wavelength'] <= bmg_90)]
    bmg_C80_wave = bmg_C80_data['Wavelength']
    bmg_C80_flux = bmg_C80_data['Flux']
    # Normalizing BL profile and finding CDF
    bmg_C80_flux_rescaled = (bmg_C80_flux - (bmg_C80_flux.min())) / \
        (bmg_C80_flux.max() - bmg_C80_flux.min())
    bmg_C80_area = integrate.trapezoid(bmg_C80_flux_rescaled, bmg_C80_wave)
    bmg_C80_flux_new = bmg_C80_flux_rescaled/bmg_C80_area
    bmg_C80_flux_norm = bmg_C80_flux_new / np.sum(bmg_C80_flux_new)
    bmg_cdf_C80 = np.cumsum(bmg_C80_flux_norm)
    # ---
    # Finding line center
    bmg_C80_ctr = np.interp(0.5, bmg_cdf_C80, bmg_C80_wave)
    # C80 velocity shift
    bmg_C80_vs = ((bmg_C80_ctr**2 - peak_nmg**2) / (bmg_C80_ctr**2 + peak_nmg**2)) * 299792 
    
    # Section 4: finding the Whittle 1985 line profile parameters; i.e. inter-
    # percentile velocity width (IPV), asymmetry (A), shift (S), and kurtosis
    # (K)
    # ---
    # Calculating FWHM
    bmg_FWHM_calc = peak_bmg_flux / 2
    bmg_FWHM_range = bmg_data.loc[(bmg_data['Flux'] >= bmg_FWHM_calc)]
    bmg_FWHM_wave_min = bmg_FWHM_range['Wavelength'].iloc[0]
    bmg_FWHM_wave_max = bmg_FWHM_range['Wavelength'].iloc[-1]
    bmg_FWHM_wave = bmg_FWHM_wave_max - bmg_FWHM_wave_min    
    bmg_FWHM = bmg_FWHM_wave / peak_nmg * 299792
    # ---
    # Calculating line parameters    
    bmg_a = bmg_C80_ctr - bmg_10
    bmg_b = (bmg_C80_ctr - bmg_90) * (-1)
    bmg_IPV = (bmg_a + bmg_b) / (peak_nmg) * 299792
    bmg_A = (bmg_a - bmg_b) / (bmg_a + bmg_b)                          # asymmetry
    bmg_K = (1.397*bmg_FWHM) / ((bmg_a + bmg_b) / (bmg_C80_ctr) * 299792) # kurtosis
    
    # Section 5: Plotting EVERYTHING onto one plot
    # ---
    fig5 = plt.figure(figsize=(16,10))
    #plt.plot(wavelength, nl, c='#0B6E4F', linewidth=2.25, label='NL Profile')
    plt.plot(wavelength, bl, c='#CB2C2A', linewidth=2.25, label='BL Profile')
    plt.plot(wavelength, bl_profile, c='b', linewidth=2.25, label='Cleaned BL Profile')
    plt.plot(wavelength, data_contFeII_sub, c='k', linewidth=2.25, label='Data - Continuum - FeII')
    # Plotting corresponding NL peak wavelength and the three wavelengths of 
    # the calculated velocity shifts as vertical lines 
    #plt.axvline(peak_nmg, c='#146746', linestyle=':', linewidth=1.75, label='NL Peak')
    # Use next line if you had to use vacuum wavelength
    plt.axvline(peak_nmg, c='#146746', linestyle=':', linewidth=1.75, label='Vacuum Wavelength')
    plt.axvline(peak_bmg, c='#CB2C2A', linestyle=':', linewidth=1.75, label='BL Peak')
    plt.axvline(bmg_ctr, c='#629FD0', linestyle='--', linewidth=1.75, label='BL Centroid')
    plt.axvline(bmg_C80_ctr, c='#9B469B', linestyle='--', linewidth=1.75, label='BL C80')
    # Plot details
    plt.xlabel(r'$\rm Rest \, Wavelength$ ($\rm \AA$)', fontsize=16)
    plt.xlim(2700, 2900)                                  # match with bmg_data
    plt.ylabel(r'$\rm f_{\lambda}$ ($\rm 10^{-17} erg\;s^{-1}\;cm^{-2}\;\AA^{-1}$)', fontsize=16)
    plt.title(source+': MgII Line Complex', fontsize=25)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.tick_params(which='major', length=12, width=1)
    plt.legend(fontsize=12)
    plt.savefig(path2+source+'_BMG_ProfileCalc.pdf')
    
    
# -----------------------------------------------------------------------------

# Velocity Plots

# This part of the code will overplot the normalized BL profiles on a velocity
# scale for direct comparison

plot_vel = 'yes'
if(plot_vel == 'yes'):
    # Converting wavelengths to velocities
    if(bha_vs == 'yes'):
        bha_vel = (bha_wave - peak_nha) / (peak_nha) * 299792
    if(bhb_vs == 'yes'):
        bhb_vel = (bhb_wave - peak_nhb) / (peak_nhb) * 299792
    if(mg2_vs == 'yes'):
        bmg_vel = (bmg_wave - peak_nmg) / (peak_nmg) * 299792
    fig3 = plt.figure(figsize=(16,10))
    if(bha_vs == 'yes'):
        plt.plot(bha_vel, bha_flux_norm, c='purple', linewidth=2, label=r'H$\alpha$ BL Profile')
    if(bhb_vs == 'yes'):
        plt.plot(bhb_vel, bhb_flux_norm, c='darkgoldenrod', linewidth=2, label=r'H$\beta$ BL Profile')
    if(mg2_vs == 'yes'):
        plt.plot(bmg_vel, bmg_flux_norm, c='darkgreen', linewidth=2, label='MgII BL Profile')
    plt.axvline(0, c='k', linestyle=(0, (5, 5)), linewidth=1)
    # Plot details
    plt.xlabel(r'Velocity (km s$^{-1}$)', fontsize=16)                  
    plt.ylabel('Normalized Flux', fontsize=16)
    plt.title(source, fontsize=25)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.tick_params(which='major', length=12, width=1)
    plt.legend(fontsize=12)
    plt.savefig(path2+source+'_BL_VProfile.pdf')


# -----------------------------------------------------------------------------

# Saving calculated velocity shifts and Whittle 1985 profile parameters into a 
# separate .txt file

with open(path2+source+"_BLV-Shifts.txt","w") as f:
    print('Calculated BL Velocity Shifts and Line Profile Parameters:', file=f)
    print(" ", file=f)
    if(bha_vs == 'yes'):
        print("   Peak H\u03B1 Velocity Shift =", '%.2f' % bha_pvs, 'km s\u207B\u00B9', file=f)
        print("   Center H\u03B1 Velocity Shift =", '%.2f' % bha_cvs, 'km s\u207B\u00B9', file=f)
        print("   C80 H\u03B1 Velocity Shift =", '%.2f' % bha_C80_vs, 'km s\u207B\u00B9', file=f)
        print("   H\u03B1 IPV width =", '%.2f' % bha_IPV, 'km s\u207B\u00B9', file=f)
        print("   H\u03B1 Asymmetry =", '%.5f' % bha_A, file=f)
        print("   H\u03B1 Kurtosis =", '%.5f' % bha_K, file=f)
        print(" ", file=f)
    if(bhb_vs == 'yes'):
        print("   Peak H\u03B2 Velocity Shift =", '%.2f' % bhb_pvs, 'km s\u207B\u00B9', file=f)
        print("   Center H\u03B2 Velocity Shift =", '%.2f' % bhb_cvs, 'km s\u207B\u00B9', file=f)
        print("   C80 H\u03B2 Velocity Shift =", '%.2f' % bhb_C80_vs, 'km s\u207B\u00B9', file=f)
        print("   H\u03B2 IPV width =", '%.2f' % bhb_IPV, 'km s\u207B\u00B9', file=f)
        print("   H\u03B2 Asymmetry =", '%.5f' % bhb_A, file=f)
        print("   H\u03B2 Kurtosis =", '%.5f' % bhb_K, file=f)
        print(" ", file=f)
    if(mg2_vs == 'yes'):
        print("   Peak MgII Velocity Shift =", '%.2f' % bmg_pvs, 'km s\u207B\u00B9', file=f)
        print("   Center MgII Velocity Shift =", '%.2f' % bmg_cvs, 'km s\u207B\u00B9', file=f)
        print("   C80 MgII Velocity Shift =", '%.2f' % bmg_C80_vs, 'km s\u207B\u00B9', file=f)
        print("   MgII IPV width =", '%.2f' % bmg_IPV, 'km s\u207B\u00B9', file=f)
        print("   MgII Asymmetry =", '%.5f' % bmg_A, file=f)
        print("   MgII Kurtosis =", '%.5f' % bmg_K, file=f)
