"""
Fitting-Script.py
L. Matrecito, lmm8709@rit.edu
Mon Jun 13 12:56:12 2022

This code can be used to run PyQSOFit after an install of the source code. 

More specifically, it is used with the SDSS DR7 Quasar Catalog. It will fit the
spectrum of a quasar by taking a FITS file and running it with PyQSOFit. It
also includes plotting of the fitted line complexes separately, saving fitted 
line properties to a text tile, and (optional) the cleaning of the data so that 
only the broad-line (BL) profile is left. The fitted data is stored separately 
to be analyzed by LineProfile_Calc.py, a code that calculates velocity shifts 
and characteristic line profile shapes. Please look through the code as there 
are important notes (separated by blocks, titled IMPORTANT) PRIOR to running.
"""

import os,timeit
import numpy as np
from PyQSOFit import QSOFit
from astropy.io import fits
import matplotlib.pyplot as plt
import warnings

# Use custom matplotlib style
QSOFit.set_mpl_style()
# This was in the example code, not sure for what exactly...
warnings.filterwarnings("ignore")

# Setting the file paths that the code uses
# The path of the source code file and qsopar.fits
path1 = '/Users/lmm8709/PyQSOFit/'
# The path of fit results
path2 = path1+'Fit Results/'     
# The path of fits results for the spectrum 
path3 = path2+'QA Other/'
# The path of dust reddening map
path4 = path1+'sfddata/'


# -----------------------------------------------------------------------------

# Opening spectrum to be fitted. NOTE: SDSS fits files are saved as
# spec-plateID-MJD-fiberID.fits. Source is just the plateID-MJD-fiberID
source = '0545-52202-0238'
spec = 'spec-'+source+'.fits'
data = fits.open(os.path.join(path1+'Data/'+spec))
lam = 10**data[1].data['loglam']                           # OBS wavelength (A)
flux = data[1].data['flux']                           # OBS flux (erg/s/cm^2/A)
err = 1./np.sqrt(data[1].data['ivar'])                          # 1 sigma error
z = data[2].data['z'][0]                                             # Redshift

# Optional information... 
ra = data[0].header['plug_ra']                                             # RA 
dec = data[0].header['plug_dec']                                          # DEC
plateid = data[0].header['plateid']                             # SDSS plate ID
mjd = data[0].header['mjd']                                          # SDSS MJD
fiberid = data[0].header['fiberid']                             # SDSS fiber ID


# -----------------------------------------------------------------------------

# IMPORTANT

# Remember to make a directory within the Line Complex Properties directory
# using the plateID-MJD-fiberID as the folder name PRIOR to running the code
# or else it will not work because it will not be able to find that directory.
# A portion of this code will create those directories automatically
# and is currently a WIP!


# -----------------------------------------------------------------------------

# PyQSOFit - fitting portion
# Preparing spectrum data
q_mle = QSOFit(lam, flux, err, z, ra=ra, dec=dec, plateid=plateid, mjd=mjd, 
               fiberid=fiberid, path=path1)
start = timeit.default_timer()

# Do the fitting. NOTE: Change arguments accordingly
q_mle.Fit(name=None, nsmooth=1, deredden=True, 
          reject_badpix=False, wave_range=None, 
          wave_mask=np.array([[3808,3811],[3811,3815]]), 
          decompose_host=True, host_line_mask=True, BC03=False, Mi=None, 
          npca_gal=5, npca_qso=10, Fe_uv_op=True, Fe_flux_range=None, 
          poly=True, BC=False, initial_guess=None, tol=1e-10, use_ppxf=False,
          n_pix_min_conti=100, param_file_name='qsopar.fits', MC=False, 
          MCMC=False, nburn=20, nsamp=200, nthin=10, epsilon_jitter=1e-4, 
          linefit=True, save_result=True, plot_fig=True, 
          save_fig=True, plot_corner=False, save_fig_path=path2,
          save_fits_path=path3, save_fits_name=None, verbose=False, 
          kwargs_conti_emcee={}, kwargs_line_emcee={})
end = timeit.default_timer()
print('Fitting finished in : '+str(np.round(end-start))+'s')


# -----------------------------------------------------------------------------

# Obtaining fit result files saved under QA Other subdirectory in Fit Results
# source code directory

data = fits.open(path3+source+'.fits') 


# -----------------------------------------------------------------------------

# IMPORTANT


# WIP toautoomatically create directories so that the user does not manually
# have to make them + remember to have to make them...
    
# Creating directories for fit results - WIP
#fit_plots = '/Fit Results/Line Complex Properties/'+source+'/'
#if not os.path.exists(fit_plots):
#    os.makedirs(fit_plots, mode=0o777)
#fit_data = '/Fit Results/Line Complex Properties/'+source+'/Fit Data/'
#if not os.path.exists(fit_data):
#    os.makedirs(fit_data, mode=0o777)


# -----------------------------------------------------------------------------

# Saving each line complex individually because PyQSOFit has been modified to 
# not plot these under the full fit figure; therefore, we plot them here 
# separately. In addition, we only plot H_alpha, H_beta, and MgII line 
# complexes since those will be used for finding velocity shifts

# Path of line complex plots
path5 = path2+'Line Complex Properties/'+source+'/'
                   
# Plotting H_alpha coomplex
plot_Ha = 'no'  
if(plot_Ha =='yes'):
    # Plotting broad H_alpha, NII, and SII line complex
    fig1 = plt.figure(figsize=(16,12))
    for p in range(len(q_mle.gauss_result)//3):
        if q_mle.CalFWHM(q_mle.gauss_result[3*p+2]) < 1200:          # FWHM max
            color = 'g' # narrow
        else:
            color = 'r' # broad
        plt.plot(q_mle.wave, q_mle.Onegauss(np.log(q_mle.wave), \
                                 q_mle.gauss_result[p*3:(p+1)*3]), color=color)  
    # Plot total line model
    plt.plot(q_mle.wave, q_mle.Manygauss(np.log(q_mle.wave), \
                                         q_mle.gauss_result),'b', lw=2)
    plt.plot(q_mle.wave, q_mle.line_flux,'k')
    plt.xlim(6200, 6900)
    plt.xlabel(r'$\rm Rest \, Wavelength$ ($\rm \AA$)', fontsize = 20)
    plt.ylabel(r'$\rm f_{\lambda}$ ($\rm 10^{-17} erg\;s^{-1}\;cm^{-2}\;\AA^{-1}$)'
               , fontsize = 20)
    plt.title(r'Broad $H{\alpha}+[NII]+[SII]$', fontsize = 30)
    plt.savefig(path5+source+'_BroadHa_LineComplex.pdf')


# Plotting broad H_beta and [OIII] line complex
plot_Hb = 'yes'
if(plot_Hb =='yes'):
    fig2 = plt.figure(figsize=(16,12))
    for p in range(len(q_mle.gauss_result)//3):
        if q_mle.CalFWHM(q_mle.gauss_result[3*p+2]) < 1200:  
            color = 'g' # narrow
        else:
            color = 'r' # broad
        plt.plot(q_mle.wave, q_mle.Onegauss(np.log(q_mle.wave), \
                                 q_mle.gauss_result[p*3:(p+1)*3]), color=color)
    # Plot total line model
    plt.plot(q_mle.wave, q_mle.Manygauss(np.log(q_mle.wave), \
                                         q_mle.gauss_result), 'b', lw=2)
    plt.plot(q_mle.wave, q_mle.line_flux,'k')
    plt.xlim(4650, 5100)
    plt.xlabel(r'$\rm Rest \, Wavelength$ ($\rm \AA$)', fontsize = 20)
    plt.ylabel(r'$\rm f_{\lambda}$ ($\rm 10^{-17} erg\;s^{-1}\;cm^{-2}\;\AA^{-1}$)'
               , fontsize = 20)
    plt.title(r'Broad $H{\beta}+[OIII]$', fontsize = 30)
    plt.savefig(path5+source+'_BroadHb_LineComplex.pdf')

# For high z, MgII lines appear and can be used for calculating velocity shifts
plot_MgII = 'yes'                  
if(plot_MgII =='yes'):
    fig4 = plt.figure(figsize=(16,12))
    for p in range(len(q_mle.gauss_result)//3):
        if q_mle.CalFWHM(q_mle.gauss_result[3*p+2]) < 1200: 
            color = 'g' # narrow
        else:
            color = 'r' # broad
        plt.plot(q_mle.wave, q_mle.Onegauss(np.log(q_mle.wave), 
        q_mle.gauss_result[p*3:(p+1)*3]), color=color) 
    # Plot total line model
    plt.plot(q_mle.wave, q_mle.Manygauss(np.log(q_mle.wave), \
                                         q_mle.gauss_result), 'b', lw=2)
    plt.plot(q_mle.wave, q_mle.line_flux,'k')
    plt.xlim(2600, 3000)
    plt.xlabel(r'$\rm Rest \, Wavelength$ ($\rm \AA$)', fontsize = 20)
    plt.ylabel(r'$\rm f_{\lambda}$ ($\rm 10^{-17} erg\;s^{-1}\;cm^{-2}\;\AA^{-1}$)'
                   , fontsize = 20)
    plt.title(r'Broad MgII', fontsize = 30)
    plt.savefig(path5+source+'_BroadMgII_LineComplex.pdf')



# -----------------------------------------------------------------------------    

# PyQSOFit calculates FWHM, Sigma, EW, Area, and SNR for each broad and narrow
# component of emission lines. That information is obtained and then saved to a 
# separate .txt file

# Broad H_alpha
fwhm_bha, sigma_bha, ew_bha, peak_bha, area_bha, snr_bha = \
    q_mle.line_prop_from_name('Ha_br','broad')
# Narrow H_alpha
fwhm_nha, sigma_nha, ew_nha, peak_nha, area_nha, snr_nha = \
    q_mle.line_prop_from_name('Ha_na', 'narrow')
# Broad H_beta 
fwhm_bhb, sigma_bhb, ew_bhb, peak_bhb, area_bhb, snr_bhb = \
    q_mle.line_prop_from_name('Hb_br', 'broad')
# Narrow H_beta
fwhm_nhb, sigma_nhb, ew_nhb, peak_nhb, area_nhb, snr_nhb = \
    q_mle.line_prop_from_name('Hb_na', 'narrow')
# [OIII]5007 - core
fwhm_oIII5, sigma_oIII5, ew_oIII5, peak_oIII5, area_oIII5, snr_oIII5 = \
    q_mle.line_prop_from_name('OIII5007c', 'narrow')
# [OIII]4959 - core
fwhm_oIII4, sigma_oIII4, ew_oIII4, peak_oIII4, area_oIII4, snr_oIII4 = \
    q_mle.line_prop_from_name('OIII4959c', 'narrow')
# Broad MgII
fwhm_bmgII, sigma_bmgII, ew_bmgII, peak_bmgII, area_bmgII, snr_bmgII = \
    q_mle.line_prop_from_name('MgII_br', 'broad')
# Narrow MgII
fwhm_nmgII, sigma_nmgII, ew_nmgII, peak_nmgII, area_nmgII, snr_nmgII = \
    q_mle.line_prop_from_name('MgII_na', 'narrow')

with open(path5+source+"_LineProperties.txt","w") as f:
    print('PyQSOFit Calculated Line Properties', file=f)
    print('', file=f)
    # Broad H_alpha
    print('Broad H\u03B1:', file=f)
    print(' '  + "FWHM = " '%.2f' % fwhm_bha, "km s\u207B\u00B9", file=f)
    print(' '  + "\u03C3 = " '%.2f' % sigma_bha, "km s\u207B \u00B9", file=f)
    print(' '  + "EW = ", '%.2f' % ew_bha, "\u212B", file=f)
    print(' '  + "Peak = ", '%.2f' % peak_bha, "\u212B", file=f)
    print(' '  + "Area = ", '%.2f' % area_bha, 
    "x 10\u207B\u00B9\u2077 erg s\u207B\u00B9 cm\u207B\u00B2", file=f)
    # Narrow H_alpha
    print('Narrow H\u03B1:', file=f)
    print(' '  + "FWHM = " '%.2f' % fwhm_nha, "km s\u207B\u00B9", file=f)
    print(' '  + "\u03C3 = " '%.2f' % sigma_nha, "km s\u207B \u00B9", file=f)
    print(' '  + "EW = ", '%.2f' % ew_nha, "\u212B", file=f)
    print(' '  + "Peak = ", '%.2f' % peak_nha, "\u212B", file=f)        
    print(' '  + "Area = ", '%.2f' % area_nha, 
    "x 10\u207B\u00B9\u2077 erg s\u207B\u00B9 cm\u207B\u00B2", file=f)
    # Broad H_beta
    print('Broad H\u03B2:', file=f)
    print(' '  + "FWHM = " '%.2f' % fwhm_bhb, "km s\u207B\u00B9", file=f)
    print(' '  + "\u03C3 = " '%.2f' % sigma_bhb, "km s\u207B \u00B9", file=f)
    print(' '  + "EW = ", '%.2f' % ew_bhb, "\u212B", file=f)
    print(' '  + "Peak = ", '%.2f' % peak_bhb, "\u212B", file=f)
    print(' '  + "Area = ", '%.2f' % area_bhb, 
    "x 10\u207B\u00B9\u2077 erg s\u207B\u00B9 cm\u207B\u00B2", file=f)
    # Narrow H_beta
    print('Narrow H\u03B2:', file=f)
    print(' '  + "FWHM = " '%.2f' % fwhm_nhb, "km s\u207B\u00B9", file=f)
    print(' '  + "\u03C3 = " '%.2f' % sigma_nhb, "km s\u207B \u00B9", file=f)
    print(' '  + "EW = ", '%.2f' % ew_nhb, "\u212B", file=f)
    print(' '  + "Peak = ", '%.2f' % peak_nhb, "\u212B", file=f)        
    print(' '  + "Area = ", '%.2f' % area_nhb, 
    "x 10\u207B\u00B9\u2077 erg s\u207B\u00B9 cm\u207B\u00B2", file=f)
    # Narrow [OIII]5007
    print('Narrow [OIII]5007:', file=f)
    print(' '  + "FWHM = " '%.2f' % fwhm_oIII5, "km s\u207B\u00B9", file=f)
    print(' '  + "\u03C3 = " '%.2f' % sigma_oIII5, "km s\u207B \u00B9", file=f)
    print(' '  + "EW = ", '%.2f' % ew_oIII5, "\u212B", file=f)
    print(' '  + "Peak = ", '%.2f' % peak_oIII5, "\u212B", file=f)
    print(' '  + "Area = ", '%.2f' % area_oIII5, 
    "x 10\u207B\u00B9\u2077 erg s\u207B\u00B9 cm\u207B\u00B2", file=f)
    # Narrow [OIII]4959
    print('Narrow [OIII]4959:', file=f)
    print(' '  + "FWHM = " '%.2f' % fwhm_oIII4, "km s\u207B\u00B9", file=f)
    print(' '  + "\u03C3 = " '%.2f' % sigma_oIII4, "km s\u207B \u00B9", file=f)
    print(' '  + "EW = ", '%.2f' % ew_oIII4, "\u212B", file=f)
    print(' '  + "Peak = ", '%.2f' % peak_oIII4, "\u212B", file=f)
    print(' '  + "Area = ", '%.2f' % area_oIII4, 
    "x 10\u207B\u00B9\u2077 erg s\u207B\u00B9 cm\u207B\u00B2", file=f)
    # Broad MgII
    print('Broad MgII:', file=f)
    print(' '  + "FWHM = " '%.2f' % fwhm_bmgII, "km s\u207B\u00B9", file=f)
    print(' '  + "\u03C3 = " '%.2f' % sigma_bmgII, "km s\u207B \u00B9", file=f)
    print(' '  + "EW = ", '%.2f' % ew_bmgII, "\u212B", file=f)
    print(' '  + "Peak = ", '%.2f' % peak_bmgII, "\u212B", file=f)
    print(' '  + "Area = ", '%.2f' % area_bmgII, 
    "x 10\u207B\u00B9\u2077 erg s\u207B\u00B9 cm\u207B\u00B2", file=f)
    # Narrow MgII
    print('Narrow MgII:', file=f)
    print(' '  + "FWHM = " '%.2f' % fwhm_nmgII, "km s\u207B\u00B9", file=f)
    print(' '  + "\u03C3 = " '%.2f' % sigma_nmgII, "km s\u207B \u00B9", file=f)
    print(' '  + "EW = ", '%.2f' % ew_nmgII, "\u212B", file=f)
    print(' '  + "Peak = ", '%.2f' % peak_nmgII, "\u212B", file=f)
    print(' '  + "Area = ", '%.2f' % area_nmgII, 
    "x 10\u207B\u00B9\u2077 erg s\u207B\u00B9 cm\u207B\u00B2", file=f)
      
    
# -----------------------------------------------------------------------------

# Cleaing data through model subtraction to obtain a clean BL profile of the
# spectrum. This part of the code is optional

# Data subtraction and plotting 
data_subtraction ='yes'                             
if(data_subtraction == 'yes'):
    # Obtaining narrow lines from the fitted spectrum
    n_lines = np.zeros(len(q_mle.wave))
    for p in range(len(q_mle.gauss_result)//3):
        if q_mle.CalFWHM(q_mle.gauss_result[3*p+2]) < 1200: 
            na = q_mle.Onegauss(np.log(q_mle.wave), \
                                q_mle.gauss_result[p*3:(p+1)*3])
            n_lines = n_lines + na
            
    # Obtaining broad lines from the fitted spectrum
    b_lines = np.zeros(len(q_mle.wave))
    for p in range(len(q_mle.gauss_result)//3):
        if q_mle.CalFWHM(q_mle.gauss_result[3*p+2]) > 1200: 
            ba = q_mle.Onegauss(np.log(q_mle.wave), \
                                q_mle.gauss_result[p*3:(p+1)*3])
            b_lines = b_lines + ba
    
    # Calling the separate models from the fit
    data = q_mle.flux                               # Flux from SDSS .fits file
    continuum_FeII = q_mle.f_conti_model            # FeII template + continuum
    wavelength = q_mle.wave
    
    # Skip the error results before obtaining fitted line flux
    if q_mle.MCMC == True:
        gauss_result = q_mle.gauss_result[::2]
    else:
        gauss_result = q_mle.gauss_result
        
    line = q_mle.Manygauss(np.log(q_mle.wave), gauss_result) \
        + q_mle.f_conti_model

    # Performing data subtraction
    data_contFeII_sub = data - continuum_FeII
    data_sub = data - continuum_FeII - n_lines

    # Plotting cleaned data
    fig6 = plt.figure(figsize=(15,5))
    plt.plot(wavelength, data_sub, c='k', label='BL Profile')
    plt.xlabel(r'$\rm Rest \, Wavelength$ ($\rm \AA$)', fontsize=20)
    plt.ylabel(r'$\rm f_{\lambda}$ ($\rm 10^{-17} erg\;s^{-1}\;cm^{-2}\;\AA^{-1}$)', 
               fontsize=20)
    plt.title(f'{source}  z = {np.round(float(z), 4)}',
          fontsize=20)
    plt.legend()
    plt.savefig(path5+source+'_BLProfile.pdf')
    
    # Saving subtracted data into file to use for further calculations
    np.save(path5+'/Fit Data/'+source+'_DataCFeII', data_contFeII_sub)
    np.save(path5+'/Fit Data/'+source+'_Data', data)
    np.save(path5+'/Fit Data/'+source+'_Wavelength', wavelength)
    np.save(path5+'/Fit Data/'+source+'_BLSpectrum', data_sub)
    np.save(path5+'/Fit Data/'+source+'_NLData', n_lines)
    np.save(path5+'/Fit Data/'+source+'_BLData', b_lines)


# -----------------------------------------------------------------------------

