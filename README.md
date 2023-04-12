# RBH-Scripts
Python scripts for calculating BL profiles of SDSS quasar spectra.

These codes make use of "PyQSOFit: A code to fit the spectrum of quasar", found at https://github.com/legolason/PyQSOFit.git

Fitting-Script.py is used to run PyQSOFit after an install of the source code. More specifically, it is used with the SDSS DR7 Quasar Catalog. 
It will fit the spectrum of a quasar by taking a FITS file and running it with PyQSOFit. It also includes plotting of the fitted line complexes 
separately, saving fitted line properties to a text tile, and (optional) the cleaning of the data so that only the broad-line (BL) profile is left. 
The fitted data is stored separately to be analyzed by LineProfile_Calc.py, a code that calculates velocity shifts and characteristic line profile 
shapes.

LineProfile_Calc.py analyzes the fitted broad-line (BL) compnents of several line complexes from the results of Fitting-Script.py for PyQSOFit. 
It calculates the peak, centroid, line center at 80% of the area (C80, from Whittle 1985) velocity shifts. In addition, it calculates characteristic 
line profile shapes using the area parameters proposed by Whittle 1985. 
