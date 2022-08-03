
### Static Correlators ####

## makeM2.py

# Defining the files to process:

The routines getNvalues(), getm2values(), getloader()

#  run vev2vsN() 

This computes the mean value <M2>  of lattice size, producing
the file, e.g. "vevscan_Nxxx_m-0482360_h000000_c00500_M2.txt",  for all values
of m2

# run makeallfits().  

Do the Hasenfratz fits using the routine makeallfits(). This takes the files such as "filename_M2.txt" and produces fits the results "filename_fit1.txt" and "filename_fit2.txt".  The fit parameters are in "vevscan_Nxxx_allfits.txt".  This loops through  the masses, does the fits etc

## Other files for making plots

# vevfits.gpi 

Makes a plot of fit1 and fit2 resulting from makeallfits()

# vevscan.gpi 

Plots sigma2 vs m2.


