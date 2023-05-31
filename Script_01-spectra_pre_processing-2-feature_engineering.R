# Introduction ------------------------------------------------------------

# Code written to perform feature engineering in reflectance spectral
# curves of soil samples related to particle size (physical analysis),
# nitrogen and carbon (chemistry analysis). The samples were collected in
# soil profiles along the State of Pernambuco, Brazil.

# The predictors, or features, are derived from soil spectral signatures
# from Visible, Near-Infrared, Shortwave Infrared, and Middle Infrared.
# In this code Visible, Near-Infrared and Shortwave Infrared are considered
# one spectra, and the Middle Infrared other. In both, VNIR and MIR spectra,
# reflectance can be transformed by applying: continuum removal and
# spectral derivative of first and second orders; it means expanding the
# space of predictors.

# This code is part of Erli Pinto dos Santos thesis at
## Department of Agricultural Engineering, Universidade Federal de Viçosa,
## Brazil
## Author: Erli Pinto dos Santos
## Contact-me at: erlipinto@gmail.com or erlispinto@outlook.com

# Last update: May 31th, 2023

# Requested packages ------------------------------------------------------

# Helper packages
library(readxl)       # For reading Excel files

# Dealing with data
library(dplyr)        # For data wrangling
library(tibble)       # A sophysticated data frame

# Spectra processing
library(hsdar)        # For dealing with spectral curves

# Getting data ------------------------------------------------------------
# Importing Excel file with spectral curves
Vis_NIR_SWIR_MIR <- read_excel("./01_database/01_Vis_NIR_SWIR_MIR.xlsx")

# Creating a vector of wavelenghts for each spectral intervals
## VNIR
VNIR_SWIR_wavelenght_nm <- Vis_NIR_SWIR_MIR[18:2168] %>%
  names() %>% as.numeric()
## VNIR
MIR_wavelenght_nm <- Vis_NIR_SWIR_MIR[2169:5502] %>%
  names() %>% as.numeric()

# Gettin data from imported sheet and casting to matrix objects
## VNIR
spectra_VNIR_SWIR <- Vis_NIR_SWIR_MIR[18:2168] %>% as.matrix()
## MIR
spectra_MIR <- Vis_NIR_SWIR_MIR[2169:5502] %>% as.matrix()

# Creating spectral libraries for each spectral range by informing the
# wavelenght vector:

## First for VNIR
spectra_VNIR_SWIR <- speclib(spectra_VNIR_SWIR,
                             wavelength = VNIR_SWIR_wavelenght_nm) %>%
  ## Here is being applyied a mean filter with a window of size 4 
  noiseFiltering(method = "mean", p = 4) 
## And for MIR
spectra_MIR <- speclib(spectra_MIR, wavelength = MIR_wavelenght_nm) %>% 
  ## Here is being applyied a mean filter with a window of size 4 
  noiseFiltering(method = "mean", p = 4)

# Visualizing spectral curves ---------------------------------------------
# Visualizing reflectance curves in VNIR:
plot(spectra_VNIR_SWIR, main = "Reflectance: VNIR")
# And in MIR:
plot(spectra_MIR, main = "Reflectance: MIR")

# Applying continuum removal technique ------------------------------------
# Transform spectral libraries (VNIR and MIR) via continuum removal
# technique:

## VNIR
ch_ratio_VNIR_SWIR <- transformSpeclib(spectra_VNIR_SWIR, method = "ch",
                                       out = "ratio")
## MIR
ch_ratio_MIR <- transformSpeclib(spectra_MIR, method = "ch", out = "ratio")

# Visualizing spectral curves ---------------------------------------------
## Visualizing normalized reflectance curves in VNIR:
plot(ch_ratio_VNIR_SWIR, ispec = 1, numeratepoints = FALSE,
     main = "VNIR-SWIR Convex hull - Continuum line")
## And MIR:
plot(ch_ratio_MIR, ispec = 1, numeratepoints = FALSE,
     main = "MIR Convex hull - Continuum line")

# Applying derivative spectra technique -----------------------------------
# Transforming spectral libraries (VNIR and MIR) via derivative spectra
# technique. Initially using first derivative:

## For VNIR spectra:
spec_1deriv_VNIR_SWIR <- derivative.speclib(spectra_VNIR_SWIR,
                                            m = 1 # order of derivative
                                            )
## For MIR spectra:
spec_1deriv_MIR <- derivative.speclib(spectra_MIR, m = 1)

# After for second derivative:
## For VNIR spectra:
spec_2deriv_VNIR_SWIR <- derivative.speclib(spectra_VNIR_SWIR,
                                            m = 2 # order of derivative
                                            )
## For MIR spectra:
spec_2deriv_MIR <- derivative.speclib(spectra_MIR, m = 2)

# Visualizing spectral curves ---------------------------------------------
# Visualizing derivative of the reflectance curves

## First derivative of VNIR reflectance: 
plot(spec_1deriv_VNIR_SWIR, FUN = 1, main = "VNIR-SWIR First derivation")
## First derivative of MIR reflectance: 
plot(spec_1deriv_MIR, FUN = 1, main = "MIR First derivation")

## Second derivative of VNIR reflectance: 
plot(spec_2deriv_VNIR_SWIR, FUN = 1, main = "VNIR-SWIR Second derivation")
## Second derivative of MIR reflectance: 
plot(spec_2deriv_MIR, FUN = 1, main = "MIR Second derivation")

# Gathering data ----------------------------------------------------------

# Vis-NIR-SWIR
VNIR_SWIR_ch <- as_tibble(spectra(ch_ratio_VNIR_SWIR)) %>% 
  rename_with(function(x) paste("ch_",VNIR_SWIR_wavelenght_nm,"_nm", sep = ""),
              starts_with("V"))

VNIR_SWIR_std <- as_tibble(spectra(spec_1deriv_VNIR_SWIR)) %>% 
  rename_with(function(x) paste("std_",VNIR_SWIR_wavelenght_nm ,"_nm", sep = ""),
              starts_with("V"))

VNIR_SWIR_scd <- as_tibble(spectra(spec_2deriv_VNIR_SWIR)) %>% 
  rename_with(function(x) paste("scd_",VNIR_SWIR_wavelenght_nm ,"_nm", sep = ""),
              starts_with("V"))

VNIR_SWIR <- as_tibble(spectra(spectra_VNIR_SWIR)) %>% 
  rename_with(function(x) paste("p_",VNIR_SWIR_wavelenght_nm ,"_nm"),
              starts_with("V"))

Vis_NIR_SWIR <- bind_cols(Vis_NIR_SWIR_MIR[1:17], VNIR_SWIR, VNIR_SWIR_ch,
                          VNIR_SWIR_std, VNIR_SWIR_scd)
remove(VNIR_SWIR_ch, VNIR_SWIR_std, VNIR_SWIR_scd, VNIR_SWIR)

# MIR
MIR_ch <- as_tibble(spectra(ch_ratio_MIR)) %>% 
  rename_with(function(x) paste("ch_",MIR_wavelenght_nm,"_nm", sep = ""),
              starts_with("V"))

MIR_std <- as_tibble(spectra(spec_1deriv_MIR)) %>% 
  rename_with(function(x) paste("std_",MIR_wavelenght_nm ,"_nm", sep = ""),
              starts_with("V"))

MIR_scd <- as_tibble(spectra(spec_2deriv_MIR)) %>% 
  rename_with(function(x) paste("scd_",MIR_wavelenght_nm ,"_nm", sep = ""),
              starts_with("V"))

MIR <- as_tibble(spectra(spectra_MIR)) %>% 
  rename_with(function(x) paste("p_",MIR_wavelenght_nm ,"_nm"), starts_with("V"))

MIR <- bind_cols(Vis_NIR_SWIR_MIR[1:17], MIR, MIR_ch, MIR_std, MIR_scd)
remove(MIR_ch, MIR_std, MIR_scd)

# Exporting data ----------------------------------------------------------

#write_xlsx(Vis_NIR_SWIR, "./01_database/02_Vis_NIR_SWIR.xlsx", col_names = T)
#write_xlsx(MIR, "./01_database/03_MIR.xlsx",col_names = T)

save(Vis_NIR_SWIR, file = "./01_database/02_Vis_NIR_SWIR.RData")
save(MIR, file ="./01_database/03_MIR.RData")
