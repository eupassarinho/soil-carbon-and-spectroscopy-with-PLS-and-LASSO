# Introduction ------------------------------------------------------------

# Code written to perform modeling of Soil Organic Carbon, in spread
# soil profiles along the State of Pernambuco, Brazil, using: Partial
# Least Squares regression (PLS); which is a non-penalized and combination
# based method.

# The predictors, or features, are derived from soil spectral signatures
# from Visible, Near-Infrared, Shortwave Infrared, and Middle Infrared.
# The two types of spectra (VNIR-SWIR and MIR) were previously processed
# by applying: continuum removal and spectral derivative of first and
# second orders; it means expanding the space of predictors.

# This code is part of Erli Pinto dos Santos thesis at
## Department of Agricultural Engineering, Universidade Federal de Viçosa,
## Brazil
## Author: Erli Pinto dos Santos
## Contact-me at: erlipinto@gmail.com or erlispinto@outlook.com

# Last update: May 31th, 2023

# Requested packages ------------------------------------------------------
# Optimizating
memory.limit(size = 32000)

# Helper packages
library(readxl)       # For reading Excel files
library(writexl)      # For writting Excel files
library(random)       # For generating true random numbers

# Dealing with data
library(tidyr)        # For data gathering
library(dplyr)        # Functions for data wrangling
library(tibble)       # A soft data frame
library(lubridate)    # For dealing with dates
library(glue)         # To copy and paste

# Data visualization
library(ggplot2)      # For awesome plots!

# Modelling packages
library(rsample)      # For data subsampling
library(caret)        # Machine Learning Modeling
library(CAST)         # For space-time validation and testing
library(pls)          # FOr partial least square models

# Importing data ----------------------------------------------------------
# Getting Middle Infrared data
load("./01_database/03_MIR.RData")
# Getting Visible - Near Infrared - Shortwave Infrared data
load("./01_database/02_Vis_NIR_SWIR.RData")

# Soil profile numeric variable -------------------------------------------
# Creating a variable to identify each soil profile
# numerically
SoilProfile <- tibble(Amostra = MIR$Amostra %>% unique(),
                      Perfil = 1:(MIR$Amostra %>% unique() %>% length()))

MIR <- MIR %>% mutate(Perfil = match(MIR$Amostra, SoilProfile$Amostra))
VNIR_SWIR <- Vis_NIR_SWIR %>% mutate(Perfil = match(Vis_NIR_SWIR$Amostra,
                                                    SoilProfile$Amostra))
remove(Vis_NIR_SWIR)

# Splitting datasets ------------------------------------------------------
## Splitting datasets for Leave-Location-Out holdout testing
set.seed(256)

ProfileF <- sample(1:(MIR$Amostra %>% unique() %>% length()), 
                   round(0.8*(MIR$Amostra %>% unique() %>% length()), 0))

MIR_LLO_training <- MIR %>% dplyr::filter(Perfil %in% ProfileF)
#MIR_LLO_testing <- MIR %>% dplyr::filter(!Perfil %in% ProfileF)

set.seed(365)
ProfileF <- sample(1:(MIR$Amostra %>% unique() %>% length()), 
                   round(0.8*(MIR$Amostra %>% unique() %>% length()), 0))

VNIR_LLO_training <- VNIR_SWIR %>% dplyr::filter(Perfil %in% ProfileF)
#VNIR_LLO_testing <- VNIR_SWIR %>% dplyr::filter(!Perfil %in% ProfileF)

# Getting a vector of true random numbers ---------------------------------
# We are using the package "random" to get true random numbers,
# derived from atmospheric noise. The functions only work with
# internet connection.
#randomNumbers(n = 100,     # The number of numbers to use as randomization seeds     
#              min = 100,   # The minimum number value
#              max = 100000,# The maximum number value
#              col = 1      # The number of columns to allocate that numbers
#              )

## The true random numbers were get once and stored in the following vector:
seeds <- c(5975, 99313, 33793, 55501, 40294, 92680, 62083, 81352, 25090,
           10696, 96800, 20974, 940, 68193, 11611, 51541, 69547, 99820,
           78468, 27883, 33767, 1117, 89593, 58773, 99559, 6692, 21182,
           52077, 264, 84118, 15944, 1778, 93727, 11018, 2497, 95227, 66520,
           18800, 98853, 23193, 84471, 52279, 17489, 22773, 49651, 12605,
           70764, 17790, 43738, 87967, 38580, 10636, 22038, 4030,
           44850, 86156, 27284, 42820, 66387, 84272, 3044, 6157, 28415,
           50002, 88919, 6142, 83139, 3412, 25630, 79118, 12352, 70395,
           81905, 32101, 96607, 16639, 65378, 64884, 31661, 55190, 29096,
           72494, 85776, 80748, 60487, 8734, 17397, 76482, 88638, 92082,
           90560, 40158, 41207, 86727, 90484, 27324, 83288, 66581, 65811, 54017)

# Training with LLO-CV ----------------------------------------------------

ti <- Sys.time()
# Calibrating MIR based models using PLS method and LLO cross-validation
pls_MIR_LLO_cross_validation <- list()
pls_MIR_LLO_best_models <- list()

for (i in 1:100) {
  ## Setting randomization seed
  set.seed(seeds[i])
  
  ## Preparing for Space Folds
  MIR_LLO_obj <- CreateSpacetimeFolds(
    MIR_LLO_training,
    spacevar = "Amostra", class = "Amostra",
    k = 10)
  
  MIR_LLO_trCtrl <- trainControl(
    method = "cv",
    savePredictions = TRUE,
    index = MIR_LLO_obj$index,
    indexOut = MIR_LLO_obj$indexOut
  )
  
  ## Training model
  MIR_PLS_LLO <- train(
    `C (g kg)` ~ .,
    data = MIR_LLO_training %>%
      select(c(14, 18:(ncol(MIR_LLO_training)-1))),
    method = "pls",
    preProc = c("zv", "center", "scale"),
    trControl = MIR_LLO_trCtrl,
    metric = "RMSE",
    maximize = F,
    tuneLength = 10
  )
  
  rm(MIR_LLO_obj, MIR_LLO_trCtrl)
  
  ## Saving tuned models
  save(MIR_PLS_LLO,
       file = paste("./02_tuned_models/pls_MIR_LLO_tuned_model_",
                    i,".RData", sep = ""))
  
  ## Getting cross-validation metrics
  cv <- MIR_PLS_LLO[["resample"]] %>% mutate(model = i)
  pls_MIR_LLO_cross_validation[[i]] <- cv
  
  # Getting best models stats
  bestIter <- MIR_PLS_LLO[["finalModel"]][["bestIter"]][1,1]
  result <- MIR_PLS_LLO[["results"]] %>% filter(ncomp == bestIter)
  pls_MIR_LLO_best_models[[i]] <- result %>% mutate(model = i)
  
  ## Cleaning up memory space
  rm(MIR_PLS_LLO, cv, bestIter, result)
  gc()
  
}

## Binding models results
pls_MIR_LLO_cross_validation <- bind_rows(pls_MIR_LLO_cross_validation)
pls_MIR_LLO_best_models <- bind_rows(pls_MIR_LLO_best_models)

## Writting in disc models results
save(pls_MIR_LLO_cross_validation ,
     file = "./03_crossValidation/pls_MIR_LLO_cross_validation.RData")
save(pls_MIR_LLO_best_models,
     file = "./03_crossValidation/pls_MIR_LLO_best_models.RData")
write_xlsx(pls_MIR_LLO_cross_validation,
           "./03_crossValidation/pls_MIR_LLO_cross_validation.xlsx",
           col_names = TRUE)
write_xlsx(pls_MIR_LLO_best_models,
           "./03_crossValidation/pls_MIR_LLO_best_models.xlsx",
           col_names = TRUE)

remove(pls_MIR_LLO_cross_validation, pls_MIR_LLO_best_models)

# Calibrating VNIR based models using PLS method and LLO cross-validation
pls_VNIR_LLO_cross_validation <- list()
pls_VNIR_LLO_best_models <- list()

for (i in 1:100) {
  ## Setting randomization seed
  set.seed(seeds[i])
  
  ## Preparing for Space Folds
  VNIR_LLO_obj <- CreateSpacetimeFolds(
    VNIR_LLO_training,
    spacevar = "Amostra", class = "Amostra",
    k = 10)
  
  VNIR_LLO_trCtrl <- trainControl(
    method = "cv",
    savePredictions = TRUE,
    index = VNIR_LLO_obj$index,
    indexOut = VNIR_LLO_obj$indexOut
  )
  
  ## Training model
  VNIR_PLS_LLO <- train(
    `C (g kg)` ~ .,
    data = VNIR_LLO_training %>%
      select(c(14, 18:(ncol(VNIR_LLO_training)-1))),
    method = "pls",
    preProc = c("zv", "center", "scale"),
    trControl = VNIR_LLO_trCtrl,
    metric = "RMSE",
    maximize = F,
    tuneLength = 10
  )
  
  rm(VNIR_LLO_obj, VNIR_LLO_trCtrl)
  
  ## Saving tuned models
  save(VNIR_PLS_LLO,
       file = paste("./02_tuned_models/pls_VNIR_LLO_tuned_model_",
                    i,".RData", sep = ""))
  
  ## Getting cross-validation metrics
  cv <- VNIR_PLS_LLO[["resample"]] %>% mutate(model = i)
  pls_VNIR_LLO_cross_validation[[i]] <- cv
  
  # Getting best models stats
  bestIter <- VNIR_PLS_LLO[["finalModel"]][["bestIter"]][1,1]
  result <- VNIR_PLS_LLO[["results"]] %>% filter(ncomp == bestIter)
  pls_VNIR_LLO_best_models[[i]] <- result %>% mutate(model = i)
  
  ## Cleaning up memory space
  rm(VNIR_PLS_LLO, cv, bestIter, result)
  gc()
  
}

## Binding models results
pls_VNIR_LLO_cross_validation <- bind_rows(pls_VNIR_LLO_cross_validation)
pls_VNIR_LLO_best_models <- bind_rows(pls_VNIR_LLO_best_models)

## Writting in disc models results
save(pls_VNIR_LLO_cross_validation ,
     file = "./03_crossValidation/pls_VNIR_LLO_cross_validation.RData")
save(pls_VNIR_LLO_best_models,
     file = "./03_crossValidation/pls_VNIR_LLO_best_models.RData")
write_xlsx(pls_VNIR_LLO_cross_validation,
           "./03_crossValidation/pls_VNIR_LLO_cross_validation.xlsx",
           col_names = TRUE)
write_xlsx(pls_VNIR_LLO_best_models,
           "./03_crossValidation/pls_VNIR_LLO_best_models.xlsx",
           col_names = TRUE)

remove(pls_VNIR_LLO_cross_validation, pls_VNIR_LLO_best_models)

tf <- Sys.time()

write.table(paste0("Tempo requerido pelo PLS com LLO CV","\n",
                   "Tempo inicial = ", ti, "\n",
                   "Tempo final = ", tf, "\n",
                   "Diferença de tempo = ", (tf-ti)),
            file = "training_time_PLS_LLO.txt")

# Training with k-Fold CV -------------------------------------------------
controlObject <- trainControl(method = "repeatedcv", number = 10,
                              repeats = 5)

ti <- Sys.time()
# Calibrating MIR based models using PLS method and k-Fold cross-validation
pls_MIR_kFold_cross_validation <- list()
pls_MIR_kFold_best_models <- list()

for (i in 1:100) {
  ## Setting randomization seed
  set.seed(seeds[i])
  
  ## Training model
  MIR_PLS_kFold <- train(
    `C (g kg)` ~ .,
    data = MIR_LLO_training %>%
      select(c(14, 18:(ncol(MIR_LLO_training)-1))),
    method = "pls",
    preProc = c("zv", "center", "scale"),
    trControl = controlObject,
    metric = "RMSE",
    maximize = F,
    tuneLength = 10
  )
  
  ## Saving tuned models
  save(MIR_PLS_kFold,
       file = paste("./02_tuned_models/pls_MIR_kFold_tuned_model_",
                    i,".RData", sep = ""))
  
  ## Getting cross-validation metrics
  cv <- MIR_PLS_kFold[["resample"]] %>% mutate(model = i)
  pls_MIR_kFold_cross_validation[[i]] <- cv
  
  # Getting best models stats
  bestIter <- MIR_PLS_kFold[["finalModel"]][["bestIter"]][1,1]
  result <- MIR_PLS_kFold[["results"]] %>% filter(ncomp == bestIter)
  pls_MIR_kFold_best_models[[i]] <- result %>% mutate(model = i)
  
  ## Cleaning up memory space
  rm(MIR_PLS_kFold, cv, bestIter, result)
  gc()
  
}

## Binding models results
pls_MIR_kFold_cross_validation <- bind_rows(pls_MIR_kFold_cross_validation)
pls_MIR_kFold_best_models <- bind_rows(pls_MIR_kFold_best_models)

## Writting in disc models results
save(pls_MIR_kFold_cross_validation ,
     file = "./03_crossValidation/pls_MIR_kFold_cross_validation.RData")
save(pls_MIR_kFold_best_models,
     file = "./03_crossValidation/pls_MIR_kFold_best_models.RData")
write_xlsx(pls_MIR_kFold_cross_validation,
           "./03_crossValidation/pls_MIR_kFold_cross_validation.xlsx",
           col_names = TRUE)
write_xlsx(pls_MIR_kFold_best_models,
           "./03_crossValidation/pls_MIR_kFold_best_models.xlsx",
           col_names = TRUE)

remove(pls_MIR_kFold_cross_validation, pls_MIR_kFold_best_models)

# Calibrating VNIR based models using PLS method and k-Fold cross-validation
pls_VNIR_kFold_cross_validation <- list()
pls_VNIR_kFold_best_models <- list()

for (i in 3:100) {
  ## Setting randomization seed
  set.seed(seeds[i])
  
  ## Training model
  VNIR_PLS_kFold <- train(
    `C (g kg)` ~ .,
    data = VNIR_LLO_training %>%
      select(c(14, 18:(ncol(VNIR_LLO_training)-1))),
    method = "pls",
    preProc = c("zv", "center", "scale"),
    trControl = controlObject,
    metric = "RMSE",
    maximize = F,
    tuneLength = 10
  )
  
  ## Saving tuned models
  save(VNIR_PLS_kFold,
       file = paste("./02_tuned_models/pls_VNIR_kFold_tuned_model_",
                    i,".RData", sep = ""))
  
  ## Getting cross-validation metrics
  cv <- VNIR_PLS_kFold[["resample"]] %>% mutate(model = i)
  pls_VNIR_kFold_cross_validation[[i]] <- cv
  
  # Getting best models stats
  bestIter <- VNIR_PLS_kFold[["finalModel"]][["bestIter"]][1,1]
  result <- VNIR_PLS_kFold[["results"]] %>% filter(ncomp == bestIter)
  pls_VNIR_kFold_best_models[[i]] <- result %>% mutate(model = i)
  
  ## Cleaning up memory space
  rm(VNIR_PLS_kFold, cv, bestIter, result)
  gc()
  
}

## Binding models results
pls_VNIR_kFold_cross_validation <- bind_rows(pls_VNIR_kFold_cross_validation)
pls_VNIR_kFold_best_models <- bind_rows(pls_VNIR_kFold_best_models)

## Writting in disc models results
save(pls_VNIR_kFold_cross_validation ,
     file = "./03_crossValidation/pls_VNIR_kFold_cross_validation.RData")
save(pls_VNIR_kFold_best_models,
     file = "./03_crossValidation/pls_VNIR_kFold_best_models.RData")
write_xlsx(pls_VNIR_kFold_cross_validation,
           "./03_crossValidation/pls_VNIR_kFold_cross_validation.xlsx",
           col_names = TRUE)
write_xlsx(pls_VNIR_kFold_best_models,
           "./03_crossValidation/pls_VNIR_kFold_best_models.xlsx",
           col_names = TRUE)

remove(pls_VNIR_kFold_cross_validation, pls_VNIR_kFold_best_models,
       controlObject)

tf <- Sys.time()

write.table(paste0("Tempo requerido pelo PLS com k-Fold CV","\n",
                   "Tempo inicial = ", ti, "\n",
                   "Tempo final = ", tf, "\n",
                   "Diferença de tempo = ", (tf-ti)),
            file = "training_time_PLS_kFold.txt")
