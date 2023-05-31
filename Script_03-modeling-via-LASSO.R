# Introduction ------------------------------------------------------------

# Code written to perform modeling of Soil Organic Carbon, in spread
# soil profiles along the State of Pernambuco, Brazil, using the easyly
# explainable method: Least Absolute Shrinkage Selection Operator (LASSO);
# which is a penalized method.

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
library(writexl)      # For writting Excel files
library(random)       # For generating true random numbers

# Dealing with data
library(tidyr)        # For data gathering
library(dplyr)        # Functions for data wrangling
library(tibble)       # A soft data frame
library(glue)         # To copy and paste

# For awesome plots
library(ggplot2)      # For data visualization

# Modelling packages
library(rsample)      # For data subsampling
library(caret)        # Machine Learning Modeling
library(CAST)         # For space-time validation and testing
library(glmnet)          # FOr partial least square models

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

# Casting data frames to arrays, required by LASSO method -----------------
## First for MIR datasets
X_MIR_LLO_training <- model.matrix(
  `C (g kg)` ~.,
  MIR_LLO_training %>% select(c(10, 14, 18:(ncol(MIR_LLO_training)-1))))[,-1]
Y_MIR_LLO_training <- MIR_LLO_training$`C (g kg)`
  
## After for VNIR datasets
X_VNIR_LLO_training <- model.matrix(
  `C (g kg)` ~.,
  VNIR_LLO_training %>% select(c(3:4, 10, 14, 18:(ncol(VNIR_LLO_training)-1))))[,-1]
Y_VNIR_LLO_training <- VNIR_LLO_training$`C (g kg)`

rm(MIR, VNIR_SWIR)

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
# Calibrating MIR based models using LASSO method and LLO cross-validation
lasso_MIR_LLO_cross_validation <- list()
lasso_MIR_LLO_best_models <- list()

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
  MIR_LASSO_LLO <- train(
    x = X_MIR_LLO_training,
    y = Y_MIR_LLO_training,
    method = "glmnet",
    preProc = c("zv", "center", "scale"),
    trControl = MIR_LLO_trCtrl,
    metric = "RMSE",
    maximize = F,
    tuneGrid = expand.grid(alpha = 1,
                           lambda = seq(from = 0, to = 2, by = 0.05))
  )
  
  ## Saving tuned models
  save(MIR_LASSO_LLO,
       file = paste("./02_tuned_models/lasso_MIR_LLO_tuned_model_",
                    i,".RData", sep = ""))
  
  ## Getting cross-validation metrics
  cv <- MIR_LASSO_LLO[["resample"]] %>% mutate(model = i)
  lasso_MIR_LLO_cross_validation[[i]] <- cv
  
  # Getting best models stats
  alphaLambda <- MIR_LASSO_LLO[["finalModel"]][["tuneValue"]]
  result <- MIR_LASSO_LLO[["results"]] %>%
    filter(lambda == alphaLambda[1,2])
  lasso_MIR_LLO_best_models[[i]] <- result %>% mutate(model = i)
  
  ## Cleaning up memory space
  rm(MIR_LLO_obj, MIR_LLO_trCtrl, MIR_LASSO_LLO, cv, alphaLambda, result)
  gc()
  
}

## Binding models results
lasso_MIR_LLO_cross_validation <- bind_rows(lasso_MIR_LLO_cross_validation)
lasso_MIR_LLO_best_models <- bind_rows(lasso_MIR_LLO_best_models)

## Writting in disc models results
save(lasso_MIR_LLO_cross_validation ,
     file = "./03_crossValidation/lasso_MIR_LLO_cross_validation.RData")
save(lasso_MIR_LLO_best_models,
     file = "./03_crossValidation/lasso_MIR_LLO_best_models.RData")
write_xlsx(lasso_MIR_LLO_cross_validation,
           "./03_crossValidation/lasso_MIR_LLO_cross_validation.xlsx",
           col_names = TRUE)
write_xlsx(lasso_MIR_LLO_best_models,
           "./03_crossValidation/lasso_MIR_LLO_best_models.xlsx",
           col_names = TRUE)

remove(lasso_MIR_LLO_cross_validation, lasso_MIR_LLO_best_models)

# Calibrating VNIR based models using LASSO method and LLO cross-validation
lasso_VNIR_LLO_cross_validation <- list()
lasso_VNIR_LLO_best_models <- list()

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
  VNIR_LASSO_LLO <- train(
    x = X_VNIR_LLO_training,
    y = Y_VNIR_LLO_training,
    method = "glmnet",
    preProc = c("zv", "center", "scale"),
    trControl = VNIR_LLO_trCtrl,
    metric = "RMSE",
    maximize = F,
    tuneGrid = expand.grid(alpha = 1,
                           lambda = seq(from = 0, to = 2, by = 0.05))
  )
  
  ## Saving tuned models
  save(VNIR_LASSO_LLO,
       file = paste("./02_tuned_models/lasso_VNIR_LLO_tuned_model_",
                    i,".RData", sep = ""))
  
  ## Getting cross-validation metrics
  cv <- VNIR_LASSO_LLO[["resample"]] %>% mutate(model = i)
  lasso_VNIR_LLO_cross_validation[[i]] <- cv
  
  # Getting best models stats
  alphaLambda <- VNIR_LASSO_LLO[["finalModel"]][["tuneValue"]]
  result <- VNIR_LASSO_LLO[["results"]] %>%
    filter(lambda == alphaLambda[1,2])
  lasso_VNIR_LLO_best_models[[i]] <- result %>% mutate(model = i)
  
  ## Cleaning up memory space
  rm(VNIR_LLO_obj, VNIR_LLO_trCtrl, VNIR_LASSO_LLO, cv, alphaLambda, result)
  gc()
  
}

## Binding models results
lasso_VNIR_LLO_cross_validation <- bind_rows(lasso_VNIR_LLO_cross_validation)
lasso_VNIR_LLO_best_models <- bind_rows(lasso_VNIR_LLO_best_models)

## Writting in disc models results
save(lasso_VNIR_LLO_cross_validation ,
     file = "./03_crossValidation/lasso_VNIR_LLO_cross_validation.RData")
save(lasso_VNIR_LLO_best_models,
     file = "./03_crossValidation/lasso_VNIR_LLO_best_models.RData")
write_xlsx(lasso_VNIR_LLO_cross_validation,
           "./03_crossValidation/lasso_VNIR_LLO_cross_validation.xlsx",
           col_names = TRUE)
write_xlsx(lasso_VNIR_LLO_best_models,
           "./03_crossValidation/lasso_VNIR_LLO_best_models.xlsx",
           col_names = TRUE)

remove(lasso_VNIR_LLO_cross_validation, lasso_VNIR_LLO_best_models)

tf <- Sys.time()

write.table(paste0("Tempo requerido pelo LASSO com LLO CV","\n",
                   "Tempo inicial = ", ti, "\n",
                   "Tempo final = ", tf, "\n",
                   "Diferença de tempo = ", (tf-ti)),
            file = "training_time_LASSO_LLO.txt")

# Training with k-Fold CV -------------------------------------------------
controlObject <- trainControl(method = "repeatedcv", number = 10,
                              repeats = 5)

ti <- Sys.time()

# Calibrating MIR based models using LASSO method and kFold cross-validation
lasso_MIR_kFold_cross_validation <- list()
lasso_MIR_kFold_best_models <- list()

for (i in 1:100) {
  ## Setting randomization seed
  set.seed(seeds[i])
  
  ## Training model
  MIR_LASSO_kFold <- train(
    x = X_MIR_LLO_training,
    y = Y_MIR_LLO_training,
    method = "glmnet",
    preProc = c("zv", "center", "scale"),
    trControl = controlObject,
    metric = "RMSE",
    maximize = F,
    tuneGrid = expand.grid(alpha = 1,
                           lambda = seq(from = 0, to = 2, by = 0.05))
  )
  
  ## Saving tuned models
  save(MIR_LASSO_kFold,
       file = paste("./02_tuned_models/lasso_MIR_kFold_tuned_model_",
                    i,".RData", sep = ""))
  
  ## Getting cross-validation metrics
  cv <- MIR_LASSO_kFold[["resample"]] %>% mutate(model = i)
  lasso_MIR_kFold_cross_validation[[i]] <- cv
  
  # Getting best models stats
  alphaLambda <- MIR_LASSO_kFold[["finalModel"]][["tuneValue"]]
  result <- MIR_LASSO_kFold[["results"]] %>%
    filter(lambda == alphaLambda[1,2])
  lasso_MIR_kFold_best_models[[i]] <- result %>% mutate(model = i)
  
  ## Cleaning up memory space
  rm(MIR_kFold_obj, MIR_kFold_trCtrl, MIR_LASSO_kFold, cv, alphaLambda, result)
  gc()
  
}

## Binding models results
lasso_MIR_kFold_cross_validation <- bind_rows(lasso_MIR_kFold_cross_validation)
lasso_MIR_kFold_best_models <- bind_rows(lasso_MIR_kFold_best_models)

## Writting in disc models results
save(lasso_MIR_kFold_cross_validation ,
     file = "./03_crossValidation/lasso_MIR_kFold_cross_validation.RData")
save(lasso_MIR_kFold_best_models,
     file = "./03_crossValidation/lasso_MIR_kFold_best_models.RData")
write_xlsx(lasso_MIR_kFold_cross_validation,
           "./03_crossValidation/lasso_MIR_kFold_cross_validation.xlsx",
           col_names = TRUE)
write_xlsx(lasso_MIR_kFold_best_models,
           "./03_crossValidation/lasso_MIR_kFold_best_models.xlsx",
           col_names = TRUE)

remove(lasso_MIR_kFold_cross_validation, lasso_MIR_kFold_best_models)

# Calibrating VNIR based models using LASSO method and kFold cross-validation
lasso_VNIR_kFold_cross_validation <- list()
lasso_VNIR_kFold_best_models <- list()

for (i in 1:100) {
  ## Setting randomization seed
  set.seed(seeds[i])
  
  ## Training model
  VNIR_LASSO_kFold <- train(
    x = X_VNIR_LLO_training,
    y = Y_VNIR_LLO_training,
    method = "glmnet",
    preProc = c("zv", "center", "scale"),
    trControl = controlObject,
    metric = "RMSE",
    maximize = F,
    tuneGrid = expand.grid(alpha = 1,
                           lambda = seq(from = 0, to = 2, by = 0.05))
  )
  
  ## Saving tuned models
  save(VNIR_LASSO_kFold,
       file = paste("./02_tuned_models/lasso_VNIR_kFold_tuned_model_",
                    i,".RData", sep = ""))
  
  ## Getting cross-validation metrics
  cv <- VNIR_LASSO_kFold[["resample"]] %>% mutate(model = i)
  lasso_VNIR_kFold_cross_validation[[i]] <- cv
  
  # Getting best models stats
  alphaLambda <- VNIR_LASSO_kFold[["finalModel"]][["tuneValue"]]
  result <- VNIR_LASSO_kFold[["results"]] %>%
    filter(lambda == alphaLambda[1,2])
  lasso_VNIR_kFold_best_models[[i]] <- result %>% mutate(model = i)
  
  ## Cleaning up memory space
  rm(VNIR_kFold_obj, VNIR_kFold_trCtrl, VNIR_LASSO_kFold, cv, alphaLambda, result)
  gc()
  
}

## Binding models results
lasso_VNIR_kFold_cross_validation <- bind_rows(lasso_VNIR_kFold_cross_validation)
lasso_VNIR_kFold_best_models <- bind_rows(lasso_VNIR_kFold_best_models)

## Writting in disc models results
save(lasso_VNIR_kFold_cross_validation ,
     file = "./03_crossValidation/lasso_VNIR_kFold_cross_validation.RData")
save(lasso_VNIR_kFold_best_models,
     file = "./03_crossValidation/lasso_VNIR_kFold_best_models.RData")
write_xlsx(lasso_VNIR_kFold_cross_validation,
           "./03_crossValidation/lasso_VNIR_kFold_cross_validation.xlsx",
           col_names = TRUE)
write_xlsx(lasso_VNIR_kFold_best_models,
           "./03_crossValidation/lasso_VNIR_kFold_best_models.xlsx",
           col_names = TRUE)

remove(lasso_VNIR_kFold_cross_validation, lasso_VNIR_kFold_best_models, controlObject)

tf <- Sys.time()

write.table(paste0("Tempo requerido pelo LASSO com k-Fold CV","\n",
                   "Tempo inicial = ", ti, "\n",
                   "Tempo final = ", tf, "\n",
                   "Diferença de tempo = ", (tf-ti)),
            file = "training_time_LASSO_kFold.txt")
