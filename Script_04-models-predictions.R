# Introduction ------------------------------------------------------------

# Code written to perform prediction of Soil Organic Carbon in testing
# datasets with calibrated models in Script 02 and Script 03, that are of,
# respectively, PLS and LASSO methods.

# This code is part of Erli Pinto dos Santos thesis at
## Department of Agricultural Engineering, Universidade Federal de Viçosa,
## Brazil
## Author: Erli Pinto dos Santos
## Contact-me at: erlipinto@gmail.com or erlispinto@outlook.com

# Last update: May 31th, 2023

# Requested packages ------------------------------------------------------
# Optimizating
library(doParallel)   # For parallel processing

memory.limit(size = 99999999999)
n_Core <- detectCores()-2
cl <- makePSOCKcluster(n_Core)
registerDoParallel(cl)

# Helper packages
library(readxl)       # For reading Excel files
library(writexl)      # For writting Excel files

# Dealing with data
library(tidyr)        # For data gathering
library(dplyr)        # Functions for data wrangling
library(tibble)       # A soft data frame
library(glue)         # To copy and paste

# Data visualization
library(ggplot2)      # For awesome plots!

# Modelling packages
library(rsample)      # For data subsampling
library(vip)          # For getting variable importance measurements

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

#MIR_LSPO_training <- MIR %>% dplyr::filter(Perfil %in% ProfileF)
MIR_LSPO_testing <- MIR %>% dplyr::filter(!Perfil %in% ProfileF)

set.seed(365)
ProfileF <- sample(1:(MIR$Amostra %>% unique() %>% length()), 
                   round(0.8*(MIR$Amostra %>% unique() %>% length()), 0))

#VNIR_LSPO_training <- VNIR_SWIR %>% dplyr::filter(Perfil %in% ProfileF)
VNIR_LSPO_testing <- VNIR_SWIR %>% dplyr::filter(!Perfil %in% ProfileF)

# Casting data frames to arrays, required by LASSO method -----------------
## First for MIR datasets
X_MIR_LSPO_testing <- model.matrix(
  `C (g kg)` ~.,
  MIR_LSPO_testing %>% select(c(10, 14, 18:(ncol(MIR_LSPO_testing)-1))))[,-1]
Y_MIR_LSPO_testing <- MIR_LSPO_testing$`C (g kg)`

## After for VNIR datasets
X_VNIR_LSPO_testing <- model.matrix(
  `C (g kg)` ~.,
  VNIR_LSPO_testing %>% select(c(3:4, 10, 14, 18:(ncol(VNIR_LSPO_testing)-1))))[,-1]
Y_VNIR_LSPO_testing <- VNIR_LSPO_testing$`C (g kg)`

rm(MIR, VNIR_SWIR)

# Predicting carbon with the trained models -------------------------------
# Predicting with MIR PLS LSPO models --------------------------------------

ti <- Sys.time()

list_MIR_PLS_LSPO <- list.files("./02_tuned_models/",
                               pattern = "pls_MIR_LSPO_tuned_model")

## Part 1
pred_MIR_PLS_LSPO_1 <- list()

for (i in 1:50) {
  load(paste("./02_tuned_models/",list_MIR_PLS_LSPO[i], sep = ""))
  var_name <- strsplit(list_MIR_PLS_LSPO[i], '\\.')[[1]][1]
  
  pred_MIR_PLS_LSPO_1[[i]] <- tibble(SOC_model_ = predict(MIR_PLS_LSPO,
                                                         MIR_LSPO_testing)) %>%
    rename_with(.fn = ~ glue("{var_name}"))
  # Cleaning up memory
  rm(MIR_PLS_LSPO, var_name)
}
pred_MIR_PLS_LSPO_1 <- bind_cols(MIR_LSPO_testing[,1:17], pred_MIR_PLS_LSPO)
save(pred_MIR_PLS_LSPO,
     file = "./04_modelsPredictions_vip/pred_MIR_PLS_LSPO.RData")
rm(list_MIR_PLS_LSPO)

list_MIR_PLS_LSPO <- list.files("./02_tuned_models/",
                               pattern = "pls_MIR_LSPO_tuned_model")

## Part 2
pred_MIR_PLS_LSPO_2 <- list()

for (i in 51:75) {
  load(paste("./02_tuned_models/",list_MIR_PLS_LSPO[i], sep = ""))
  var_name <- strsplit(list_MIR_PLS_LSPO[i], '\\.')[[1]][1]
  
  pred_MIR_PLS_LSPO_2[[i]] <- tibble(
    SOC_model_ = predict(MIR_PLS_LSPO, MIR_LSPO_testing)) %>%
    rename_with(.fn = ~ glue("{var_name}"))
  # Cleaning up memory
  rm(MIR_PLS_LSPO, var_name)
}
pred_MIR_PLS_LSPO_2  <- bind_cols(MIR_LSPO_testing[,1:17], pred_MIR_PLS_LSPO_2)
save(pred_MIR_PLS_LSPO_2, 
     file = "./04_modelsPredictions_vip/pred_MIR_PLS_LSPO_2.RData")
rm(list_MIR_PLS_LSPO)

list_MIR_PLS_LSPO <- list.files("./02_tuned_models/",
                               pattern = "pls_MIR_LSPO_tuned_model")

## Part 3
pred_MIR_PLS_LSPO_3 <- list()
for (i in 76:100) {
  load(paste("./02_tuned_models/",list_MIR_PLS_LSPO[i], sep = ""))
  var_name <- strsplit(list_MIR_PLS_LSPO[i], '\\.')[[1]][1]
  
  pred_MIR_PLS_LSPO_3[[i]] <- tibble(
    SOC_model_ = predict(MIR_PLS_LSPO, MIR_LSPO_testing)) %>%
    rename_with(.fn = ~ glue("{var_name}"))
  # Cleaning up memory
  rm(MIR_PLS_LSPO, var_name)
}

pred_MIR_PLS_LSPO_3  <- bind_cols(MIR_LSPO_testing[,1:17], pred_MIR_PLS_LSPO_3)
save(pred_MIR_PLS_LSPO_3, 
     file = "./04_modelsPredictions_vip/pred_MIR_PLS_LSPO_3.RData")

rm(list_MIR_PLS_LSPO)

###########################################################################
tf  <- Sys.time()
tf - ti

test_MIR_PLS_LSPO <- bind_cols(pred_MIR_PLS_LSPO_2 %>% select(1:17),
                bind_cols(pred_MIR_PLS_LSPO_1),
                pred_MIR_PLS_LSPO_2 %>% select(18:42),
                pred_MIR_PLS_LSPO_3 %>% select(18:42))

write_xlsx(test_MIR_PLS_LSPO,
           "./04_modelsPredictions_vip/test_MIR_PLS_LSPO.xlsx")

# Predicting with MIR PLS k-Fold models -----------------------------------

ti <- Sys.time()

list_MIR_PLS_kFold <- list.files("./02_tuned_models_kFold/",
                                 pattern = "pls_MIR_kFold_tuned_model")

## Part 1
pred_MIR_PLS_kFold_1 <- list()

for (i in 1:50) {
  load(paste("./02_tuned_models_kFold/",list_MIR_PLS_kFold[i], sep = ""))
  var_name <- strsplit(list_MIR_PLS_kFold[i], '\\.')[[1]][1]
  
  pred_MIR_PLS_kFold_1[[i]] <- tibble(SOC_model_ = predict(MIR_PLS_kFold,
                                                           MIR_LSPO_testing)) %>%
    rename_with(.fn = ~ glue("{var_name}"))
  # Cleaning up memory
  rm(MIR_PLS_kFold, var_name)
}
pred_MIR_PLS_kFold_1 <- bind_cols(MIR_LSPO_testing[,1:17], pred_MIR_PLS_kFold_1)
save(pred_MIR_PLS_kFold_1,
     file = "./04_modelsPredictions_vip/pred_MIR_PLS_kFold.RData")

## Part 2
pred_MIR_PLS_kFold_2 <- list()

for (i in 51:75) {
  load(paste("./02_tuned_models_kFold/",list_MIR_PLS_kFold[i], sep = ""))
  var_name <- strsplit(list_MIR_PLS_kFold[i], '\\.')[[1]][1]
  
  pred_MIR_PLS_kFold_2[[i]] <- tibble(
    SOC_model_ = predict(MIR_PLS_kFold, MIR_LSPO_testing)) %>%
    rename_with(.fn = ~ glue("{var_name}"))
  # Cleaning up memory
  rm(MIR_PLS_kFold, var_name)
}
pred_MIR_PLS_kFold_2  <- bind_cols(MIR_LSPO_testing[,1:17], pred_MIR_PLS_kFold_2)
save(pred_MIR_PLS_kFold_2, 
     file = "./04_modelsPredictions_vip/pred_MIR_PLS_kFold_2.RData")

## Part 3
pred_MIR_PLS_kFold_3 <- list()
for (i in 76:100) {
  load(paste("./02_tuned_models_kFold/",list_MIR_PLS_kFold[i], sep = ""))
  var_name <- strsplit(list_MIR_PLS_kFold[i], '\\.')[[1]][1]
  
  pred_MIR_PLS_kFold_3[[i]] <- tibble(
    SOC_model_ = predict(MIR_PLS_kFold, MIR_LSPO_testing)) %>%
    rename_with(.fn = ~ glue("{var_name}"))
  # Cleaning up memory
  rm(MIR_PLS_kFold, var_name)
}

pred_MIR_PLS_kFold_3  <- bind_cols(MIR_LSPO_testing[,1:17], pred_MIR_PLS_kFold_3)
save(pred_MIR_PLS_kFold_3, 
     file = "./04_modelsPredictions_vip/pred_MIR_PLS_kFold_3.RData")

rm(list_MIR_PLS_kFold)

###########################################################################
tf  <- Sys.time()
tf - ti

test_MIR_PLS_kFold <- bind_cols(pred_MIR_PLS_kFold_1,
                                pred_MIR_PLS_kFold_2 %>% select(18:42),
                                pred_MIR_PLS_kFold_3 %>% select(18:42))

write_xlsx(test_MIR_PLS_kFold,
           "./04_modelsPredictions_vip/test_MIR_PLS_kFold.xlsx")

# Predicting with VNIR PLS LSPO models --------------------------------------

ti <- Sys.time()

list_VNIR_PLS_LSPO <- list.files("./02_tuned_models/",
                               pattern = "pls_VNIR_LSPO_tuned_model")

## Part 1
pred_VNIR_PLS_LSPO_1 <- list()

for (i in 1:50) {
  load(paste("./02_tuned_models/",list_VNIR_PLS_LSPO[i], sep = ""))
  var_name <- strsplit(list_VNIR_PLS_LSPO[i], '\\.')[[1]][1]
  
  pred_VNIR_PLS_LSPO_1[[i]] <- tibble(SOC_model_ = predict(VNIR_PLS_LSPO,
                                                         VNIR_LSPO_testing)) %>%
    rename_with(.fn = ~ glue("{var_name}"))
  # Cleaning up memory
  rm(VNIR_PLS_LSPO, var_name)
}
pred_VNIR_PLS_LSPO_1 <- bind_cols(VNIR_LSPO_testing[,1:17], pred_VNIR_PLS_LSPO)
save(pred_VNIR_PLS_LSPO,
     file = "./04_modelsPredictions_vip/pred_VNIR_PLS_LSPO.RData")
rm(list_VNIR_PLS_LSPO)

list_VNIR_PLS_LSPO <- list.files("./02_tuned_models/",
                               pattern = "pls_VNIR_LSPO_tuned_model")

## Part 2
pred_VNIR_PLS_LSPO_2 <- list()

for (i in 51:75) {
  load(paste("./02_tuned_models/",list_VNIR_PLS_LSPO[i], sep = ""))
  var_name <- strsplit(list_VNIR_PLS_LSPO[i], '\\.')[[1]][1]
  
  pred_VNIR_PLS_LSPO_2[[i]] <- tibble(
    SOC_model_ = predict(VNIR_PLS_LSPO, VNIR_LSPO_testing)) %>%
    rename_with(.fn = ~ glue("{var_name}"))
  # Cleaning up memory
  rm(VNIR_PLS_LSPO, var_name)
}
pred_VNIR_PLS_LSPO_2  <- bind_cols(VNIR_LSPO_testing[,1:17], pred_VNIR_PLS_LSPO_2)
save(pred_VNIR_PLS_LSPO_2, 
     file = "./04_modelsPredictions_vip/pred_VNIR_PLS_LSPO_2.RData")
rm(list_VNIR_PLS_LSPO)

list_VNIR_PLS_LSPO <- list.files("./02_tuned_models/",
                               pattern = "pls_VNIR_LSPO_tuned_model")

## Part 3
pred_VNIR_PLS_LSPO_3 <- list()
for (i in 76:100) {
  load(paste("./02_tuned_models/",list_VNIR_PLS_LSPO[i], sep = ""))
  var_name <- strsplit(list_VNIR_PLS_LSPO[i], '\\.')[[1]][1]
  
  pred_VNIR_PLS_LSPO_3[[i]] <- tibble(
    SOC_model_ = predict(VNIR_PLS_LSPO, VNIR_LSPO_testing)) %>%
    rename_with(.fn = ~ glue("{var_name}"))
  # Cleaning up memory
  rm(VNIR_PLS_LSPO, var_name)
}

pred_VNIR_PLS_LSPO_3  <- bind_cols(VNIR_LSPO_testing[,1:17], pred_VNIR_PLS_LSPO_3)
save(pred_VNIR_PLS_LSPO_3, 
     file = "./04_modelsPredictions_vip/pred_VNIR_PLS_LSPO_3.RData")

rm(list_VNIR_PLS_LSPO)

###########################################################################
tf  <- Sys.time()
tf - ti

test_VNIR_PLS_LSPO <- bind_cols(pred_VNIR_PLS_LSPO_2 %>% select(1:17),
                              bind_cols(pred_VNIR_PLS_LSPO_1),
                              pred_VNIR_PLS_LSPO_2 %>% select(18:42),
                              pred_VNIR_PLS_LSPO_3 %>% select(18:42))

write_xlsx(test_VNIR_PLS_LSPO,
           "./04_modelsPredictions_vip/test_VNIR_PLS_LSPO.xlsx")

# Predicting with VNIR PLS k-Fold models -----------------------------------

ti <- Sys.time()

list_VNIR_PLS_kFold <- list.files("./02_tuned_models_kFold/",
                                 pattern = "pls_VNIR_kFold_tuned_model")

## Part 1
pred_VNIR_PLS_kFold_1 <- list()

for (i in 1:50) {
  load(paste("./02_tuned_models_kFold/",list_VNIR_PLS_kFold[i], sep = ""))
  var_name <- strsplit(list_VNIR_PLS_kFold[i], '\\.')[[1]][1]
  
  pred_VNIR_PLS_kFold_1[[i]] <- tibble(SOC_model_ = predict(VNIR_PLS_kFold,
                                                           VNIR_LSPO_testing)) %>%
    rename_with(.fn = ~ glue("{var_name}"))
  # Cleaning up memory
  rm(VNIR_PLS_kFold, var_name)
}
pred_VNIR_PLS_kFold_1 <- bind_cols(VNIR_LSPO_testing[,1:17], pred_VNIR_PLS_kFold_1)
save(pred_VNIR_PLS_kFold_1,
     file = "./04_modelsPredictions_vip/pred_VNIR_PLS_kFold.RData")

## Part 2
pred_VNIR_PLS_kFold_2 <- list()

for (i in 51:75) {
  load(paste("./02_tuned_models_kFold/",list_VNIR_PLS_kFold[i], sep = ""))
  var_name <- strsplit(list_VNIR_PLS_kFold[i], '\\.')[[1]][1]
  
  pred_VNIR_PLS_kFold_2[[i]] <- tibble(
    SOC_model_ = predict(VNIR_PLS_kFold, VNIR_LSPO_testing)) %>%
    rename_with(.fn = ~ glue("{var_name}"))
  # Cleaning up memory
  rm(VNIR_PLS_kFold, var_name)
}
pred_VNIR_PLS_kFold_2  <- bind_cols(VNIR_LSPO_testing[,1:17], pred_VNIR_PLS_kFold_2)
save(pred_VNIR_PLS_kFold_2, 
     file = "./04_modelsPredictions_vip/pred_VNIR_PLS_kFold_2.RData")

## Part 3
pred_VNIR_PLS_kFold_3 <- list()
for (i in 76:100) {
  load(paste("./02_tuned_models_kFold/",list_VNIR_PLS_kFold[i], sep = ""))
  var_name <- strsplit(list_VNIR_PLS_kFold[i], '\\.')[[1]][1]
  
  pred_VNIR_PLS_kFold_3[[i]] <- tibble(
    SOC_model_ = predict(VNIR_PLS_kFold, VNIR_LSPO_testing)) %>%
    rename_with(.fn = ~ glue("{var_name}"))
  # Cleaning up memory
  rm(VNIR_PLS_kFold, var_name)
}

pred_VNIR_PLS_kFold_3  <- bind_cols(VNIR_LSPO_testing[,1:17], pred_VNIR_PLS_kFold_3)
save(pred_VNIR_PLS_kFold_3, 
     file = "./04_modelsPredictions_vip/pred_VNIR_PLS_kFold_3.RData")

rm(list_VNIR_PLS_kFold)

###########################################################################
tf  <- Sys.time()
tf - ti

test_VNIR_PLS_kFold <- bind_cols(pred_VNIR_PLS_kFold_1,
                                pred_VNIR_PLS_kFold_2 %>% select(18:42),
                                pred_VNIR_PLS_kFold_3 %>% select(18:42))

write_xlsx(test_VNIR_PLS_kFold,
           "./04_modelsPredictions_vip/test_VNIR_PLS_kFold.xlsx")

# Predicting with MIR LASSO LSPO models ------------------------------------

ti <- Sys.time()

list_MIR_LASSO_LSPO <- list.files("./02_tuned_models/",
                                 pattern = "lasso_MIR_LSPO_tuned_model")

## Part 1
pred_MIR_LASSO_LSPO_1 <- list()

for (i in 1:50) {
  load(paste("./02_tuned_models/",list_MIR_LASSO_LSPO[i], sep = ""))
  var_name <- strsplit(list_MIR_LASSO_LSPO[i], '\\.')[[1]][1]
  
  pred_MIR_LASSO_LSPO_1[[i]] <- tibble(SOC_model_ = predict(MIR_LASSO_LSPO,
                                                           X_MIR_LSPO_testing)) %>%
    rename_with(.fn = ~ glue("{var_name}"))
  # Cleaning up memory
  rm(MIR_LASSO_LSPO, var_name)
}
pred_MIR_LASSO_LSPO_1 <- bind_cols(MIR_LSPO_testing[,1:17], pred_MIR_LASSO_LSPO_1)
save(pred_MIR_LASSO_LSPO_1,
     file = "./04_modelsPredictions_vip/pred_MIR_LASSO_LSPO.RData")

## Part 2
pred_MIR_LASSO_LSPO_2 <- list()

for (i in 51:75) {
  load(paste("./02_tuned_models/",list_MIR_LASSO_LSPO[i], sep = ""))
  var_name <- strsplit(list_MIR_LASSO_LSPO[i], '\\.')[[1]][1]
  
  pred_MIR_LASSO_LSPO_2[[i]] <- tibble(
    SOC_model_ = predict(MIR_LASSO_LSPO, X_MIR_LSPO_testing)) %>%
    rename_with(.fn = ~ glue("{var_name}"))
  # Cleaning up memory
  rm(MIR_LASSO_LSPO, var_name)
}
pred_MIR_LASSO_LSPO_2  <- bind_cols(MIR_LSPO_testing[,1:17], pred_MIR_LASSO_LSPO_2)
save(pred_MIR_LASSO_LSPO_2, 
     file = "./04_modelsPredictions_vip/pred_MIR_LASSO_LSPO_2.RData")

## Part 3
pred_MIR_LASSO_LSPO_3 <- list()
for (i in 76:100) {
  load(paste("./02_tuned_models/",list_MIR_LASSO_LSPO[i], sep = ""))
  var_name <- strsplit(list_MIR_LASSO_LSPO[i], '\\.')[[1]][1]
  
  pred_MIR_LASSO_LSPO_3[[i]] <- tibble(
    SOC_model_ = predict(MIR_LASSO_LSPO, X_MIR_LSPO_testing)) %>%
    rename_with(.fn = ~ glue("{var_name}"))
  # Cleaning up memory
  rm(MIR_LASSO_LSPO, var_name)
}

pred_MIR_LASSO_LSPO_3  <- bind_cols(MIR_LSPO_testing[,1:17], pred_MIR_LASSO_LSPO_3)
save(pred_MIR_LASSO_LSPO_3, 
     file = "./04_modelsPredictions_vip/pred_MIR_LASSO_LSPO_3.RData")

rm(list_MIR_LASSO_LSPO)

###########################################################################
tf  <- Sys.time()
tf - ti

test_MIR_LASSO_LSPO <- bind_cols(pred_MIR_LASSO_LSPO_1,
                                pred_MIR_LASSO_LSPO_2 %>% select(18:42),
                                pred_MIR_LASSO_LSPO_3 %>% select(18:42))

write_xlsx(test_MIR_LASSO_LSPO,
           "./04_modelsPredictions_vip/test_MIR_LASSO_LSPO.xlsx")

# Predicting with MIR LASSO kFold models ------------------------------------

ti <- Sys.time()

list_MIR_LASSO_kFold <- list.files("./02_tuned_models_kFold/",
                                 pattern = "lasso_MIR_kFold_tuned_model")

## Part 1
pred_MIR_LASSO_kFold_1 <- list()

for (i in 1:50) {
  load(paste("./02_tuned_models_kFold/",list_MIR_LASSO_kFold[i], sep = ""))
  var_name <- strsplit(list_MIR_LASSO_kFold[i], '\\.')[[1]][1]
  
  pred_MIR_LASSO_kFold_1[[i]] <- tibble(SOC_model_ = predict(MIR_LASSO_kFold,
                                                           X_MIR_LSPO_testing)) %>%
    rename_with(.fn = ~ glue("{var_name}"))
  # Cleaning up memory
  rm(MIR_LASSO_kFold, var_name)
}
pred_MIR_LASSO_kFold_1 <- bind_cols(MIR_LSPO_testing[,1:17], pred_MIR_LASSO_kFold_1)
save(pred_MIR_LASSO_kFold_1,
     file = "./04_modelsPredictions_vip/pred_MIR_LASSO_kFold_1.RData")

## Part 2
pred_MIR_LASSO_kFold_2 <- list()

for (i in 51:75) {
  load(paste("./02_tuned_models_kFold/",list_MIR_LASSO_kFold[i], sep = ""))
  var_name <- strsplit(list_MIR_LASSO_kFold[i], '\\.')[[1]][1]
  
  pred_MIR_LASSO_kFold_2[[i]] <- tibble(
    SOC_model_ = predict(MIR_LASSO_kFold, X_MIR_LSPO_testing)) %>%
    rename_with(.fn = ~ glue("{var_name}"))
  # Cleaning up memory
  rm(MIR_LASSO_kFold, var_name)
}
pred_MIR_LASSO_kFold_2  <- bind_cols(MIR_LSPO_testing[,1:17], pred_MIR_LASSO_kFold_2)
save(pred_MIR_LASSO_kFold_2, 
     file = "./04_modelsPredictions_vip/pred_MIR_LASSO_kFold_2.RData")

## Part 3
pred_MIR_LASSO_kFold_3 <- list()
for (i in 76:100) {
  load(paste("./02_tuned_models_kFold/",list_MIR_LASSO_kFold[i], sep = ""))
  var_name <- strsplit(list_MIR_LASSO_kFold[i], '\\.')[[1]][1]
  
  pred_MIR_LASSO_kFold_3[[i]] <- tibble(
    SOC_model_ = predict(MIR_LASSO_kFold, X_MIR_LSPO_testing)) %>%
    rename_with(.fn = ~ glue("{var_name}"))
  # Cleaning up memory
  rm(MIR_LASSO_kFold, var_name)
}

pred_MIR_LASSO_kFold_3  <- bind_cols(MIR_LSPO_testing[,1:17], pred_MIR_LASSO_kFold_3)
save(pred_MIR_LASSO_kFold_3, 
     file = "./04_modelsPredictions_vip/pred_MIR_LASSO_kFold_3.RData")

rm(list_MIR_LASSO_kFold)

###########################################################################
tf  <- Sys.time()
tf - ti

test_MIR_LASSO_kFold <- bind_cols(pred_MIR_LASSO_kFold_1,
                                  pred_MIR_LASSO_kFold_2 %>% select(18:42),
                                  pred_MIR_LASSO_kFold_3 %>% select(18:42))

write_xlsx(test_MIR_LASSO_kFold,
           "./04_modelsPredictions_vip/test_MIR_LASSO_kFold.xlsx")

# Predicting with VNIR LASSO LSPO models ------------------------------------

ti <- Sys.time()

list_VNIR_LASSO_LSPO <- list.files("./02_tuned_models/",
                                 pattern = "lasso_VNIR_LSPO_tuned_model")

## Part 1
pred_VNIR_LASSO_LSPO_1 <- list()

for (i in 1:50) {
  load(paste("./02_tuned_models/",list_VNIR_LASSO_LSPO[i], sep = ""))
  var_name <- strsplit(list_VNIR_LASSO_LSPO[i], '\\.')[[1]][1]
  
  pred_VNIR_LASSO_LSPO_1[[i]] <- tibble(SOC_model_ = predict(VNIR_LASSO_LSPO,
                                                           X_VNIR_LSPO_testing)) %>%
    rename_with(.fn = ~ glue("{var_name}"))
  # Cleaning up memory
  rm(VNIR_LASSO_LSPO, var_name)
}
pred_VNIR_LASSO_LSPO_1 <- bind_cols(VNIR_LSPO_testing[,1:17], pred_VNIR_LASSO_LSPO_1)
save(pred_VNIR_LASSO_LSPO_1,
     file = "./04_modelsPredictions_vip/pred_VNIR_LASSO_LSPO.RData")

## Part 2
pred_VNIR_LASSO_LSPO_2 <- list()

for (i in 51:75) {
  load(paste("./02_tuned_models/",list_VNIR_LASSO_LSPO[i], sep = ""))
  var_name <- strsplit(list_VNIR_LASSO_LSPO[i], '\\.')[[1]][1]
  
  pred_VNIR_LASSO_LSPO_2[[i]] <- tibble(
    SOC_model_ = predict(VNIR_LASSO_LSPO, X_VNIR_LSPO_testing)) %>%
    rename_with(.fn = ~ glue("{var_name}"))
  # Cleaning up memory
  rm(VNIR_LASSO_LSPO, var_name)
}
pred_VNIR_LASSO_LSPO_2  <- bind_cols(VNIR_LSPO_testing[,1:17], pred_VNIR_LASSO_LSPO_2)
save(pred_VNIR_LASSO_LSPO_2, 
     file = "./04_modelsPredictions_vip/pred_VNIR_LASSO_LSPO_2.RData")

## Part 3
pred_VNIR_LASSO_LSPO_3 <- list()
for (i in 76:100) {
  load(paste("./02_tuned_models/",list_VNIR_LASSO_LSPO[i], sep = ""))
  var_name <- strsplit(list_VNIR_LASSO_LSPO[i], '\\.')[[1]][1]
  
  pred_VNIR_LASSO_LSPO_3[[i]] <- tibble(
    SOC_model_ = predict(VNIR_LASSO_LSPO, X_VNIR_LSPO_testing)) %>%
    rename_with(.fn = ~ glue("{var_name}"))
  # Cleaning up memory
  rm(VNIR_LASSO_LSPO, var_name)
}

pred_VNIR_LASSO_LSPO_3  <- bind_cols(VNIR_LSPO_testing[,1:17], pred_VNIR_LASSO_LSPO_3)
save(pred_VNIR_LASSO_LSPO_3, 
     file = "./04_modelsPredictions_vip/pred_VNIR_LASSO_LSPO_3.RData")

rm(list_VNIR_LASSO_LSPO)

###########################################################################
tf  <- Sys.time()
tf - ti

test_VNIR_LASSO_LSPO <- bind_cols(pred_VNIR_LASSO_LSPO_1,
                                pred_VNIR_LASSO_LSPO_2 %>% select(18:42),
                                pred_VNIR_LASSO_LSPO_3 %>% select(18:42))

write_xlsx(test_VNIR_LASSO_LSPO,
           "./04_modelsPredictions_vip/test_VNIR_LASSO_LSPO.xlsx")

# Predicting with VNIR LASSO kFold models ------------------------------------

ti <- Sys.time()

list_VNIR_LASSO_kFold <- list.files("./02_tuned_models_kFold/",
                                   pattern = "lasso_VNIR_kFold_tuned_model")

## Part 1
pred_VNIR_LASSO_kFold_1 <- list()

for (i in 1:50) {
  load(paste("./02_tuned_models_kFold/",list_VNIR_LASSO_kFold[i], sep = ""))
  var_name <- strsplit(list_VNIR_LASSO_kFold[i], '\\.')[[1]][1]
  
  pred_VNIR_LASSO_kFold_1[[i]] <- tibble(SOC_model_ = predict(VNIR_LASSO_kFold,
                                                             X_VNIR_LSPO_testing)) %>%
    rename_with(.fn = ~ glue("{var_name}"))
  # Cleaning up memory
  rm(VNIR_LASSO_kFold, var_name)
}
pred_VNIR_LASSO_kFold_1 <- bind_cols(VNIR_LSPO_testing[,1:17], pred_VNIR_LASSO_kFold_1)
save(pred_VNIR_LASSO_kFold_1,
     file = "./04_modelsPredictions_vip/pred_VNIR_LASSO_kFold_1.RData")

## Part 2
pred_VNIR_LASSO_kFold_2 <- list()

for (i in 51:75) {
  load(paste("./02_tuned_models_kFold/",list_VNIR_LASSO_kFold[i], sep = ""))
  var_name <- strsplit(list_VNIR_LASSO_kFold[i], '\\.')[[1]][1]
  
  pred_VNIR_LASSO_kFold_2[[i]] <- tibble(
    SOC_model_ = predict(VNIR_LASSO_kFold, X_VNIR_LSPO_testing)) %>%
    rename_with(.fn = ~ glue("{var_name}"))
  # Cleaning up memory
  rm(VNIR_LASSO_kFold, var_name)
}
pred_VNIR_LASSO_kFold_2  <- bind_cols(VNIR_LSPO_testing[,1:17], pred_VNIR_LASSO_kFold_2)
save(pred_VNIR_LASSO_kFold_2, 
     file = "./04_modelsPredictions_vip/pred_VNIR_LASSO_kFold_2.RData")

## Part 3
pred_VNIR_LASSO_kFold_3 <- list()
for (i in 76:100) {
  load(paste("./02_tuned_models_kFold/",list_VNIR_LASSO_kFold[i], sep = ""))
  var_name <- strsplit(list_VNIR_LASSO_kFold[i], '\\.')[[1]][1]
  
  pred_VNIR_LASSO_kFold_3[[i]] <- tibble(
    SOC_model_ = predict(VNIR_LASSO_kFold, X_VNIR_LSPO_testing)) %>%
    rename_with(.fn = ~ glue("{var_name}"))
  # Cleaning up memory
  rm(VNIR_LASSO_kFold, var_name)
}

pred_VNIR_LASSO_kFold_3  <- bind_cols(VNIR_LSPO_testing[,1:17], pred_VNIR_LASSO_kFold_3)
save(pred_VNIR_LASSO_kFold_3, 
     file = "./04_modelsPredictions_vip/pred_VNIR_LASSO_kFold_3.RData")

rm(list_VNIR_LASSO_kFold)

###########################################################################
tf  <- Sys.time()
tf - ti

test_VNIR_LASSO_kFold <- bind_cols(pred_VNIR_LASSO_kFold_1,
                                  pred_VNIR_LASSO_kFold_2 %>% select(18:42),
                                  pred_VNIR_LASSO_kFold_3 %>% select(18:42))

write_xlsx(test_VNIR_LASSO_kFold,
           "./04_modelsPredictions_vip/test_VNIR_LASSO_kFold.xlsx")


# Getting variable importance data ----------------------------------------
# Getting vip from PLS-based models ---------------------------------------
## PLS - MIR - LSPO:
vip_MIR_PLS_LSPO <- list()
list_MIR_PLS_LSPO <- list.files("./02_tuned_models/",
                               pattern = "pls_MIR_LSPO_tuned_model")
for (i in 1:100) {
  
  load(paste("./02_tuned_models/",list_MIR_PLS_LSPO[i], sep = ""))
  
  vip_MIR_PLS_LSPO[[i]] <- vi_model(MIR_PLS_LSPO) %>%
    rename_with(.fn = function(x) paste(x,"MIR_PLS_LSPO_model_", i)
    )
  rm(MIR_PLS_LSPO)
}
vip_MIR_PLS_LSPO <- bind_cols(vip_MIR_PLS_LSPO)
save(vip_MIR_PLS_LSPO, file = "./04_modelsPredictions_vip/vip_MIR_PLS_LSPO.RData")
rm(vip_MIR_PLS_LSPO)

## PLS - VNIR - LSPO:
vip_VNIR_PLS_LSPO <- list()
list_VNIR_PLS_LSPO <- list.files("./02_tuned_models/",
                               pattern = "pls_VNIR_LSPO_tuned_model")
for (i in 1:100) {
  
  load(paste("./02_tuned_models/",list_VNIR_PLS_LSPO[i], sep = ""))
  
  vip_VNIR_PLS_LSPO[[i]] <- vi_model(VNIR_PLS_LSPO) %>%
    rename_with(.fn = function(x) paste(x,"VNIR_PLS_LSPO_model_", i)
    )
  rm(VNIR_PLS_LSPO)
}
vip_VNIR_PLS_LSPO <- bind_cols(vip_VNIR_PLS_LSPO)
save(vip_VNIR_PLS_LSPO, 
     file = "./04_modelsPredictions_vip/vip_VNIR_PLS_LSPO.RData")
rm(vip_VNIR_PLS_LSPO)

## PLS - MIR - kFold:
vip_MIR_PLS_kFold <- list()
list_MIR_PLS_kFold <- list.files("./02_tuned_models/",
                               pattern = "pls_MIR_kFold_tuned_model")
for (i in 1:100) {
  
  load(paste("./02_tuned_models/",list_MIR_PLS_kFold[i], sep = ""))
  
  vip_MIR_PLS_kFold[[i]] <- vi_model(MIR_PLS_kFold) %>%
    rename_with(.fn = function(x) paste(x,"MIR_PLS_kFold_model_", i)
    )
  rm(MIR_PLS_kFold)
}
vip_MIR_PLS_kFold <- bind_cols(vip_MIR_PLS_kFold)
save(vip_MIR_PLS_kFold, 
     file = "./04_modelsPredictions_vip/vip_MIR_PLS_kFold.RData")
rm(vip_MIR_PLS_kFold)

## PLS - VNIR - kFold:
vip_VNIR_PLS_kFold <- list()
list_VNIR_PLS_kFold <- list.files("./02_tuned_models/",
                                pattern = "pls_VNIR_kFold_tuned_model")
for (i in 1:100) {
  
  load(paste("./02_tuned_models/",list_VNIR_PLS_kFold[i], sep = ""))
  
  vip_VNIR_PLS_kFold[[i]] <- vi_model(VNIR_PLS_kFold) %>%
    rename_with(.fn = function(x) paste(x,"VNIR_PLS_kFold_model_", i)
    )
  rm(VNIR_PLS_kFold)
}
vip_VNIR_PLS_kFold <- bind_cols(vip_VNIR_PLS_kFold)
save(vip_VNIR_PLS_kFold, 
     file = "./04_modelsPredictions_vip/vip_VNIR_PLS_kFold.RData")
rm(vip_VNIR_PLS_kFold)

# Getting vip from LASSO-based models ---------------------------------------
## LASSO - MIR - LSPO:
vip_MIR_LASSO_LSPO <- list()
list_MIR_LASSO_LSPO <- list.files("./02_tuned_models/",
                               pattern = "lasso_MIR_LSPO_tuned_model")
for (i in 1:100) {
  
  load(paste("./02_tuned_models/",list_MIR_LASSO_LSPO[i], sep = ""))
  
  vip_MIR_LASSO_LSPO[[i]] <- vi_model(MIR_LASSO_LSPO) %>%
    rename_with(.fn = function(x) paste(x,"MIR_LASSO_LSPO_model_", i)
    )
  rm(MIR_LASSO_LSPO)
}
vip_MIR_LASSO_LSPO <- bind_cols(vip_MIR_LASSO_LSPO)
save(vip_MIR_LASSO_LSPO,
     file = "./04_modelsPredictions_vip/vip_MIR_LASSO_LSPO.RData")
rm(vip_MIR_LASSO_LSPO)

## LASSO - VNIR - LSPO:
vip_VNIR_LASSO_LSPO <- list()
list_VNIR_LASSO_LSPO <- list.files("./02_tuned_models/",
                                pattern = "lasso_VNIR_LSPO_tuned_model")
for (i in 1:100) {
  
  load(paste("./02_tuned_models/",list_VNIR_LASSO_LSPO[i], sep = ""))
  
  vip_VNIR_LASSO_LSPO[[i]] <- vi_model(VNIR_LASSO_LSPO) %>%
    rename_with(.fn = function(x) paste(x,"VNIR_LASSO_LSPO_model_", i)
    )
  rm(VNIR_LASSO_LSPO)
}
vip_VNIR_LASSO_LSPO <- bind_cols(vip_VNIR_LASSO_LSPO)
save(vip_VNIR_LASSO_LSPO, 
     file = "./04_modelsPredictions_vip/vip_VNIR_LASSO_LSPO.RData")
rm(vip_VNIR_LASSO_LSPO)

## LASSO - MIR - kFold:
vip_MIR_LASSO_kFold <- list()
list_MIR_LASSO_kFold <- list.files("./02_tuned_models/",
                                pattern = "lasso_MIR_kFold_tuned_model")
for (i in 1:100) {
  
  load(paste("./02_tuned_models/",list_MIR_LASSO_kFold[i], sep = ""))
  
  vip_MIR_LASSO_kFold[[i]] <- vi_model(MIR_LASSO_kFold) %>%
    rename_with(.fn = function(x) paste(x,"MIR_LASSO_kFold_model_", i)
    )
  rm(MIR_LASSO_kFold)
}
vip_MIR_LASSO_kFold <- bind_cols(vip_MIR_LASSO_kFold)
save(vip_MIR_LASSO_kFold, 
     file = "./04_modelsPredictions_vip/vip_MIR_LASSO_kFold.RData")
rm(vip_MIR_LASSO_kFold)

## LASSO - VNIR - kFold:
vip_VNIR_LASSO_kFold <- list()
list_VNIR_LASSO_kFold <- list.files("./02_tuned_models/",
                                 pattern = "lasso_VNIR_kFold_tuned_model")
for (i in 1:100) {
  
  load(paste("./02_tuned_models/",list_VNIR_LASSO_kFold[i], sep = ""))
  
  vip_VNIR_LASSO_kFold[[i]] <- vi_model(VNIR_LASSO_kFold) %>%
    rename_with(.fn = function(x) paste(x,"VNIR_LASSO_kFold_model_", i)
    )
  rm(VNIR_LASSO_kFold)
}
vip_VNIR_LASSO_kFold <- bind_cols(vip_VNIR_LASSO_kFold)
save(vip_VNIR_LASSO_kFold, 
     file = "./04_modelsPredictions_vip/vip_VNIR_LASSO_kFold.RData")
rm(vip_VNIR_LASSO_kFold)
