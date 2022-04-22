# -*- coding: utf-8 -*-
"""
Created on Fri Apr 22 15:25:12 2022

@author: Wenting Wang
"""

#--------------------------------------------------------------------------------------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------------------------------------------------------------------------------
# P2D post-processing, and P2P post-processing
# BON:  Latitude: 40.05192, Longitude: -88.37309, Time zone: UTC-6 (Etc/GMT+6)
# DRA:  Latitude: 36.62373, Longitude: -116.01947, Time zone: UTC-8 (Etc/GMT+8)
# FPK:  Latitude: 48.30783, Longitude: -105.10170, Time zone: UTC-7 (Etc/GMT+7)
# GWN:  Latitude: 34.2547, Longitude: -89.8729, Time zone: UTC-6 (Etc/GMT+6)
# PSU:  Latitude: 40.72012, Longitude: -77.93085, Time zone: UTC-6 (Etc/GMT+6)
# SXF:  Latitude: 43.73403, Longitude: -96.62328, Time zone: UTC-6 (Etc/GMT+6)
# TBL:  Latitude: 40.12498, Longitude: -105.23680, Time zone: UTC-7 (Etc/GMT+7) 
# Year: 2017,2018,2019,2020

import pandas as pd
import numpy as np
from scipy import special
import scipy.stats as stats
from scipy.optimize import Bounds
from scipy.optimize import minimize
from sklearn.metrics import mean_squared_error
#--------------------------------------------------------------------------------------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------------------------------------------------------------------------------
# load BON data
# Data can be replaced here with other sites (DRA, FPK, GWN, PSU, SXF, TBL)
data_target = pd.read_csv('C:/WWT/论文/ECMWF_ENS/supplementary material/post-processing/data/BON.csv',index_col=-2)
# find zenith > 85 degree
data_target.loc[data_target['Solar Zenith Angle'] > 85,'NSRDB_GHI'] = np.nan

#--------------------------------------------------------------------------------------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------------------------------------------------------------------------------
# member mean
data_target['member_mean'] = data_target.iloc[:,:50].mean(axis=1)
# member median
data_target['member_median'] = data_target.iloc[:,:50].median(axis=1)
# delete zenith > 85 degree
data_target = data_target.dropna()

#--------------------------------------------------------------------------------------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------------------------------------------------------------------------------
# define mean absolute error
def mae(y_true, y_pred):
    """
    Mean absolute error
    
    Parameters
    ----------
    y_true: array
        observed value
    y_pred: array
        forecasts 
    """  
    return np.mean(np.abs((y_pred - y_true)))
#--------------------------------------------------------------------------------------------------------------------------------------------------------------------
# P2D post_processing
# MAE
mae_median = mae(data_target.NSRDB_GHI, data_target.member_median)
mae_mean = mae(data_target.NSRDB_GHI, data_target.member_mean)
# RMSE
rmse_median = mean_squared_error(data_target.NSRDB_GHI, data_target.member_median, squared=False)
rmse_mean = mean_squared_error(data_target.NSRDB_GHI, data_target.member_mean, squared=False)

#--------------------------------------------------------------------------------------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------------------------------------------------------------------------------
# P2P post-processing
normconst = 1 / np.sqrt(2 * np.pi)
def normpdf(x):
    """Probability density function of a univariate standard Gaussian
    distribution with zero mean and unit variance
    """
    return normconst * 1 * np.exp(-(x*x)/2)
# Cumulative distribution function of a univariate standard Gaussian
# distribution with zero mean and unit variance
normcdf = special.ndtr
# weights
member_number = 50
weights = 1 / member_number
#---------------------------------------------------------------------------------------------------------------------------------------------------------------------
# three years (2017--2019) of data are used for training,
# and the remaining year (2020) is used for verification.

# train
data_train_2017 = data_target.filter(like='2017', axis=0)
data_train_2018 = data_target.filter(like='2018', axis=0)
data_train_2019 = data_target.filter(like='2019', axis=0)
data_train = pd.concat([data_train_2017,data_train_2018,data_train_2019])
# test
data_test = data_target.filter(like='2020', axis=0)

# The mean of the forecast normal distribution (2017--2019)
length_train = data_train.shape[0]
mu_train = np.zeros(length_train)
for t in range(length_train):
    for k in range(member_number):
        mu_train[t] = mu_train[t] + weights * data_train.iloc[t,k]

# The sample variance S2
s2_train = np.zeros(length_train)
for t in range(length_train):
    s2_train[t] = data_train.iloc[t,:50].var()

# satellite observations 2019
observations_train = np.array(data_train['NSRDB_GHI'])
# optimization objective: three parameter
def objective(x):
    XX = ( observations_train - x[0] - mu_train )/np.sqrt(x[1]*s2_train + x[2])
    CRPS = np.mean(  np.sqrt(x[1]*s2_train + x[2]) * ( XX*(2*normcdf(XX)-1) + 2*normpdf(XX) - 1/np.sqrt(np.pi)  )  )
    return CRPS

bounds = Bounds([0.001,0.001,0.001],[10,10,1000])
x0 = np.array([0.6,0.5,0.5])
res = minimize(objective, x0, method='SLSQP', bounds=bounds, options={'ftol': 1e-9, 'maxiter': 200,  'disp': True})
# results
results_train = res.x

#----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# the remaining year (2020) is used for verification
# The mean of the forecast normal distribution
length_test = data_test.shape[0]
mu_test = np.zeros(length_test)
for t in range(length_test):
    for k in range(member_number):
        mu_test[t] = mu_test[t] + weights * data_test.iloc[t,k]

# The sample variance S2
s2_test = np.zeros(length_test)
for t in range(length_test):
    s2_test[t] = data_test.iloc[t,:50].var()

# satellite observations 2020
observations_test = np.array(data_test['NSRDB_GHI'])

# continuous ranked probability score (CRPS)
# output CRPS without EMOS
XX_withoutEMOS_test = ( observations_test - mu_test )/np.sqrt(s2_test)
CRPS_withoutEMOS_test = np.mean(  np.sqrt(s2_test) * ( XX_withoutEMOS_test*(2*normcdf(XX_withoutEMOS_test)-1) + 2*normpdf(XX_withoutEMOS_test) - 1/np.sqrt(np.pi)  )  )

# output CRPS with EMOS
XX_withEMOS_test = ( observations_test - results_train[0] - mu_test )/np.sqrt(results_train[1]*s2_test + results_train[2])
CRPS_withEMOS_test = np.mean(  np.sqrt(results_train[1]*s2_test + results_train[2]) * ( XX_withEMOS_test*(2*normcdf(XX_withEMOS_test)-1) + 2*normpdf(XX_withEMOS_test) - 1/np.sqrt(np.pi)  )  )

#-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# (PIT)
var_emos = results_train[1]*s2_test + results_train[2]
st = np.sqrt(var_emos)
mu = results_train[0] + mu_test

predictive_distribution = stats.norm.cdf(data_test.NSRDB_GHI, loc=mu, scale=st)

pd_cdf = pd.DataFrame(predictive_distribution, columns = ['distribution'])
pd_cdf['STN'] = 'BON'
# pd_cdf.to_csv("PIT_BON.csv")








