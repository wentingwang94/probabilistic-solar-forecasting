# -*- coding: utf-8 -*-
"""
Created on Sun Feb 20 18:29:28 2022

@author: 81095
"""

import pandas as pd
import numpy as np
import pvlib
from scipy import special
import scipy.stats as stats
from scipy.optimize import Bounds
from scipy.optimize import minimize
from sklearn.metrics import mean_squared_error
###############################################################################
# load ECMWF ENS data
# FPK:  Latitude: 48.30783, Longitude: -105.10170 
# Time zone: UTC-7 (Etc/GMT+7)
metadata1 = pd.read_csv('C:/Users/81095/PY/ENS/FPK_2017_2018.csv')
metadata2 = pd.read_csv('C:/Users/81095/PY/ENS/FPK_2019_2020.csv')
# Define a new dataframe: "data_FPK", which stores ensemble forecasts of 50 members.
data_FPK = pd.DataFrame(columns=[f'EC_GHI_{x+1}' for x in range(50)])

for i in range(50):
    ghi1 = metadata1.loc[(metadata1['number'] == i+1)]
    ghi2 = metadata2.loc[(metadata2['number'] == i+1)]
    ghi = pd.concat([ghi1,ghi2], axis=1)
    # extract 24h.
    ghi = ghi.iloc[1:25,:]
    # extract all columns "ghi".
    ghi_col = ghi.filter(regex='ghi')
    ghi_re = np.reshape(np.array(ghi_col), 24*1461, order='F')
    # extract all columns "time".
    time = ghi.filter(regex='time')
    time_re = np.reshape(np.array(time), 24*1461, order='F')
    # define a dictionary.
    d = {'ghi': ghi_re}
    # convert dictionary to dataframe.
    ghi_df = pd.DataFrame(data=d)
    ghi_df = ghi_df.iloc[:-1,:]
    # loop assignment.
    data_FPK[f'EC_GHI_{i+1}'] = ghi_df['ghi']

###############################################################################
# download zenith angle from NSRDB
Lat = 48.30783
Lon = -105.10170
# 2016
data_2016 = pvlib.iotools.get_psm3(
    latitude=Lat, 
    longitude=Lon, 
    api_key='wy050SejIsPycfUD4yROBmC77DOlHsO6t44Q0xCf', 
    email='wangwenting3000@gmail.com', names=2016, 
    interval=60, 
    attributes=('ghi', 'solar_zenith_angle'), 
    leap_day=True, 
    full_name='pvlib python', 
    affiliation='pvlib python', 
    timeout=60)
nsrdb2016 = data_2016[0][['Solar Zenith Angle','GHI']]
# 2017
data_2017 = pvlib.iotools.get_psm3(
    latitude=Lat, 
    longitude=Lon, 
    api_key='wy050SejIsPycfUD4yROBmC77DOlHsO6t44Q0xCf', 
    email='wangwenting3000@gmail.com', names=2017, 
    interval=60, 
    attributes=('ghi', 'solar_zenith_angle'), 
    leap_day=False, 
    full_name='pvlib python', 
    affiliation='pvlib python', 
    timeout=60)
nsrdb2017 = data_2017[0][['Solar Zenith Angle','GHI']]
# 2018
data_2018 = pvlib.iotools.get_psm3(
    latitude=Lat, 
    longitude=Lon, 
    api_key='wy050SejIsPycfUD4yROBmC77DOlHsO6t44Q0xCf', 
    email='wangwenting3000@gmail.com', names=2018, 
    interval=60, 
    attributes=('ghi', 'solar_zenith_angle'), 
    leap_day=False, 
    full_name='pvlib python', 
    affiliation='pvlib python', 
    timeout=60)
nsrdb2018 = data_2018[0][['Solar Zenith Angle','GHI']]
# 2019
data_2019 = pvlib.iotools.get_psm3(
    latitude=Lat, 
    longitude=Lon, 
    api_key='wy050SejIsPycfUD4yROBmC77DOlHsO6t44Q0xCf', 
    email='wangwenting3000@gmail.com', names=2019, 
    interval=60, 
    attributes=('ghi', 'solar_zenith_angle'), 
    leap_day=False, 
    full_name='pvlib python', 
    affiliation='pvlib python', 
    timeout=60)
nsrdb2019 = data_2019[0][['Solar Zenith Angle','GHI']]
# 2020
data_2020 = pvlib.iotools.get_psm3(
    latitude=Lat, 
    longitude=Lon, 
    api_key='wy050SejIsPycfUD4yROBmC77DOlHsO6t44Q0xCf', 
    email='wangwenting3000@gmail.com', names=2020, 
    interval=60, 
    attributes=('ghi', 'solar_zenith_angle'), 
    leap_day=True, 
    full_name='pvlib python', 
    affiliation='pvlib python', 
    timeout=60)
nsrdb2020 = data_2020[0][['Solar Zenith Angle','GHI']]
# 5 years (2016,2017,2018,2019,2020) of solar zenith angle from NSRDB
nsrdb = pd.concat([nsrdb2016,nsrdb2017,nsrdb2018,nsrdb2019,nsrdb2020])

################################################################################
# convert time zone
Times = pd.date_range(start = '2016-01-01 08:00:00',end='2021-01-01 07:00:00',freq = '1h')
nsrdb['UTC'] = Times
nsrdb['UTC-7'] = nsrdb.index
nsrdb.index = range(len(nsrdb))
# index number "8776" is UTC: 2017-01-01 01:00:00
# index number "43838" is UTC: 2020-12-31 23:00:00
nsrdb_target = nsrdb.iloc[8777:43840,:]
nsrdb_target = nsrdb_target.rename(columns={'GHI':'NSRDB_GHI'})
nsrdb_target.index = range(len(nsrdb_target))

################################################################################
# establish target data
data_target = pd.concat([data_FPK,nsrdb_target],axis=1) 

# find zenith > 85 degree
data_target.loc[data_target['Solar Zenith Angle'] > 85,'NSRDB_GHI'] = np.nan
data_target.loc[data_target['NSRDB_GHI'] == 0,'NSRDB_GHI'] = np.nan
###############################################################################
# member mean
data_target['member_mean'] = data_target.iloc[:,:50].mean(axis=1)
# member median
data_target['member_median'] = data_target.iloc[:,:50].median(axis=1)


# data_target['stn'] = 'FPK'
# data_target.to_csv('FPK.csv')

# delete zenith > 85 degree
data_target = data_target.dropna()
data_target.index = range(len(data_target))
###############################################################################
# MAPE
def mape(y_true, y_pred):
    """
    Mean absolute percentage error
    
    Parameters
    ----------
    y_true: array
        observed value
    y_pred: array
        forecasts 
    """    
    return np.mean(np.abs((y_pred - y_true) / y_true)) * 100

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

# MAE
mae_median = mae(data_target.NSRDB_GHI, data_target.member_median)
mae_mean = mae(data_target.NSRDB_GHI, data_target.member_mean)

# RMSE
rmse_median = mean_squared_error(data_target.NSRDB_GHI, data_target.member_median, squared=False)
rmse_mean = mean_squared_error(data_target.NSRDB_GHI, data_target.member_mean, squared=False)

###############################################################################
# CPRS
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


# three years (2017--2019) of data are used for training,
# and the remaining year (2020) is used for verification.

# train
data_train = data_target.iloc[:11910,:]
data_train.index = range(len(data_train))
# test
data_test = data_target.iloc[11910:,:]
data_test.index = range(len(data_test))


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

bounds = Bounds([0.001,0.001,0.001],[30,10,2000])
x0 = np.array([0.6,0.5,0.5])
res = minimize(objective, x0, method='SLSQP', bounds=bounds, options={'ftol': 1e-9, 'maxiter': 200,  'disp': True})

# results
results_train = res.x



# # optimization objective: two parameter
# def objective(x):
#     XX = ( observations_train - x[0] - mu_train )/np.sqrt(x[1]*s2_train)
#     CRPS = np.mean(  np.sqrt(x[1]*s2_train) * ( XX*(2*normcdf(XX)-1) + 2*normpdf(XX) - 1/np.sqrt(np.pi)  )  )
#     return CRPS

# bounds = Bounds([0.001,0.001],[20,20])
# x0 = np.array([0.6,0.5])
# res = minimize(objective, x0, method='SLSQP', bounds=bounds, options={'ftol': 1e-9, 'maxiter': 200,  'disp': True})

# # results
# results_train = res.x


# # optimization objective: one parameter
# def objective(x):
#     XX = ( observations_train - mu_train )/np.sqrt(x*s2_train)
#     CRPS = np.mean(  np.sqrt(x*s2_train) * ( XX*(2*normcdf(XX)-1) + 2*normpdf(XX) - 1/np.sqrt(np.pi)  )  )
#     return CRPS

# bounds = Bounds([0.001],[5])
# x0 = np.array([0.6])
# res = minimize(objective, x0, method='SLSQP', bounds=bounds, options={'ftol': 1e-9, 'maxiter': 200,  'disp': True})

# # results
# results_train = res.x

################################################################################
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



# output CRPS without EMOS
XX_withoutEMOS_test = ( observations_test - mu_test )/np.sqrt(s2_test)
CRPS_withoutEMOS_test = np.mean(  np.sqrt(s2_test) * ( XX_withoutEMOS_test*(2*normcdf(XX_withoutEMOS_test)-1) + 2*normpdf(XX_withoutEMOS_test) - 1/np.sqrt(np.pi)  )  )


# output CRPS with EMOS
XX_withEMOS_test = ( observations_test - results_train[0] - mu_test )/np.sqrt(results_train[1]*s2_test + results_train[2])
CRPS_withEMOS_test = np.mean(  np.sqrt(results_train[1]*s2_test + results_train[2]) * ( XX_withEMOS_test*(2*normcdf(XX_withEMOS_test)-1) + 2*normpdf(XX_withEMOS_test) - 1/np.sqrt(np.pi)  )  )


##############################################################################
# PIT

var_emos = results_train[1]*s2_test + results_train[2]
st = np.sqrt(var_emos)

mu = results_train[0] + mu_test

predictive_distribution = stats.norm.cdf(data_test.NSRDB_GHI, loc=mu, scale=st)

pd_cdf = pd.DataFrame(predictive_distribution, columns = ['distribution'])
pd_cdf['STN'] = 'FPK'
pd_cdf.to_csv("PIT_FPK.csv")




# from plotnine import * 
# import mizani
# fig, plot = ( ggplot(pd_cdf,aes('distribution', y=after_stat('ncount'))) +
#   geom_histogram(bins=51, fill='grey',colour='white',size=0.3) +
#   scale_y_continuous(trans=mizani.transforms.pseudo_log_trans(0.001,10), breaks=np.array([0,0.01,0.1])) +
#   facet_wrap('STN',ncol = 7) +
#   ylab("Relative frequency") + 
#   theme(axis_title = element_text(size = 11, family = "times new roman"),
#                     strip_text = element_text(size = 11, family = "times new roman"),
#                     axis_text = element_text(size = 11, family = "times new roman"),
#                     panel_spacing = 0.02,
#                     legend_title = element_text(size=11, family = "times new roman"), 
#                     strip_margin_y = 0,
#                     legend_position='none',
#                     legend_text = element_text(size=11, family = "times new roman"),
#                     figure_size=(3, 1.5),
#                     dpi=500
#                     )
#   ).draw(return_ggplot=True)




