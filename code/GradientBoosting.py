# -*- coding: utf-8 -*-
"""
Created on Sun Apr  3 09:25:53 2022

@author: Wenting Wang
"""
#-----------------------------------------------------------------------------------------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------------------------------------------------------------------------------
# goal: Regression
# reference paper: A semi-empirical approach using gradient boosting and k-nearest neighbors regression for GEFCom2014 probabilistic solar power forecasting
# method: gradient boosting
# De-trending:
# method: low-pass filter using Fourier transformation
# 1. The annual cycle of solar radiation and power is modelled using a low-pass filter built using a Fourier transformation.
# 2. The diurnal cycle is handled by fitting separate models for each hour of the day with the positive solar radiation.
import pandas as pd
import numpy as np
import pvlib
import seaborn as sns
from scipy import fftpack
from sklearn import ensemble
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
#-----------------------------------------------------------------------------------------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------------------------------------------------------------------------------
# download McClear 

# Time reference: Universal time (UTC)
# More information at: http://www.soda-pro.com/web-services/radiation/cams-mcclear
# Latitude: 32.6193, Longitude: -116.130 (Jacumba Solar Farm)
# Columns:
# 1. Observation period (ISO 8601)
# 2. TOA. Irradiation on horizontal plane at the top of atmosphere (Wh/m2)
# 3. Clear sky GHI. Clear sky global irradiation on horizontal plane at ground level (Wh/m2)
# 4. Clear sky BHI. Clear sky beam irradiation on horizontal plane at ground level (Wh/m2)
# 5. Clear sky DHI. Clear sky diffuse irradiation on horizontal plane at ground level (Wh/m2)
# 6. Clear sky BNI. Clear sky beam irradiation on mobile plane following the sun at normal incidence (Wh/m2)

# read McClear of Jacumba
metadata_McClear = pd.read_csv('C:/WWT/论文/ECMWF_ENS/supplementary material/model chain/data/McClear_Jacumba.csv',sep=';')
# extract two columns: "Observation period" and "Clear sky GHI"
# "Observation period": beginning/end of the time period with the format "yyyy-mm-ddTHH:MM:SS.S/yyyy-mm-ddTHH:MM:SS.S" (ISO 8601)
McClear = metadata_McClear[["Observation period","Clear sky GHI","Clear sky DHI","Clear sky BNI"]]
# set the beginning of the time period as the index.
begin_time_McClear = pd.date_range(start='2017-01-01 00:00:00', end='2020-12-31 23:45:00', freq='15min')
McClear.index = begin_time_McClear
# aggregate time series into 1 hour
# dataframe "McClear_agg_1h_raw" is to aggregate 00:00:00, 00:15:00, 00:30:00, 00:45:00. as 01:00:00
McClear_agg_1h_raw = McClear.resample("1h").sum()
# In the ECMWF dataset, the time is stamped at the end of the hour.
# Thus, the aggregate value of four period, namely, 00:00:00, 00:15:00, 00:30:00, and 00:45:00, is stamped at 01:00:00
McClear_agg_1h_raw.index = McClear_agg_1h_raw.index + pd.Timedelta("1h")
#-----------------------------------------------------------------------------------------------------------------------------------------------------------------
# 30min in advance
# we also want to aggregate "00:30:00, 00:45:00, 01:00:00, and 01:15:00" as the 01:00:00
McClear_advance_30min = McClear.copy() 
McClear_advance_30min.index = McClear_advance_30min.index - pd.Timedelta("30min")
McClear_agg_1h_advance_30min = McClear_advance_30min.resample("1h").sum()
# In the ECMWF dataset, the time is stamped at the end of the hour.
McClear_agg_1h_advance_30min.index = McClear_agg_1h_advance_30min.index + pd.Timedelta("1h")
#-----------------------------------------------------------------------------------------------------------------------------------------------------------------
# Truncated valid period: 2017-01-01 00:01:00 ~ 2020-12-31 23:00:00
McClear_agg_1h_raw = McClear_agg_1h_raw["2017-01-01 01:00:00" : "2020-12-31 23:00:00"]
McClear_agg_1h_advance_30min = McClear_agg_1h_advance_30min["2017-01-01 01:00:00" : "2020-12-31 23:00:00"]


#-----------------------------------------------------------------------------------------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------------------------------------------------------------------------------
# read ECMWF ensemble forecasts at Jacumba
metadata_ENS_Jacumba = pd.read_csv('C:/WWT/论文/ECMWF_ENS/supplementary material/model chain/data/Jacumba_ENS.csv', index_col=1)
metadata_ENS_Jacumba = metadata_ENS_Jacumba.iloc[:,1:]
# update index type
ENS_Jacumba_time = pd.date_range(start='2017-01-01 01:00:00', end='2020-12-31 23:00:00', freq='1h')
metadata_ENS_Jacumba.index = ENS_Jacumba_time
# ECMWF time stamp: e.g., "02:00:00" stands for period "01:00:00 ~ 02:00:00"
# calculate member mean
metadata_ENS_Jacumba['member_mean'] = metadata_ENS_Jacumba.iloc[:,:50].mean(axis=1)
ENS_mean_Jacumba = metadata_ENS_Jacumba[['member_mean']]
# Using McClear to advance the current ECMWF forecasts by half an hour. define a new dataframe "ENS_mean_Jacumba_p"
# index stands for ECMWF time stamp: e.g., 01:30:00 is period "00:30:00 ~ 01:30:00"
ENS_mean_Jacumba_p = pd.DataFrame(columns=['member_mean'], index=ENS_mean_Jacumba.index)
ENS_mean_Jacumba_p['member_mean'] = ENS_mean_Jacumba['member_mean'] / McClear_agg_1h_raw['Clear sky GHI'] * McClear_agg_1h_advance_30min['Clear sky GHI']
ENS_mean_Jacumba_p.index = ENS_mean_Jacumba_p.index - pd.Timedelta("30min")


#------------------------------------------------------------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------------------------------------------------------------
# download ECMWF HRES forecasts
# Actually, index in dataframe "metadata_HRES" is ECMWF time stamp. e.g., 2017-01-01 01:00:00 is period 2017-01-01 00:00:00 ~ 2017-01-01 01:00:00
# Note that we moved all the ensemble forecasts forward by half an hour. 
# Unfortunately, we have no way to align the time stamp of the ENS, which is pushed back half an hour, with the time stamp of the HRES.
metadata_HRES = pd.read_csv('C:/WWT/论文/ECMWF_ENS/supplementary material/model chain/data/ECMWF_HRES.csv', index_col=0)
# extract four variables, namely, "u10", "v10", "t2m", "d2m"
weather_HRES = metadata_HRES[['u10','v10','t2m','d2m']]
# relative humidity
# the calculation process is detailed in "https://doi.org/10.1016/j.solener.2021.12.011".
coff = 7.591386 * ( (weather_HRES.d2m/(weather_HRES.d2m+240.7263)) - (weather_HRES.t2m/(weather_HRES.t2m+240.7263)) )
weather_HRES.insert(2, "rh", 10**coff)
# extract dependent variables: rh, u10, v10, t2m
weather_HRES = weather_HRES[['u10','v10','t2m','rh']]


#-----------------------------------------------------------------------------------------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------------------------------------------------------------------------------
# zenith angle

# latitude, longitude
lat, lon = 32.6193, -116.13
# Times
# time of zenith 01:00:00 <---> time of ECMWF 01:30:00  
zenith_time = pd.date_range(start='2017-01-01 01:00:00', end='2020-12-31 23:00:00', freq='1h', tz='UTC')
# the position of the Sun
# spa_python: the solar positionig algorithm (SPA) is commonly regarded as the most accurate one to date.
position = pvlib.solarposition.spa_python(time=zenith_time, latitude=lat, longitude=lon)
# the position of the Sun is described by the solar azimuth and zenith angles.
zenith = position.zenith
zenith_angle = pd.DataFrame(columns=['ECMWF_time','zenith'],index=zenith.index)
zenith_angle['zenith'] = zenith
zenith_angle['ECMWF_time'] = zenith_angle.index - pd.Timedelta("30min")
#-----------------------------------------------------------------------------------------------------------------------------------------------------------------
# 
# define a temporary dataframe
temp = ENS_mean_Jacumba_p.copy()
temp.index = zenith_time
# join "zenith" and "member_mean" in same dataframe named "data_target"
# data_target: index--->NSRDB, e.g., (2017-01-01 01:00:00+00:00) is period (2017-01-01 00:30:00+00:00 ~ 2017-01-01 01:30:00+00:00)
# data_target: ECMWF_time---->ECMWF, e.g., (2017-01-01 01:30:00) is period (2017-01-01 00:30:00+00:00 ~ 2017-01-01 01:30:00+00:00)
data_target = zenith_angle.copy()
data_target['member_mean'] = temp['member_mean']
#-----------------------------------------------------------------------------------------------------------------------------------------------------------------
# check zenith versus clear sky solar irradiance
McClear_agg_1h_advance_30min.index = zenith_time 
McClear_agg_1h_advance_30min['zenith'] = zenith_angle['zenith']
# Any day is selected, and the user can change it at will
d = McClear_agg_1h_advance_30min.iloc[25:49,:]
# If on a clear day the rising path (morning) of the plot does not coincide with the falling path (afternoon) of the curve, 
# it is advisable to reexamine the assumptions on the time stamp.
plt.plot(d['zenith'], d['Clear sky GHI'])


#-----------------------------------------------------------------------------------------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------------------------------------------------------------------------------
# download modeled generation estimates (regard as real PV AC power)
# source: https://data.openei.org/submissions/4503
real_PV = pd.read_csv('C:/WWT/论文/ECMWF_ENS/supplementary material/model chain/data/60947.csv', index_col=0)
real_PV.index = pd.date_range(start='2017-01-01 08:00:00', end='2021-01-01 07:00:00', freq='1h', tz='UTC')

#-----------------------------------------------------------------------------------------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------------------------------------------------------------------------------
# establish feature set.
Feature = data_target.copy()
# relative humidity at 1000 mbar(%).
Feature.insert(1,"RH",weather_HRES.rh)
# air temperature 2m above ground level.
Feature.insert(3,"T2M",weather_HRES.t2m)
# 10 m U wind component (m/s)
Feature.insert(4,"U10",weather_HRES.u10)
# 10 m V wind component (m/s)
Feature.insert(5,"V10",weather_HRES.v10)
# PV power.
Feature.insert(6,"PV",real_PV.SAM_gen)
# Delete the data with zenith Angle greater than 85 degrees
Feature.iloc[Feature['zenith'] > 85, 7 ] = np.nan
Feature = Feature.dropna()
# Truncated valid period: 2017-07-30 00:00:00+00:00 ~ 2020-12-31 23:00:00+00:00
Feature = Feature["2017-07-30 00:00:00+00:00":"2020-12-31 23:00:00+00:00"]
#-----------------------------------------------------------------------------------------------------------------------------------------------------------------
# In this benchmark, we just focus on the mean_member regression.
# If readers are interested in other ensemble member, they could change the feature set.
Feature_target = Feature[['RH','T2M','U10','V10','PV','member_mean']]

#-----------------------------------------------------------------------------------------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------------------------------------------------------------------------------
# error metric: mean absolute error
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

#-----------------------------------------------------------------------------------------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------------------------------------------------------------------------------
# the principle of forecasting outlined by Armstrong, which states that when seasonal component is present in the time series, it needs to be removed before forecasting.
# each hour with positive solar radiation
feature_index = Feature_target.copy()
# The diurnal cycle is handled by fitting separate models for each hour of the day with the positive solar radiation.
feature1  = feature_index.filter(like='14:00:00', axis=0)
feature2  = feature_index.filter(like='15:00:00', axis=0)
feature3  = feature_index.filter(like='16:00:00', axis=0)
feature4  = feature_index.filter(like='17:00:00', axis=0)
feature5  = feature_index.filter(like='18:00:00', axis=0)
feature6  = feature_index.filter(like='19:00:00', axis=0)
feature7  = feature_index.filter(like='20:00:00', axis=0)
feature8  = feature_index.filter(like='21:00:00', axis=0)
feature9  = feature_index.filter(like='22:00:00', axis=0)
feature10 = feature_index.filter(like='23:00:00', axis=0)
feature11 = feature_index.filter(like='00:00:00', axis=0)
feature12 = feature_index.filter(like='01:00:00', axis=0)
feature13 = feature_index.filter(like='02:00:00', axis=0)

#-----------------------------------------------------------------------------------------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------------------------------------------------------------------------------
# A low-pass filter is built for modelling the annual cycle of the solar irradiance and power, using a Fourier transformation, due mainly to its smoothness.  
def low_pass_filter_fft(sig,n): 
    """
    Low-pass filter using Fourier transformation.
    
    Parameters
    ----------
    sig: array
        the FFT of the signal
    n: int
        Top n largest positive frequencies
    """
    # the FFT of the signal
    # scipy.fftpack.fft: return discrete Fourier transform of real or complex sequence
    sig_fft = fftpack.fft(sig) 
    # And the power  (sig_fft is of complex dtype)
    power = np.abs(sig_fft)**2
    # the corresponding frequencies
    # scipy.fftpack.fftfreq: return the discrete Fourier Transform sample frequencies
    # The returned float 
    sample_freq = fftpack.fftfreq(sig.size)
    # find the peak frequency: we can focus on only the positive frequencies
    pos_mask = np.where(sample_freq > 0)
    freqs = sample_freq[pos_mask]
    # extract more frequency
    pos_power = power[pos_mask]
    idx = pos_power.argsort()[-n:][::-1]
    peak_freq = freqs[idx]
    low_peak_freq = peak_freq[-1]  
    # argmax(): Returns the indices of the maximum values along an axis.
    # peak_freq = freqs[power[pos_mask].argmax()]  # frequency  !!!!!!!!!!!!!!!!!
    # remove all the high frequencies
    high_freq_fft = sig_fft.copy()
    high_freq_fft[np.abs(sample_freq) > low_peak_freq] = 0
    filtered_sig = fftpack.ifft(high_freq_fft)
    filtered_sig_power = np.abs(filtered_sig)
    return (filtered_sig_power)

#-----------------------------------------------------------------------------------------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------------------------------------------------------------------------------
# detrend annual cycle of PV power by FFT. 
filtered_PV_power1  = low_pass_filter_fft(feature1.PV.values,6)
filtered_PV_power2  = low_pass_filter_fft(feature2.PV.values,5)
filtered_PV_power3  = low_pass_filter_fft(feature3.PV.values,5)
filtered_PV_power4  = low_pass_filter_fft(feature4.PV.values,5)
filtered_PV_power5  = low_pass_filter_fft(feature5.PV.values,4)
filtered_PV_power6  = low_pass_filter_fft(feature6.PV.values,2)  
filtered_PV_power7  = low_pass_filter_fft(feature7.PV.values,4)  
filtered_PV_power8  = low_pass_filter_fft(feature8.PV.values,6)
filtered_PV_power9  = low_pass_filter_fft(feature9.PV.values,2)
filtered_PV_power10 = low_pass_filter_fft(feature10.PV.values,3)
filtered_PV_power11 = low_pass_filter_fft(feature11.PV.values,9)
filtered_PV_power12 = low_pass_filter_fft(feature12.PV.values,8)
filtered_PV_power13 = low_pass_filter_fft(feature13.PV.values,14)
#-----------------------------------------------------------------------------------------------------------------------------------------------------------------
# detrend annual cycle of solar irradiance by FFT
filtered_irradiance1  = low_pass_filter_fft(feature1.member_mean.values,1)
filtered_irradiance2  = low_pass_filter_fft(feature2.member_mean.values,2)
filtered_irradiance3  = low_pass_filter_fft(feature3.member_mean.values,2)
filtered_irradiance4  = low_pass_filter_fft(feature4.member_mean.values,2)
filtered_irradiance5  = low_pass_filter_fft(feature5.member_mean.values,2)
filtered_irradiance6  = low_pass_filter_fft(feature6.member_mean.values,2)  
filtered_irradiance7  = low_pass_filter_fft(feature7.member_mean.values,2)  
filtered_irradiance8  = low_pass_filter_fft(feature8.member_mean.values,2)
filtered_irradiance9  = low_pass_filter_fft(feature9.member_mean.values,2)
filtered_irradiance10 = low_pass_filter_fft(feature10.member_mean.values,2)
filtered_irradiance11 = low_pass_filter_fft(feature11.member_mean.values,2)
filtered_irradiance12 = low_pass_filter_fft(feature12.member_mean.values,2)
filtered_irradiance13 = low_pass_filter_fft(feature13.member_mean.values,2)

#-----------------------------------------------------------------------------------------------------------------------------------------------------------------
# insert "normpower" and "normirradiance"
# "normpower": normalized power, defined as the real power divided by the annual cycle model of power.
# "normirradiance": normalized solar irradiance, defined as the solar irradiance divided by its annual cycle model value.

# As the time resolution of prediction is only hourly, 
# it is feasible to develop an individual model for each hour of the day.
# In fact, there are only 13 models to be developed for different hours.

# MODEL 1: 14:00:00
feature1.insert(0,"FFT_SSRD",filtered_irradiance1)
feature1.insert(1,"FFT_PV",filtered_PV_power1)
feature1.insert(2,"normpower",feature1.PV/feature1.FFT_PV)
feature1.insert(3,"normirradiance",feature1.member_mean/feature1.FFT_SSRD)
# model 2: 15:00:00
feature2.insert(0,"FFT_SSRD",filtered_irradiance2)
feature2.insert(1,"FFT_PV",filtered_PV_power2)
feature2.insert(2,"normpower",feature2.PV/feature2.FFT_PV)
feature2.insert(3,"normirradiance",feature2.member_mean/feature2.FFT_SSRD)
# model 3: 16:00:00
feature3.insert(0,"FFT_SSRD",filtered_irradiance3)
feature3.insert(1,"FFT_PV",filtered_PV_power3)
feature3.insert(2,"normpower",feature3.PV/feature3.FFT_PV)
feature3.insert(3,"normirradiance",feature3.member_mean/feature3.FFT_SSRD)
# model 4: 17:00:00
feature4.insert(0,"FFT_SSRD",filtered_irradiance4)
feature4.insert(1,"FFT_PV",filtered_PV_power4)
feature4.insert(2,"normpower",feature4.PV/feature4.FFT_PV)
feature4.insert(3,"normirradiance",feature4.member_mean/feature4.FFT_SSRD)
# model 5: 18:00:00
feature5.insert(0,"FFT_SSRD",filtered_irradiance5)
feature5.insert(1,"FFT_PV",filtered_PV_power5)
feature5.insert(2,"normpower",feature5.PV/feature5.FFT_PV)
feature5.insert(3,"normirradiance",feature5.member_mean/feature5.FFT_SSRD)
# model 6: 19:00:00
feature6.insert(0,"FFT_SSRD",filtered_irradiance6)
feature6.insert(1,"FFT_PV",filtered_PV_power6)
feature6.insert(2,"normpower",feature6.PV/feature6.FFT_PV)
feature6.insert(3,"normirradiance",feature6.member_mean/feature6.FFT_SSRD)
# model 7: 20:00:00
feature7.insert(0,"FFT_SSRD",filtered_irradiance7)
feature7.insert(1,"FFT_PV",filtered_PV_power7)
feature7.insert(2,"normpower",feature7.PV/feature7.FFT_PV)
feature7.insert(3,"normirradiance",feature7.member_mean/feature7.FFT_SSRD)
# model 8: 21:00:00
feature8.insert(0,"FFT_SSRD",filtered_irradiance8)
feature8.insert(1,"FFT_PV",filtered_PV_power8)
feature8.insert(2,"normpower",feature8.PV/feature8.FFT_PV)
feature8.insert(3,"normirradiance",feature8.member_mean/feature8.FFT_SSRD)
# model 9: 22:00:00
feature9.insert(0,"FFT_SSRD",filtered_irradiance9)
feature9.insert(1,"FFT_PV",filtered_PV_power9)
feature9.insert(2,"normpower",feature9.PV/feature9.FFT_PV)
feature9.insert(3,"normirradiance",feature9.member_mean/feature9.FFT_SSRD)
# model 10: 23:00:00
feature10.insert(0,"FFT_SSRD",filtered_irradiance10)
feature10.insert(1,"FFT_PV",filtered_PV_power10)
feature10.insert(2,"normpower",feature10.PV/feature10.FFT_PV)
feature10.insert(3,"normirradiance",feature10.member_mean/feature10.FFT_SSRD)
# model 11: 00:00:00
feature11.insert(0,"FFT_SSRD",filtered_irradiance11)
feature11.insert(1,"FFT_PV",filtered_PV_power11)
feature11.insert(2,"normpower",feature11.PV/feature11.FFT_PV)
feature11.insert(3,"normirradiance",feature11.member_mean/feature11.FFT_SSRD)
# model 12: 01:00:00
feature12.insert(0,"FFT_SSRD",filtered_irradiance12)
feature12.insert(1,"FFT_PV",filtered_PV_power12)
feature12.insert(2,"normpower",feature12.PV/feature12.FFT_PV)
feature12.insert(3,"normirradiance",feature12.member_mean/feature12.FFT_SSRD)
# model 13: 02:00:00
feature13.insert(0,"FFT_SSRD",filtered_irradiance13)
feature13.insert(1,"FFT_PV",filtered_PV_power13)
feature13.insert(2,"normpower",feature13.PV/feature13.FFT_PV)
feature13.insert(3,"normirradiance",feature13.member_mean/feature13.FFT_SSRD)

#-----------------------------------------------------------------------------------------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------------------------------------------------------------------------------
# Forecasting MODEL
# Deterministic forecasting using gradient boosting
# Parameters of gradient boosting model
params = {
    "n_estimators": 500,
    "max_depth": 4,
    "min_samples_split": 5,
    "learning_rate": 0.01,
}
#-----------------------------------------------------------------------------------------------------------------------------------------------------------------
# define a function "predict_GBM"
def predict_GBM(feature_set): 
    """
    Deterministic forecasting using gradient boosting.
    Three years (2017,2018,2019) of data are used for training,
    and the remaining year (2020) is used for verification.
    
    Parameters
    ----------
    feature_set: dataframe
        the target feature set
    """
    # X_norm: the output of a solar energy system as the indenpent variable (or the predictand)
    # y_norm: irradiance and other weather variables as the independent variable (or the predictors)
    # "X_norm" and "y_norm": forecasting normalized PV power.
    X_norm = feature_set[['normirradiance','RH','T2M','U10','V10']]
    y_norm = feature_set[['normpower']]
    # forecasting normalized PV power.
    # train set, test set  X
    X_norm_train_2017 = X_norm.filter(like='2017', axis=0)
    X_norm_train_2018 = X_norm.filter(like='2018', axis=0)
    X_norm_train_2019 = X_norm.filter(like='2019', axis=0)
    X_norm_train = pd.concat([X_norm_train_2017,X_norm_train_2018,X_norm_train_2019], axis=0)
    X_norm_test = X_norm.filter(like='2020',axis=0)
    # train set, test set  y
    y_norm_train_2017 = y_norm.filter(like='2017',axis=0)
    y_norm_train_2018 = y_norm.filter(like='2018',axis=0)
    y_norm_train_2019 = y_norm.filter(like='2019',axis=0)
    y_norm_train = pd.concat([y_norm_train_2017,y_norm_train_2018,y_norm_train_2019], axis=0)
    y_norm_test = y_norm.filter(like='2020',axis=0)
    # Gradient boosting regressor model.
    reg_norm = ensemble.GradientBoostingRegressor(**params)
    reg_norm.fit(X_norm_train, y_norm_train)
    results_norm = reg_norm.predict(X_norm_test)    
    # define a new dataframe "Results"
    re_norm = {'predict_normpower':results_norm}
    Results = pd.DataFrame(data=re_norm, index=y_norm_test.index)
  
    # restore seasonality
    y_FFTPV = feature_set[['FFT_PV']].filter(like='2020',axis=0)
    Results['predict_power'] = Results['predict_normpower'] * y_FFTPV['FFT_PV']
    Results['real_PV'] = feature_set[['PV']].filter(like='2020',axis=0)
    return (Results)

#-----------------------------------------------------------------------------------------------------------------------------------------------------------------
# predict 13 models by function "predict_GBM".
Results1 = predict_GBM(feature1)
Results2 = predict_GBM(feature2)
Results3 = predict_GBM(feature3)
Results4 = predict_GBM(feature4)
Results5 = predict_GBM(feature5)
Results6 = predict_GBM(feature6)
Results7 = predict_GBM(feature7)
Results8 = predict_GBM(feature8)
Results9 = predict_GBM(feature9)
Results10 = predict_GBM(feature10)
Results11 = predict_GBM(feature11)
Results12 = predict_GBM(feature12)
Results13 = predict_GBM(feature13)

Results_GBM = pd.concat([Results1,Results2,Results3,Results4,Results5,Results6,Results7,Results8,Results9,Results10,Results11,Results12,Results13], axis=0)
# save result
Results_GBM_sort = Results_GBM.sort_index()
# Results_GBM_sort.to_csv("Results_GBM.csv") 

# RMSE MAE
mae_mean = mae(Results_GBM.real_PV, Results_GBM.predict_power)
rmse_mean = mean_squared_error(Results_GBM.real_PV, Results_GBM.predict_power, squared=False)

# scatter plot
sns.set_theme(style="darkgrid")
sns.scatterplot(x=Results_GBM.real_PV, y=Results_GBM.predict_power)




















