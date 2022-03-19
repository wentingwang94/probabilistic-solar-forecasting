# -*- coding: utf-8 -*-
"""
Created on Thu Jan 27 10:41:54 2022

@author: 81095
"""
###############################################################################
# goal: De-trending
# 1. The annual cycle of solar radiation and power is modelled using a low-pass filter built using a Fourier transformation.
# 2. The diurnal cycle is handled by fitting separate models for each hour of the day with the positive solar radiation.
###############################################################################
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np 
from sklearn.metrics import mean_squared_error
from sklearn import ensemble
from scipy import fftpack

###############################################################################
# load UPV
# source: https://data.openei.org/submissions/4503
real_PV = pd.read_csv('C:/Users/81095/PY/PV forecasting model chain/60947.csv')
# # extract 2018,2019,2020    unit: MW
# Real_PV = real_PV.iloc[8752:35056,:2]
# Real_PV.index = range(Real_PV.shape[0])
# extract 2017(start at 2017-07-30 00:00:00), 2018, 2019,2020    unit:MW
Real_PV = real_PV.iloc[5032:35056,:2]
Real_PV.index = range(Real_PV.shape[0])

################################################################################
# McClear 

# Time reference: Universal time (UT)
# More information at: http://www.soda-pro.com/web-services/radiation/cams-mcclear
# Latitude: 32.6193, Longitude: -116.130 (Jacumba Solar Farm)
# Columns:
# 1. Observation period (ISO 8601)
# 2. TOA. Irradiation on horizontal plane at the top of atmosphere (Wh/m2)
# 3. Clear sky GHI. Clear sky global irradiation on horizontal plane at ground level (Wh/m2)
# 4. Clear sky BHI. Clear sky beam irradiation on horizontal plane at ground level (Wh/m2)
# 5. Clear sky DHI. Clear sky diffuse irradiation on horizontal plane at ground level (Wh/m2)
# 6. Clear sky BNI. Clear sky beam irradiation on mobile plane following the sun at normal incidence (Wh/m2)
McClear_Jacumba = pd.read_csv('C:/Users/81095/PY/PV forecasting model chain/McClear_Jacumba_ens_pd.csv',sep=';')
Clearsky_GHI = McClear_Jacumba[["Observation period","Clear sky GHI"]]

Time = pd.date_range(start = '2017-01-01 01:00:00',end='2020-12-31 23:00:00',freq = '1h')
# Define a new dataframe: "Clearsky_GHI_1h", and column "Clearsky_GHI" is unprocessed McClear, column "Clearsky_GHI_p" is the McClear of a half-hour delay.
Clearsky_GHI_1h = pd.DataFrame(columns=['Time_UTC','Clearsky_GHI','Clearsky_GHI_p'])
Clearsky_GHI_1h['Time_UTC'] = Time
# In order to facilitate subsequent loop assignments, the dataframe  "Clearsky_GHI" is clipped.
Clearsky_GHI_cut = Clearsky_GHI[1:-3]
Clearsky_GHI_cut_p = Clearsky_GHI[3:-1]
# Redefine indexes
Clearsky_GHI_cut.index = range(140252)
Clearsky_GHI_cut_p.index = range(140252)

# Convert a time resolution of 15 minutes to 1 hour. The final result is stored in dataframe "Clearsky_GHI_1h".
for i in range(35063):
    a = Clearsky_GHI_cut.iloc[i*4:(i+1)*4,:]
    b = a[["Clear sky GHI"]]
    c = b.mean()
    d = c[0]
    Clearsky_GHI_1h.iloc[i,1] = d*4
    
for i in range(35063):
    a = Clearsky_GHI_cut_p.iloc[i*4:(i+1)*4,:]
    b = a[["Clear sky GHI"]]
    c = b.mean()
    d = c[0]
    Clearsky_GHI_1h.iloc[i,2] = d*4
    
    
###############################################################################
# load ECMWF ENS data at Jacumba
metadata1 = pd.read_csv('C:/Users/81095/PY/PV forecasting model chain/NCdata/Jacumba_2017_2018.csv')
metadata2 = pd.read_csv('C:/Users/81095/PY/PV forecasting model chain/NCdata/Jacumba_2019_2020.csv')
# Define a new dataframe: "data_Jacumba", which stores ensemble forecasts of 50 members.
data_Jacumba = pd.DataFrame(columns=[f'EC_GHI_{x+1}' for x in range(50)])
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
    data_Jacumba[f'EC_GHI_{i+1}'] = ghi_df['ghi']

# insert index of UTC time into dataframe "data_Jacumba".
t = {'time':time_re}
time_df = pd.DataFrame(data=t)
data_Jacumba.insert(0,"Time", time_df.iloc[:-1,:])

# define a new dataframe "exmwf", and store the clipped dataframe "data_Jacumba" in it.
# 2017-07-30 00:00:00 -- 2020-12-31 23:00:00.
ecmwf = data_Jacumba.copy()
ecmwf = ecmwf.iloc[5039:,:]
ecmwf.index = range(ecmwf.shape[0])


###################################################################################
# insert "Clear sky index" in ecmwf (Jacumba)
# extract 2017(start at 2017-07-30 00:00:00), 2018, 2019,2020
McClear_extract = Clearsky_GHI_1h.iloc[5039:,:]
McClear_extract.index = range(McClear_extract.shape[0])

# "p" stands for "processed".
# insert columns "Clearsky_GHI" and "Clearsky_GHI_p" in dataframe "ecmwf".
ecmwf['Clearsky_GHI'] = McClear_extract['Clearsky_GHI']
ecmwf['Clearsky_GHI_p'] = McClear_extract['Clearsky_GHI_p']

# The sampling time of EC is consistent with that of PV
# Define a new Dataframe (ecmwf_p) to store the processed GHI
ecmwf_p = pd.DataFrame(columns=[f'EC_GHI_p{x+1}' for x in range(50)])
ecmwf_p.insert(0,"Time",ecmwf.Time)
for x in range(50):
    ecmwf_p[f'EC_GHI_p{x+1}'] = ecmwf[f'EC_GHI_{x+1}'] / ecmwf['Clearsky_GHI'] * ecmwf['Clearsky_GHI_p']

###############################################################################
# Download zenith
jacumba_NSRDB = pd.read_csv('C:/Users/81095/PY/PV forecasting model chain/Jacumba_NSRDB.csv')
# time zone, Etc-7 to UTC
Time = pd.date_range(start = '2017-01-01 09:00:00',end='2021-01-01 08:00:00',freq = '1h')
jacumba_NSRDB.insert(1,"UTC",Time)
# extract 2017(start at 2017-07-30 00:00:00),2018,2019,2020
Jacumba_NSRDB = jacumba_NSRDB.iloc[5031:35055,:]
# extract column "Solar Zenith Angle"
zenith_NSRDB = Jacumba_NSRDB.iloc[:,[1,12]]
zenith_NSRDB.index = range(zenith_NSRDB.shape[0])


###############################################################################
# load ECMWF HRES
ecmwf_hres = pd.read_csv('C:/Users/81095/PY/PV forecasting model chain/ECMWF_HRES.csv')
# extract 2017(start at 2017-07-30 00:00:00),2018,2019,2020
ec_hres = ecmwf_hres.iloc[5039:35063,:]

# relative humidity
# the calculation process is detailed in "https://doi.org/10.1016/j.solener.2021.12.011".
coff = 7.591386 * ( (ec_hres.d2m/(ec_hres.d2m+240.7263)) - (ec_hres.t2m/(ec_hres.t2m+240.7263)) )
ec_hres['rh'] = 10**coff

# extract dependent variables: rh, u10, v10, t2m
hres = ec_hres.iloc[:,[0,4,5,6,15]]
hres.index = range(hres.shape[0])

###############################################################################
# establish feature set.
Feature = ecmwf_p.copy()
# insert member mean
Feature['member_mean'] = Feature.iloc[:,1:51].mean(axis=1)
# insert member median
Feature['member_median'] = Feature.iloc[:,1:51].median(axis=1)    
# relative humidity at 1000 mbar(%).
Feature.insert(1,"RH",hres.rh)
# solar zenith angle (not feature).
Feature.insert(2,"zenith",zenith_NSRDB['Solar Zenith Angle'])
# air temperature 2m above ground level.
Feature.insert(3,"T2M",hres.t2m)
# 10 m U wind component (m/s)
Feature.insert(4,"U10",hres.u10)
# 10 m V wind component (m/s)
Feature.insert(5,"V10",hres.v10)
# PV power.
Feature.insert(6,"PV",Real_PV.SAM_gen)
# Delete the data with zenith Angle greater than 85 degrees
Feature.iloc[Feature['zenith'] > 85, 7: ] = np.nan
Feature = Feature.dropna()
Feature.index = range(Feature.shape[0])


###############################################################################
# In this benchmark, we just focus on the mean_member regression.
# If readers are interested in other ensemble member, they could change the feature set.
Feature_target = Feature[['Time','RH','T2M','U10','V10','PV','member_mean']]


###############################################################################
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

###############################################################################
# the principle of forecasting outlined by Armstrong, which states that 
# when seasonal component is present in the time series, it needs to be removed before forecasting.

# each hour with positive solar radiation
feature_index = Feature_target.copy()
# define time index
feature_index.index = Feature_target.Time
feature_index = feature_index.drop('Time',axis=1)

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


#####################################################################################
# A low-pass filter is built for modelling the annual cycle of the solar irradiance and power,
# using a Fourier transformation, due mainly to its smoothness.  
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
    # plt.figure(figsize=(6,5))
    # plt.plot(sample_freq, power)
    # plt.xlabel('Frequency [Hz]')
    # plt.ylabel('plower')
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


###############################################################################
# detrend PV by FFT
# feature1,feature2, feature3,...,feature13, put them in feature_list[], and call this dataframe in batches.
# feature_list= []
# for i in range(1,14): #~ look up list comprehensions for a more elegant way to do this.
#     feature_list.append('feature'+str(i))

# sig_PV_list = []
# for i in range(1,14):
#     sig_PV_list.append('sig_PV'+str(i))


# for feature_temporary in feature_list:
#     for sig_PV_temporary in sig_PV:
#         sig_PV = eval(feature_temporary)['PV'].values


###############################################################################
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

# # plot show
# plt.figure(figsize=(6, 5))
# plt.plot(feature1.member_mean.values, label='Original signal')
# plt.plot(filtered_irradiance1, linewidth=3, label='Filtered signal')
# plt.xlabel('Time [s]')
# plt.ylabel('Amplitude')

###############################################################################
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


###############################################################################
# Forecasting MODEL
# Deterministic forecasting using gradient boosting
# Parameters of gradient boosting model
params = {
    "n_estimators": 500,
    "max_depth": 4,
    "min_samples_split": 5,
    "learning_rate": 0.01,
}

# define a function "predict_GBM"
# def predict_GBM(feature_set): 
#     """
#     Deterministic forecasting using gradient boosting.
#     Three years (2017,2018,2019) of data are used for training,
#     and the remaining year (2020) is used for verification.
    
#     Parameters
#     ----------
#     feature_set: dataframe
#         the target feature set
#     """
#     # X_norm: the output of a solar energy system as the indenpent variable (or the predictand)
#     # y_norm: irradiance and other weather variables as the independent variable (or the predictors)
#     # "X_norm" and "y_norm": forecasting normalized PV power.
#     X_norm = feature_set[['normirradiance','RH','T2M','U10','V10']]
#     y_norm = feature_set[['normpower']]
#     #############################################################
#     # forecasting normalized PV power.
#     # train set, test set  X
#     X_norm_train_2017 = X_norm.filter(like='2017', axis=0)
#     X_norm_train_2018 = X_norm.filter(like='2018', axis=0)
#     X_norm_train_2019 = X_norm.filter(like='2019', axis=0)
#     X_norm_train = pd.concat([X_norm_train_2017,X_norm_train_2018,X_norm_train_2019], axis=0)
#     X_norm_test = X_norm.filter(like='2020',axis=0)
#     # train set, test set  y
#     y_norm_train_2017 = y_norm.filter(like='2017',axis=0)
#     y_norm_train_2018 = y_norm.filter(like='2018',axis=0)
#     y_norm_train_2019 = y_norm.filter(like='2019',axis=0)
#     y_norm_train = pd.concat([y_norm_train_2017,y_norm_train_2018,y_norm_train_2019], axis=0)
#     y_norm_test = y_norm.filter(like='2020',axis=0)
#     # Gradient boosting regressor model.
#     reg_norm = ensemble.GradientBoostingRegressor(**params)
#     reg_norm.fit(X_norm_train, y_norm_train)
#     results_norm = reg_norm.predict(X_norm_test)    
#     # define a new dataframe "Results"
#     re_norm = {'predict_normpower':results_norm}
#     Results = pd.DataFrame(data=re_norm, index=y_norm_test.index)
    
#     ##############################################################
#     # forecasting annual cycle
#     # X_FFT: the output of a solar energy system as the indenpent variable (or the predictand)
#     # y_FFT: irradiance and other weather variables as the independent variable (or the predictors)
#     # "X_FFT" and "y_FFT": forecasting annual cycle model of PV power.
#     X_FFT = feature_set[['FFT_SSRD','RH','T2M','U10','V10']]
#     y_FFT = feature_set[['FFT_PV']]
#     # train set, test set  X
#     X_FFT_train_2017 = X_FFT.filter(like='2017', axis=0)
#     X_FFT_train_2018 = X_FFT.filter(like='2018', axis=0)
#     X_FFT_train_2019 = X_FFT.filter(like='2019', axis=0)
#     X_FFT_train = pd.concat([X_FFT_train_2017,X_FFT_train_2018,X_FFT_train_2019], axis=0)
#     X_FFT_test = X_FFT.filter(like='2020',axis=0)
#     # train set, test set  y
#     y_FFT_train_2017 = y_FFT.filter(like='2017',axis=0)
#     y_FFT_train_2018 = y_FFT.filter(like='2018',axis=0)
#     y_FFT_train_2019 = y_FFT.filter(like='2019',axis=0)
#     y_FFT_train = pd.concat([y_FFT_train_2017,y_FFT_train_2018,y_FFT_train_2019], axis=0)
#     y_FFT_test = y_FFT.filter(like='2020',axis=0)
#     # Gradient boosting regressor model.
#     reg_FFT = ensemble.GradientBoostingRegressor(**params)
#     reg_FFT.fit(X_FFT_train, y_FFT_train)
#     results_FFT = reg_FFT.predict(X_FFT_test)        
#     # define a new dataframe "Results_FFT"
#     re_FFT = {'predict_FFTpower':results_FFT}
#     Results_FFT = pd.DataFrame(data=re_FFT, index=y_FFT_test.index)  
    
#     # restore seasonality
#     Results['predict_power'] = Results['predict_normpower'] * Results_FFT['predict_FFTpower']
#     Results['real_PV'] = feature_set[['PV']].filter(like='2020',axis=0)
#     return (Results)


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
    #############################################################
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


################################################################################
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

# # PLOT SHOW
# plt.figure(figsize=(6, 5))
# plt.plot(Results13.predict_power, label='pre')
# plt.plot(Results13.real_PV, linewidth=3, label='real')


Results_GBM = pd.concat([Results1,Results2,Results3,Results4,Results5,Results6,Results7,Results8,Results9,Results10,Results11,Results12,Results13], axis=0)

# save result
Results_GBM_sort = Results_GBM.sort_index()
Results_GBM_sort.to_csv("Results_GBM.csv") 

# RMSE MAE
mae_mean = mae(Results_GBM.real_PV, Results_GBM.predict_power)
rmse_mean = mean_squared_error(Results_GBM.real_PV, Results_GBM.predict_power, squared=False)

# scatter plot
sns.set_theme(style="darkgrid")
sns.scatterplot(x=Results_GBM.real_PV, y=Results_GBM.predict_power)



