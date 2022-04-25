# -*- coding: utf-8 -*-
"""
Created on Fri Apr  1 09:40:40 2022

@author: Wenting Wang
"""
#-----------------------------------------------------------------------------------------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------------------------------------------------------------------------------
# irradiance-to-power conversion using model chain
# source: https://www.aeprenewables.com/jacumba-solar/
# plant name: Jacumba Solar Farm
# EIA Plant ID: 60947
# Latitude: 32.6193, Longitude: -116.130
# Capacity : 20 MW
# Year: 2017,2018,2019,2020

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pvlib
from pvlib.pvsystem import PVSystem
from pvlib.location import Location
from pvlib.modelchain import ModelChain
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
# extract three variables, namely, "u10", "v10", "t2m"
weather_HRES = metadata_HRES[['u10','v10','t2m']]
# calculate wind speed by "u10" and "v10". (m/s)
# Eq.(1) in https://doi.org/10.1016/j.solener.2021.12.011
weather_HRES.insert(2,"wind_speed", np.sqrt((weather_HRES.u10)**2 + (weather_HRES.v10)**2))


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
# estimate DNI and DHI from EC_GHI
# Separation modeling aims at splitting the beam and diffuse radiation components from the global one.
# When estimating global titled irradiance (GTI), both GHI and DHI are required.
# Estimate DNI and DHI from GHI using the Erbs model.
# The Erbs model estimates the diffuse fraction from global horizontal irradiance 
# through an empirical relationship between DF and the ratio of GHI to extraterrestrial irradiance, Kt. 
irradiance = pvlib.irradiance.erbs(ghi=data_target.member_mean, zenith=data_target.zenith, datetime_or_doy=data_target.index)
#-----------------------------------------------------------------------------------------------------------------------------------------------------------------
# weather: input variable to model chain
# ghi: global horizontal irradiance; dhi: diffuse horizontal irradiacne; dni: direct normal irradiance; wind_speed: wind speed; temp_air: temperature
weather = pd.DataFrame(columns=['ghi','dhi','dni','wind_speed','temp_air'], index=data_target.index)
# ghi (W/m2)
weather['ghi'] = data_target['member_mean']
# dhi (W/m2)
weather['dhi'] = irradiance['dhi']
# dni (W/m2)
weather['dni'] = irradiance['dni']
# wind_speed (m/s)
weather['wind_speed'] = weather_HRES['wind_speed']
# air_temp (℃)
weather['temp_air'] = weather_HRES['t2m']

#-----------------------------------------------------------------------------------------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------------------------------------------------------------------------------
# download modeled generation estimates (regard as real PV AC power)
# source: https://data.openei.org/submissions/4503
real_PV = pd.read_csv('C:/WWT/论文/ECMWF_ENS/supplementary material/model chain/data/60947.csv', index_col=0)
real_PV.index = pd.date_range(start='2017-01-01 08:00:00', end='2021-01-01 07:00:00', freq='1h', tz='UTC')

#-----------------------------------------------------------------------------------------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------------------------------------------------------------------------------
# PV model
# load some module and inverter specifications
# Retrieve latest module and inverter info from a local file
# 'CECMod': the CEC module database
cec_modules = pvlib.pvsystem.retrieve_sam('CECMod')
cec_module = cec_modules['Jinko_Solar_Co___Ltd_JKM350M_72B']
# 'cecinverter': the CEC Inverter database
cec_inverters = pvlib.pvsystem.retrieve_sam('cecinverter')
# inverter parameters: https://files.sma.de/downloads/SC2200-3000-EV-DS-en-59.pdf
cec_inverter = cec_inverters['SMA_America__SC_2200_US__385V_']
cec_inverter['Vdcmax'] = 1100
cec_inverter['Idcmax'] = 3960
# set parameters
array_kwargs = dict(module_parameters=cec_module,
                    temperature_model_parameters=dict(a=-3.56, b=-0.075, deltaT=3))
# time zone
time_zone = 'UTC'
# Location objects are convenient containers for latitude, longitude, timezone, and altitude data associated with a particular geographic location.
location = Location(latitude=lat, longitude=lon,tz=time_zone)
# The angle is based on your latitude minus about 15 degrees.
mount = pvlib.pvsystem.FixedMount(surface_tilt=lat-14.58, surface_azimuth=180)
# 28x224 total modules arranged in 224 strings of 28 modules each 
arrays = [pvlib.pvsystem.Array(mount=mount,modules_per_string=28,strings=224,**array_kwargs)]
# The 'PVSystem' represents one inverter and the PV modules that supply DC power to the inverter.
system = PVSystem(arrays=arrays, inverter_parameters=cec_inverter)
# The ModelChain
mc = ModelChain(system, location, transposition_model='reindl',aoi_model='no_loss', spectral_model='no_loss')
# losses_model = 'pvwatts'
# Run the model chain
mc.run_model(weather)
#-----------------------------------------------------------------------------------------------------------------------------------------------------------------
# output AC power
results_ModelChain = mc.ac
# Estimate the power output of the entire photovoltaic power station
results = results_ModelChain*11.7
# define a new dataframe "Results" to store the real PV power and the estimating PV power.
Results = pd.DataFrame(columns=['PV_AC','SAM_gen'], index=weather.index)
Results['PV_AC'] = results
Results['SAM_gen'] = real_PV['SAM_gen']
# The AC power of the PV should be within the rated capacity
Results.loc[Results['PV_AC'] < 0, 'PV_AC'] = 0
Results.loc[Results['PV_AC'] > 20000000, 'PV_AC'] = 20000000
# unit MW
Results['PV_AC'] = Results['PV_AC']/1000000
# insert zenith angle into dataframe "Results"
Results.insert(2, "zenith", zenith_angle.zenith)
Results.insert(3, "utc_time", pd.date_range(start='2017-01-01 01:00:00', end='2020-12-31 23:00:00', freq='1h'))
# Delete "PV_AC" and "SAM_gen" with zenith angle greater than 85 degrees.
Results.loc[Results['zenith'] > 85,'PV_AC'] = np.nan
Results.loc[Results['zenith'] > 85,'SAM_gen'] = np.nan
Results = Results.dropna()
# extract 2020
Results_2020  = Results.filter(like='2020', axis=0)
# save
# Results_MC = Results_2020.to_csv("C:/Users/81095/PY/PV forecasting model chain/Results_MC.csv")


#-----------------------------------------------------------------------------------------------------------------------------------------------------------------
# RMSE nRMSE
from sklearn.metrics import mean_squared_error
# root mean square error 
rmse = mean_squared_error(Results_2020.SAM_gen, Results_2020.PV_AC, squared=False)
# normalized root mean square error
mean_measurements = Results_2020.SAM_gen.mean()
nRMSE = rmse/mean_measurements
