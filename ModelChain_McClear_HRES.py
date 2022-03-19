# -*- coding: utf-8 -*-
"""
Created on Fri Mar 18 13:52:34 2022

@author: 81095
"""

##############################################################################
# irradiance-to-power conversion
# source: https://www.aeprenewables.com/jacumba-solar/
# plant name: Jacumba Solar Farm
# EIA Plant ID: 60947
# Latitude: 32.6193, Longitude: -116.130
# Capacity : 20 MW
# Year: 2017,2018,2019,2020

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pvlib
from pvlib.pvsystem import PVSystem
from pvlib.location import Location
from pvlib.modelchain import ModelChain
from sklearn.metrics import mean_squared_error
################################################################################
# download McClear 

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
# download ECMWF HRES data 
ecmwf_hres = pd.read_csv('C:/Users/81095/PY/PV forecasting model chain/ECMWF_HRES.csv')
# extract 2020 (UTC and Etc/GMT+8 are 8 h apart)
# index number 26288 in dataframe "ecmwf_hres" is 2020/1/1 9:00
# index number 35062 in dataframe "ecmwf_hres" is 2020/12/31 23:00 
# column numbers "0", "4", "5", "6" stand for "time", "u10", "v10", and "t2m".
ecmwf = ecmwf_hres.iloc[26288:35063,[0,4,5,6]]
ecmwf.index = range(ecmwf.shape[0])
# inputs of model chain are wind speed, temperature, GHI, BNI, and DHI.
# calculate wind speed by "u10" and "v10"  (m/s)
ecmwf.insert(1,"wind_speed", np.sqrt((ecmwf.u10)**2 + (ecmwf.v10)**2))
# air temperature is "t2m"
ecmwf.insert(2,"temp_air", ecmwf.t2m)
# only concern on inputs of model chain, that is "wind speed" and "air temperature"
ecmwf = ecmwf.iloc[:,0:3]
# set index according to Etc/GMT+8 time
Times = Time = pd.date_range(start = '2020-01-01 00:30:00',end='2020-12-31 14:30:00',freq = '1h', tz='Etc/GMT+8')
ecmwf.index = Times


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

# define a new dataframe "Jacumba2020", and store the clipped dataframe "data_Jacumba" in it.
Jacumba2020 = data_Jacumba.copy()
# 2020-01-01 09:00:00 -- 2020-12-31 23:00:00.
# index number 26288 in dataframe "Jacumba2020" is 2020-01-01 09:00:00
Jacumba2020 = Jacumba2020.iloc[26288:,:]
# set index according to Etc/GMT+8 time
Jacumba2020.index = ecmwf.index

# insert ensemble mean and ensmeble median
Jacumba2020['member_mean'] = Jacumba2020.iloc[:,1:51].mean(axis=1)
Jacumba2020['member_median'] = Jacumba2020.iloc[:,1:51].median(axis=1)


###############################################################################
# In this benchmark, we just focus on the "member_mean".
# If readers are interested in other ensemble member, they could change it.
Jacumab_target = Jacumba2020[['member_mean']]
Jacumba = pd.concat([ecmwf,Jacumab_target], axis=1)
# rename
Jacumba = Jacumba.rename(columns={'time':'UTC','member_mean':'EC_ENS_ghi'})

###################################################################################
# insert "Clear sky index" in Jacumba
# index number 26288 in dataframe "Clearsky_GHI_1h" is 2020-01-01 09:00:00
aa = Clearsky_GHI_1h.iloc[26288:,:]
aa.index = Jacumba.index
Jacumba['Clearsky_GHI'] = aa['Clearsky_GHI']
Jacumba['Clearsky_GHI_p'] = aa['Clearsky_GHI_p']

# "p" stands for "processed"
# The sampling time of EC is consistent with that of PV
Jacumba['EC_ENS_ghi_p'] = Jacumba['EC_ENS_ghi'] / Jacumba['Clearsky_GHI'] * Jacumba['Clearsky_GHI_p']
Jacumba['UTC'] = Jacumba2020['Time']

################################################################################
# load PV power
# source: https://data.openei.org/submissions/4503
real_PV = pd.read_csv('C:/Users/81095/PY/PV forecasting model chain/60947.csv')
# extract 2020
# index number 26281 in dataframe "real_PV" is 2020-01-01 09:00:00
# index number 35055 in dataframe "real_PV" is 2020-12-31 23:00:00
Real_PV = real_PV.iloc[26281:35056,[0,1]]
Real_PV.index = Jacumba.index

###############################################################################
# insert "SAM_gen" in dataframe "Jacumba"
Jacumba['SAM_gen'] = Real_PV['SAM_gen']
# After the adjustment of the clear-sky processing, the data would appear infinitely large or infinitely small.
Jacumba = Jacumba.replace([np.inf, -np.inf], np.nan)
Jacumba = Jacumba.fillna(0)


###############################################################################
# estimate DNI and DHI from EC_GHI
# Separation modeling aims at splitting the beam and diffuse radiation components from the global one.
# When estimating global titled irradiance (GTI), both GHI and DHI are required.
lat, lon = 32.6193, -116.13
Times = pd.date_range(start = '2020-01-01 00:30:00',end='2020-12-31 14:30:00',freq = '1h', tz='Etc/GMT+8')
# the position of the sun.
# spa_python: the solar positioning algorithm (SPA) is commonly regarded as the most accurate one to date.
position = pvlib.solarposition.spa_python(time=Times, latitude=lat, longitude=lon)
# The position of the sun is described by the solar azimuth and zenith angles.
zenith = position.zenith
# Estimate DNI and DHI from GHI using the Erbs model.
# The Erbs model estimates the diffuse fraction DF from global horizontal irradiance through an empirical relationship between DF and the ratio of GHI to extraterrestrial irradiance, Kt. 
irradiance = pvlib.irradiance.erbs(ghi=Jacumba.EC_ENS_ghi_p, zenith=zenith, datetime_or_doy=Times)
Jacumba = pd.concat([Jacumba,irradiance], axis=1)
# rename
Jacumba = Jacumba.rename(columns={'dhi':'dhi_erbs', 'dni':'dni_erbs'})

# weather: input variable to model chain
weather = Jacumba[['EC_ENS_ghi_p','wind_speed','temp_air','dni_erbs','dhi_erbs']]
weather = weather.rename(columns={'dhi_erbs':'dhi','dni_erbs':'dni','EC_ENS_ghi_p':'ghi'})


################################################################################
# PV model
# load some module and inverter specifications
# Retrieve latest module and inverter info from a local file
# 'CECMod': the CEC module database
cec_modules = pvlib.pvsystem.retrieve_sam('CECMod')
# 'cecinverter': the CEC Inverter database
cec_inverters = pvlib.pvsystem.retrieve_sam('cecinverter')
# inverter parameters: https://files.sma.de/downloads/SC2200-3000-EV-DS-en-59.pdf
cec_module = cec_modules['Jinko_Solar_Co___Ltd_JKM350M_72B']
cec_inverter = cec_inverters['SMA_America__SC_2200_US__385V_']
cec_inverter['Vdcmax'] = 1100
cec_inverter['Idcmax'] = 3960
# set parameters
array_kwargs = dict(module_parameters=cec_module,
                    temperature_model_parameters=dict(a=-3.56, b=-0.075, deltaT=3))
# time zone
time_zone = 'Etc/GMT+8'
# Location objects are convenient containers for latitude, longitude, timezone, and altitude data associated with a particular geographic location.
location = Location(latitude=lat, longitude=lon,tz=time_zone)
# The angle is based on your latitude minus about 15 degrees.
mount = pvlib.pvsystem.FixedMount(surface_tilt=lat-14.58, surface_azimuth=180)
# 28x224 total modules arranged in 224 strings of 28 modules each 
arrays = [pvlib.pvsystem.Array(mount=mount,modules_per_string=28,strings=224,**array_kwargs)]
# The 'PVSystem' represents one inverter and the PV modules that supply DC power to the inverter.
system = PVSystem(arrays=arrays, inverter_parameters=cec_inverter)
# The ModelChain
mc = ModelChain(system, location, transposition_model='perez',
                aoi_model='no_loss', spectral_model='no_loss')
# Run the model chain
mc.run_model(weather)
# output AC power
Jacumba_PV = mc.ac
# Estimate the power output of the entire photovoltaic power station
total_Jacumba_PV = Jacumba_PV*11.7
total_Jacumba_PV = pd.DataFrame(data={'PV_AC':total_Jacumba_PV})

total_Jacumba_PV.loc[total_Jacumba_PV['PV_AC'] < 0,'PV_AC'] = 0
total_Jacumba_PV.loc[total_Jacumba_PV['PV_AC'] > 20000000,'PV_AC'] = 20000000

comparsionPV = pd.concat([Real_PV,total_Jacumba_PV],axis=1)
comparsionPV['utc_time'] = Jacumba2020['Time']
# unit MW
comparsionPV['PV_AC'] = comparsionPV['PV_AC']/1000000

##############################################################################
# RMSE
comparsionPV.insert(2, "zenith", position.zenith)
comparsionPV.loc[comparsionPV['zenith'] > 85,'PV_AC'] = np.nan
comparsionPV.loc[comparsionPV['zenith'] > 85,'SAM_gen'] = np.nan
comparsionPV_withoutnan = comparsionPV.dropna()

rmse = mean_squared_error(comparsionPV_withoutnan.SAM_gen, comparsionPV_withoutnan.PV_AC, squared=False)

# save results
Results_MC = comparsionPV_withoutnan.to_csv("Results_MC.csv")
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

mae_mean2020 = mae(comparsionPV_withoutnan.SAM_gen, comparsionPV_withoutnan.PV_AC)


















