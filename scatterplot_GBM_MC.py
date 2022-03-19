# -*- coding: utf-8 -*-
"""
Created on Fri Mar 18 19:02:37 2022

@author: 81095
"""

from plotnine import * 
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
from textwrap import fill
###############################################################################
# load data
Results_GBM = pd.read_csv('C:/Users/81095/PY/PV forecasting model chain/Results_GBM.csv')
Results_MC = pd.read_csv('C:/Users/81095/PY/PV forecasting model chain/Results_MC.csv')


Results_GBM.index = Results_GBM.Time
Results_GBM = Results_GBM[['predict_power','real_PV']]

# Results_GBM.loc[Results_GBM['predict_power'] > 20,'predict_power'] = 20


Results_MC.index = Results_MC.utc_time
Results_MC = Results_MC[['SAM_gen','PV_AC']]






Results = pd.concat([Results_MC,Results_GBM], axis=1)
Results = Results.dropna()

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
# error metric: mean bias error
def mbe(y_true, y_pred):
    """
    Mean absolute error
    
    Parameters
    ----------
    y_true: array
        observed value
    y_pred: array
        forecasts 
    """
    return np.mean((y_pred - y_true))


###############################################################################
# RMSE MAE

mbe_MC = mbe(Results.SAM_gen, Results.PV_AC)
mbe_GBM = mbe(Results.real_PV, Results.predict_power)

mae_MC = mae(Results.SAM_gen, Results.PV_AC)
mae_GBM = mae(Results.real_PV, Results.predict_power)

rmse_MC = mean_squared_error(Results.SAM_gen, Results.PV_AC, squared=False)
rmse_GBM = mean_squared_error(Results.real_PV, Results.predict_power, squared=False)

###############################################################################

Result_GBM = Results[['predict_power','real_PV']]
Result_GBM['Method'] = 'Gradient boosting'


Result_MC = Results[['PV_AC','SAM_gen']]
Result_MC['Method'] = 'Model chain' 

# rename
Result_GBM = Result_GBM.rename(columns={'predict_power':'Generated PV power [MW]', 'real_PV':'PV power of Jacumba [MW]'})
Result_MC = Result_MC.rename(columns={'PV_AC':'Generated PV power [MW]', 'SAM_gen':'PV power of Jacumba [MW]'})


data_target = pd.concat([Result_MC,Result_GBM], axis=0)

data_target.index = range(len(data_target))


label = {'RMSE': ['RMSE = 2.31 MW', 'RMSE = 2.16 MW'], 'MAE': ['MAE   = 1.27 MW', 'MAE   = 1.17 MW']}
data_label = pd.DataFrame(data=label, index=['Gradient boosting','Model chain'])

aes(label = label)

fig, plot = (ggplot(data_target, aes(x='PV power of Jacumba [MW]',y='Generated PV power [MW]')) +
  
  geom_abline(intercept=0,slope=1,colour="#0072B2",size=0.1,linetype="dashed") +
  geom_point(alpha=0.2,size=0.2,colour="#0072B2") +
  facet_wrap("Method") + 
  theme(axis_title = element_text(size = 7, family = "times new roman"),
                    strip_text = element_text(size = 7, family = "times new roman"),
                    axis_text = element_text(size = 7, family = "times new roman"),
                    panel_spacing = 0.02,
                    legend_title = element_text(size=7, family = "times new roman"), 
                    strip_margin_y = 0,
                    legend_position='none',
                    legend_text = element_text(size=7, family = "times new roman"),
                    figure_size=(3.5, 1.5),
                    dpi=500
                    )
).draw(return_ggplot=True)

fig.savefig('scatterplotJacumba.pdf',bbox_inches='tight')