# Probabilistic solar forecasting
## An archived dataset from the ECMWF Ensemble Prediction System for probabilistic solar forecasting

### Requirments：
The code is written in Python, and some packages should be installed before the scripts can be executed smoothly.  
  * Package [pandas](https://pandas.pydata.org/pandas-docs/stable/index.html) is a fast, BSD-licensed library that provides high-performance data structures and data analysis tools.
  * Package [numpy](https://numpy.org/doc/stable/) is the fundamental package for scientific computing.
  * Package [pvlib](https://pvlib-python.readthedocs.io/en/stable/) provides functions for simulating the performance of PV systems.
  * Package [scipy](https://scipy.org/) provides algorithms for scientific computing.
  * Package [sklearn](https://scikit-learn.org/stable/) is the basic package for machine learning.

Other Python packages that used for plotting the results include [seaborn](https://seaborn.pydata.org/), [matplotlib](https://matplotlib.org/), and [plotnine](https://plotnine.readthedocs.io/en/stable/). 

### Data: 
A total of 7 [ENS_XXX.csv](https://github.com/wentingwang94/probabilistic-solar-forecasting/tree/main/data), [Jacumba_ENS.csv](https://github.com/wentingwang94/probabilistic-solar-forecasting/blob/main/data/Jacumba_ENS.csv), [McClear_Jacumba.csv](https://github.com/wentingwang94/probabilistic-solar-forecasting/blob/main/data/McClear_Jacumba.csv), [ECMWF_HRES.csv](https://github.com/wentingwang94/probabilistic-solar-forecasting/blob/main/data/ECMWF_HRES.csv), and [60947.csv](https://github.com/wentingwang94/probabilistic-solar-forecasting/blob/main/data/60947.csv) files are provided. These files contain four years (2017--2020) of the ECMWF ENS forecast data for seven SURFRAD stations (xxx denotes the three-letter station abbreviations), four years (2017--2020) of the ECMWF ENS forecasting data for Jacumba solar plant, clear-sky irradiance for Jacumba solar plant, ECMWF HRES forecast data, and Jacumba solar power data whose Energy Information Administration (EIA) plant ID is 60947.


### Code: 
A total of three Python scripts, namely, [post-processing.py](https://github.com/wentingwang94/probabilistic-solar-forecasting/blob/main/code/post-processing.py), [GradientBoosting.py](https://github.com/wentingwang94/probabilistic-solar-forecasting/blob/main/code/GradientBoosting.py), and [ModelChain.py](https://github.com/wentingwang94/probabilistic-solar-forecasting/blob/main/code/ModelChain.py) are provided for reproducibility. The file names are self-explanatory. In that, the [post-processing.py] provide the operational post-processing of NWP-based solar forecast at seven research-grade ground-based stations; [GradientBoosting.py](https://github.com/wentingwang94/probabilistic-solar-forecasting/blob/main/code/GradientBoosting.py) reproduces the irradiance-to-power conversion approach using gradient boosting; and [ModelChain.py](https://github.com/wentingwang94/probabilistic-solar-forecasting/blob/main/code/ModelChain.py) provides the irradiance-to-power conversion approach using model chain. To use these scripts, the user only needs to change the working directory. 
