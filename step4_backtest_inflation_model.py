import pandas as pd
import numpy as np
import datetime as dt
from dateutil.relativedelta import relativedelta
from pandas.tseries.offsets import MonthEnd

infl_dat_path = './data/inflation_model_data'
index_dat_path = './data/backtest_result'

act_cpi = pd.read_excel(infl_dat_path + '/CPI.xlsx', index_col=0, parse_dates=True)
act_cpi.columns = ['Actual']
act_cpi.index = act_cpi.index + MonthEnd(0)

prd_cpi = pd.read_csv(infl_dat_path + '/CPI_prediction.csv', index_col=0, parse_dates=True)
cpred_cpi = pd.read_csv(infl_dat_path + '/CPI_Continous_prediction.csv', index_col=0, parse_dates=True)

cpi_df = pd.concat([act_cpi, prd_cpi], axis=1).dropna()
a = 1