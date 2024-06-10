import pandas as pd
import numpy as np
import datetime as dt
from dateutil.relativedelta import relativedelta
from statsmodels.regression.rolling import RollingOLS
from statsmodels.tsa.arima.model import ARIMA
from pandas.tseries.offsets import MonthEnd
from sklearn.metrics import mean_squared_error

infl_dat_path = './data/inflation_model_data'
prediction_range = 6
rolling_window = 12

#Y
CPI = pd.read_excel(infl_dat_path+'/CPI.xlsx', index_col=0)
CPI.index = CPI.index + MonthEnd(0)
#Y for predict
Y = CPI.shift(-prediction_range).dropna()
Y.columns = ['CPI_fwd']

#X : CPI_Lag(1)
CPI_lag = CPI.copy()
CPI_lag.index = CPI_lag.index + MonthEnd(12)
CPI_lag.columns = ['CPI_lag']
#X : GDP_gap
pGDP = pd.read_excel(infl_dat_path+'/potential_GDP.xlsx', index_col=0)
pGDP.index = [dt.datetime(x, 12, 31) for x in pGDP.index]
rGDP = pd.read_excel(infl_dat_path+'/real_GDP.xlsx', index_col=0)
rGDP.index = [dt.datetime(x, 12, 31) for x in rGDP.index]
GDP_GAP = (rGDP - pGDP.values).dropna()
GDP_GAP.columns = ['GDP_gap']
#X : inflation Expectation
infl_exp = pd.read_excel(infl_dat_path+'/google_trend.xlsx', index_col=0)
infl_exp.index = infl_exp.index + MonthEnd(0)
#X : FED Confidence
fed_conf = pd.read_excel(infl_dat_path+'/google_trend_FED.xlsx', index_col=0)
fed_conf.index = fed_conf.index + MonthEnd(0)
#X : Total
X = pd.concat([CPI_lag, GDP_GAP, infl_exp, fed_conf], axis=1)
X.GDP_gap = X.GDP_gap.interpolate(method='time')
X = X.dropna()
#X_before2004: sub Total
X_before2004 = pd.DataFrame(index=pd.date_range('2000-12-31', '2004-12-31', freq='M'))
X_before2004 = pd.concat([X_before2004, CPI_lag, GDP_GAP], axis=1)
X_before2004.GDP_gap = X_before2004.GDP_gap.interpolate(method='time')
X_before2004 = X_before2004[X_before2004.index<dt.datetime(2004,12,31)]

#df
df = pd.concat([X, Y], axis=1).dropna()
#df_before2004
df_before2004 = pd.concat([X_before2004, Y], axis=1).dropna()
'''
gtrend : previous month data update in every 15th day of the month
Therefore, data we can use in end of the month is  gtrend.shift(1)

CPI : previous month data update in every 1st day of the month
Since CPI is dependant var, need CPI.shift(-3) 
'''
#'CPI_fwd ~ CPI_lag + GDP_gap + infl_exp + fed_conf -1'
model_2004 = RollingOLS.from_formula('CPI_fwd ~ GDP_gap', data=df_before2004, min_nobs=rolling_window, window=rolling_window, expanding=True)
model = RollingOLS.from_formula('CPI_fwd ~ GDP_gap + infl_exp -1', data=df, min_nobs=rolling_window, window=rolling_window, expanding=True)
res_2004 = model_2004.fit()
res = model.fit()

params = pd.concat([res_2004.params, res.params], axis=0, join='outer')
params = params.dropna(subset='GDP_gap').fillna(0)

#since we calculated parameters of rolling regressions, get historical CPI(inflation) prediction
#get total variables
X_before2004 = X_before2004[X_before2004.index<X.index[0]]
X = pd.concat([X_before2004, X], axis=0).fillna(0)
#get total coefficients
params.columns = params.columns + '_coeff'
params.index = params.index + MonthEnd(1)
#concat var and coeff
pred_var = pd.merge(X, params, how='outer', left_index=True, right_index=True).fillna(method='ffill').dropna()

pred_list=[]
for _, eachrow in pred_var.iterrows():
    prediction = eachrow.Intercept_coeff \
                 + eachrow.GDP_gap_coeff * eachrow.GDP_gap \
                 + eachrow.infl_exp_coeff * eachrow.infl_exp \
                 + eachrow.fed_conf_coeff * eachrow.fed_conf \
                 #+ eachrow.CPI_lag_coeff * eachrow.CPI_lag
    pred_list.append(prediction)

Pred_df = pd.DataFrame({'Prediction':pred_list}, index=pred_var.index)

Pred_df.to_csv(infl_dat_path+'/CPI_us_prediction.csv')

#now compare with actual CPI
CPI.columns = ['Actual']
#Pred_df.index = Pred_df.index + MonthEnd(prediction_range)
Pred_df = pd.concat([Pred_df, CPI], axis=1)

#now compare with ARIMA model
test_CPI = CPI.dropna()
test_CPI.index = pd.DatetimeIndex(test_CPI.index.values, freq='M')
predARIMA_indexlist = []
predARIMA_list = []
for idx in range(12,len(test_CPI)+1):
    input_CPI = test_CPI.iloc[:idx]
    modelARM = ARIMA(input_CPI, order=(1,1,0))
    modelARM_fit = modelARM.fit()
    predARIMA_indexlist.append(modelARM_fit.forecast(prediction_range).index[-1])
    predARIMA_list.append(modelARM_fit.forecast(prediction_range)[-1])
Pred_ARIMA = pd.DataFrame({'Prediction_ARIMA':predARIMA_list}, index=predARIMA_indexlist)
Pred_df = pd.concat([Pred_df, Pred_ARIMA], axis=1)

Pred_df = Pred_df.fillna(method='ffill').dropna()
print('each rmse of models : NKPC - {}, ARIMA(1,1) - {}'.format(
mean_squared_error(Pred_df.Prediction, Pred_df.Actual),
mean_squared_error(Pred_df.Prediction_ARIMA, Pred_df.Actual)
))

Pred_df.plot()

#look continous model's error
continous_Pred_df = pd.DataFrame()
for idx in range(len(pred_var)-5):
    temp_list = []
    temp_date_list = []
    for subidx in range(prediction_range):
        temp_date = pred_var.index[idx+subidx]
        temp_row = pred_var.iloc[idx+subidx]
        temp_coeff_row = pred_var.iloc[idx]
        prediction = temp_coeff_row.Intercept_coeff \
                     + temp_coeff_row.CPI_lag_coeff * temp_coeff_row.CPI_lag \
                     + temp_coeff_row.GDP_gap_coeff * temp_row.GDP_gap \
                     + temp_coeff_row.infl_exp_coeff * temp_row.infl_exp \
                     #+ temp_coeff_row.fed_conf_coeff * temp_row.fed_conf
        temp_list.append(prediction)
        temp_date_list.append(temp_date)
        temp_df = pd.DataFrame(temp_list, index=temp_date_list, columns=['prd'])
    continous_Pred_df = pd.concat([continous_Pred_df, temp_df], axis=1)
continous_Pred_df.to_csv(infl_dat_path+'/CPI_Continous_prediction.csv')

continous_Pred_df.index = continous_Pred_df.index + MonthEnd(prediction_range)
continous_Pred_df = pd.concat([continous_Pred_df, CPI], axis=1)

print('done')
