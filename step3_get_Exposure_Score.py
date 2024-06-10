import pandas as pd
import numpy as np
from glob import glob
import datetime as dt
from pandas.tseries.offsets import MonthEnd
from sklearn.linear_model import LinearRegression as LinReg
from tqdm import tqdm
import os

dat_path = './data/backtest_data'
bmprice = pd.read_csv(dat_path+'/bmprice.csv', index_col=0, parse_dates=True)
adjprice = pd.read_csv(dat_path+'/adjprice.csv', index_col=0, parse_dates=True)
rfrate = pd.read_csv(dat_path+'/rfrate.csv', index_col=0, parse_dates=True)
cpi_kr = pd.read_excel('./data/inflation_model_data/CPI.xlsx',index_col=0, parse_dates=True)
cpi_kr.index = cpi_kr.index + MonthEnd(0)

sim_list = glob('./factors/simil/*')
factorlist = ['D','G','V','S','L','M','Q']
quantile=10
betamultiple=False

total_list = []
for simfile in tqdm(sim_list):
    orig_simvec = pd.read_csv(simfile, index_col=0).dropna().iloc[:,3]

    date_end = dt.datetime.strptime(simfile.split('\\')[1].split('.')[0], '%Y-%m-%d')
    date_strt = date_end + MonthEnd(-12*5)
    split_price = adjprice[(adjprice.index<=date_end) & (adjprice.index>=date_strt)]
    split_bmprice = bmprice[(bmprice.index<=date_end) & (bmprice.index>=date_strt)]
    split_rfrate = rfrate[(rfrate.index<=date_end) & (rfrate.index>=date_strt)]
    split_cpi = cpi_kr[(cpi_kr.index<=date_end) & (cpi_kr.index>=date_strt)]

    #string format date to load other factor value
    date_end_str = date_end.strftime('%Y-%m-%d')
    rowwise_list = [date_end]
    for factorname in factorlist:
        simvec = orig_simvec.copy()
        valvec = pd.read_csv('./factors/{}/{}.csv'.format(factorname, date_end_str), index_col=0)

        if betamultiple:
            #get companies only which has been listed more than 1 year
            split_price = split_price.T[split_price.count()>=252].T
            #match with split_price and simvec
            match_COMP = list(set(split_price.columns) & set(simvec.index))
            #filter by match_comp list
            split_price = split_price[match_COMP]
            simvec = simvec.loc[match_COMP]

            #calculate monthly 5year beta
            #get monthly return of price and bm_price
            daterange_m = pd.date_range(start=date_strt, end=date_end, freq='M')
            split_price_mrtn = split_price.reindex(daterange_m, method='nearest').pct_change().iloc[1:]
            split_bmprice_mrtn = split_bmprice.reindex(daterange_m, method='nearest').pct_change().iloc[1:]
            split_rfrate_m = split_rfrate.reindex(daterange_m, method='nearest').iloc[1:]
            split_cpi_mchg = (split_cpi.reindex(daterange_m, method='nearest')
                              - split_cpi.reindex(daterange_m, method='nearest').shift(1).values).iloc[1:]

            betaReturn = split_bmprice_mrtn - ((split_rfrate_m/100+1)**(1/12)-1).values
            betaMacro = split_cpi_mchg
            betaData = pd.concat([betaReturn,betaMacro], axis=1)
            betaData.columns = ['MktRtn','Macro']

            #get beta value
            beta = []
            for label, y in split_price_mrtn.items():
                y = y.dropna()
                X = betaData.reindex(y.index)
                t = LinReg().fit(X,y)
                beta.append([label, *t.coef_])
            beta = pd.DataFrame(beta, columns=['CODE', 'betaReturn', 'betaMacro']).set_index(keys='CODE')

            #now concatenate with simvec
            match_COMP = list(set(simvec.index) & set(beta.index))
            simvec = simvec.loc[match_COMP]
            beta = beta.loc[match_COMP].betaMacro

            #standardize values and bound
            simvec_std = ((simvec - simvec.mean()) / simvec.std()).clip(lower=-3, upper=3)
            beta_std = ((beta - beta.mean()) / beta.std()).clip(lower=-3, upper=3)

            avg_exposure = simvec_std*0.5 + beta_std.values*0.5
            #get only half of companies which have top exposure
            #avg_exposure = avg_exposure.sort_values(ascending=False).iloc[:int(len(avg_exposure)/2)]

        else:
            avg_exposure = simvec

        valvec = valvec.iloc[:,-1].sort_values(ascending=False).iloc[:int(len(valvec)/quantile)]
        match_COMP = list(set(avg_exposure.index) & set(valvec.index))
        avg_exposure = avg_exposure.loc[match_COMP]
        valvec = valvec.loc[match_COMP]
        simvec = simvec.loc[match_COMP]

        rowwise_list.extend([(avg_exposure.apply(lambda x: np.sign(x)) * simvec).mean()])

    total_list.append(rowwise_list)

total_df = pd.DataFrame(total_list, columns=['DATE']+factorlist).set_index(keys='DATE', drop=True)
total_df.to_csv('./factors/sim_score_{}.csv'.format(quantile))