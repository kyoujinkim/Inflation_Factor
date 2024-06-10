import pandas as pd
import numpy as np
from glob import glob
import datetime as dt
from pandas.tseries.offsets import MonthEnd
from sklearn.linear_model import LinearRegression as LinReg
from tqdm import tqdm
import os

sector = pd.read_csv('./data/backtest_data/size.csv', index_col=0, parse_dates=True)
sector.index = sector.index + MonthEnd(0)
sim_list = glob('./factors/simil/*')
factorlist = ['D','G','V','S','L','M','Q']

#factorname = 'V'
for factorname in factorlist:
    total_df = pd.DataFrame()
    for simfile in tqdm(sim_list):
        orig_simvec = pd.read_csv(simfile, index_col=0).dropna().iloc[:,3]

        date_end = dt.datetime.strptime(simfile.split('\\')[1].split('.')[0], '%Y-%m-%d')
        #string format date to load other factor value
        date_end_str = date_end.strftime('%Y-%m-%d')

        simvec = pd.DataFrame(orig_simvec).T
        simvec.index = [date_end]

        currfactor = pd.read_csv('./factors/{}/{}.csv'.format(factorname, date_end_str), index_col=0)
        currfactor = currfactor.sort_values(by='SCORE', ascending=False).iloc[:int(len(currfactor)/5)]

        currsector = sector.loc[date_end].dropna()

        match_COMP = list(set(currfactor.index) & set(currsector.index) & set(simvec.columns))
        currsector = currsector.loc[match_COMP]

        sector_count = currsector.value_counts()
        sector_count = sector_count / sector_count.sum()
        sector_count = pd.DataFrame(sector_count).T
        sector_count.index = [date_end]

        total_df = pd.concat([total_df,sector_count])
    total_df.to_csv('./size_{}.csv'.format(factorname))
