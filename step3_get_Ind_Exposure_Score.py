import pandas as pd
import numpy as np
from glob import glob
import datetime as dt
from pandas.tseries.offsets import MonthEnd
from sklearn.linear_model import LinearRegression as LinReg
from tqdm import tqdm
import os

sim_list = glob('./factors/simil/*')
factorlist = ['D','G','V','S','L','M','Q']

total_list = []
total_df = pd.DataFrame()
for simfile in tqdm(sim_list):
    orig_simvec = pd.read_csv(simfile, index_col=0).dropna().iloc[:,3]

    date_end = dt.datetime.strptime(simfile.split('\\')[1].split('.')[0], '%Y-%m-%d')
    #string format date to load other factor value
    date_end_str = date_end.strftime('%Y-%m-%d')

    simvec = pd.DataFrame(orig_simvec).T
    simvec.index = [date_end]

    total_df = pd.concat([total_df,simvec])

total_df.to_csv('./total_simil_score.csv')
