from glob import glob
import pandas as pd
from datetime import datetime as dt
from tqdm import tqdm

file_list = glob('./factors/Clarifi/style_score/*')

for file in tqdm(file_list):
    scorefile = pd.read_csv(file, index_col=1)

    for colname in scorefile:
        if colname!="Date":
            rtnfile = pd.DataFrame({'CODE':scorefile.index,
                                    'SCORE':scorefile[colname]}).dropna()
            filename = colname[0]
            filedate = dt.strptime(scorefile.Date[0], '%Y-%m-%d').strftime('%Y-%m-%d')
            rtnfile.to_csv('./factors/{}/{}.csv'.format(filename, filedate), index=False)