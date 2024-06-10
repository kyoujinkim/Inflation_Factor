import pandas as pd
import bt
import riskfolio as rp
from riskfolio.ParamsEstimation import mean_vector
import cls_weight_maker as wm

portarray = pd.read_csv('./data/K200const.csv')
parray = pd.read_csv('./data/backtest_data/adjprice.csv', index_col=0, parse_dates=True)
dvdarray = pd.read_csv('./data/backtest_data/dvd.csv', index_col=0, parse_dates=True)
mktcaparray = pd.read_csv('./data/backtest_data/mktcap.csv', index_col=0, parse_dates=True)

wb = wm.WeightBuilder(port_array=portarray,
                      mktcap_array=mktcaparray,
                      price_array=parray,
                      strtdate='2004-12-31',
                      enddate='2022-09-30',
                      rebal_freq='3M')

quantile_list = [['V','1q5','equ'],
                 ['G','1q5','equ'],
                 ['M','1q5','equ'],
                 ['D','1q5','equ'],
                 ['S','1q5','equ'],
                 ['Q','1q5','equ'],
                 ['L','1q5','equ']]

total_res = pd.DataFrame()
columnslist = []
for factor_dir, quantile, method in quantile_list:
    columnslist.append(factor_dir)
    weight = wb.calc(method=method, custom_factor_dir='./factors/'+factor_dir, quantile=quantile)
    #weight = wb.calc(method='mkt', custom_factor_dir='./simil', quantile=None)

    backtest_price = parray[weight.columns][(parray.index >= weight.index[0])].ffill()
    backtest_dividend = dvdarray[weight.columns][(dvdarray.index >= weight.index[0])].fillna(0)
    weight.index.values[0] = backtest_price.index[0]

    securities = [bt.CouponPayingSecurity(name) for name in weight.columns.values]
    s = bt.Strategy('test'
                    , [bt.algos.WeighTarget(weight), bt.algos.Rebalance()]
                    , children=securities
                    )
    t = bt.Backtest(s, backtest_price
                    , commissions=lambda q, p: abs(q) * 0.00215
                    , additional_data={'coupons': backtest_dividend}
                    )
    res = bt.run(t)
    #res.prices.to_csv('./{}_{}_{}.csv'.format(factor_dir,quantile,method))
    total_res = pd.concat([total_res, pd.DataFrame(res.prices)],axis=1)

total_res.columns = columnslist
total_res.to_csv('./total_res.csv')