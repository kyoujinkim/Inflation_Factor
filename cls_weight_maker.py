import pandas as pd
import numpy as np
import datetime as dt
from dateutil.relativedelta import relativedelta
import riskfolio as rp
from riskfolio.ParamsEstimation import mean_vector
from tqdm import tqdm
tqdm.pandas()
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

def row_eval(row, list):
    return row.loc[list]

class WeightBuilder:
    def __init__(self,
                 port_array,
                 mktcap_array=None,
                 price_array=None,
                 return_array=None,
                 bm_array=None,
                 bm_return_array=None,
                 strtdate=None,
                 enddate=None,
                 rebal_freq='1M',
                 window=12,
                 interpolation=False):
        '''
        :param port_array: 월별 포트폴리오 구성종목
        :param mktcap_array: 시가총액 행렬
        :param price_array: 과거 일간 가격 행렬
        :param return_array: 가격 대신 return값을 넣어도 무방
        :param bm_array: Benchmark 가격 행렬
        :param bm_return_array: Benchmark 수익률 행렬
        :param strtdate: 포트폴리오 비중 계산 시작일, None인 경우 port_array 기준
        :param enddate: 포트폴리오 비중 계산 종료일, None인 경우 port_array 기준
        :param rebal_freq: 포트폴리오 비중 계산 주기
        :param window: 비중계산에 사용되는 가격 시계열 길이
        :param interpolation: 주어진 종목 외에 사이 간격을 추가
        '''

        self.bm_arr = bm_array

        if return_array is None and price_array is None:
            raise Exception('price or return input required')
        self.price_arr = price_array
        if return_array is None:
            self.rtn_arr = price_array.pct_change().iloc[1:]
        else:
            self.rtn_arr = return_array

        self.bm_arr = bm_array
        if bm_return_array is None and bm_array is not None:
            self.bm_rtn_arr = bm_array.pct_change().iloc[1:]
        else:
            self.bm_rtn_arr = bm_return_array

        self.port_arr = port_array
        self.datearr = self.port_arr.DATE.unique()

        self.mktcap_arr = mktcap_array

        if strtdate==None:
            self.strtdate = self.datearr[0]
        else:
            self.strtdate = strtdate

        if enddate==None:
            self.enddate = self.datearr[-1]
        else:
            self.enddate = enddate

        if interpolation==True:
            inter_date = pd.date_range(self.strtdate, self.enddate, freq=rebal_freq)
            inter_date = [eachdate.strftime('%Y-%m-%d') for eachdate in inter_date]
            inter_date_df = pd.DataFrame({'INTERDT':inter_date}).set_index(keys='INTERDT', drop=False)
            datearr_pd = pd.DataFrame({'DATE':self.datearr}).set_index(keys='DATE', drop=False)
            datearr_list = []
            for _, daterow in inter_date_df.iterrows():
                try:
                    datearr_list.append(datearr_pd[datearr_pd.DATE < daterow.INTERDT].DATE[-1])
                except:
                    datearr_list.append(datearr_pd.DATE[0])
            self.datearr = pd.DataFrame({'INTERDT':inter_date_df.INTERDT,
                                         'DATE':datearr_list})
        else:
            inter_date_df = pd.DataFrame({'INTERDT':self.datearr}).set_index(keys='INTERDT', drop=False)
            datearr_pd = pd.DataFrame({'DATE':self.datearr}).set_index(keys='DATE', drop=False)
            self.datearr = pd.concat([inter_date_df, datearr_pd], axis=1).fillna(method='ffill').drop_duplicates('INTERDT')

        self.window = window

    def calc(self,
             method,
             custom_corr_dir=None,
             custom_factor_dir=None,
             quantile=None):
        '''
        :param method: mkt(market cap weighting):시가총액 기준 방법론
                    eqw(equal weighting): 동일비중 방법론
                    mvo(mean-variance optimization weighting): 평균 분산 최적화 방법론
                    rpo(risk parity optimization weighting): 리스크 패리티 방법론
                    hrp(hierachical risk parity optimization weighting): 계층적 군집화 방법론

        :param custom_corr_dir: 커스텀 상관관계 행렬 csv 주소
        :param custom_factor_dir: 커스텀 팩터값 csv 주소
        :param quantile: quantile 구분 투자시 수익률을 알고 싶은 경우 입력 (1q10)
        :return: 비중값 반환
        '''

        weights=[]
        for _, daterow in tqdm(self.datearr.iterrows(), total=len(self.datearr)):
            eachdate = daterow.INTERDT
            updatedate = daterow.DATE

            #주어진 조건에 맞는 구간만 비중값 계산
            if eachdate < self.strtdate: continue
            if eachdate > self.enddate: break
            eachdate_dt = dt.datetime.strptime(eachdate, "%Y-%m-%d")

            windowdate_dt = eachdate_dt - relativedelta(months=self.window)
            #각 월별 종목 정보 take

            each_comp = self.port_arr[self.port_arr.DATE==updatedate].dropna().sort_index().drop_duplicates()

            each_rtn = self.rtn_arr[(self.rtn_arr.index <= eachdate_dt) & (self.rtn_arr.index > windowdate_dt)].dropna(how='all', axis=1)

            #return array와 portfolio의 종목 개수가 다르므로 통일시켜주는 작업 시행
            match_COMP = list(set(each_rtn.columns) & set(each_comp.CODE))
            each_comp = each_comp[each_comp.CODE.isin(match_COMP)]

            #return array 중 필요한 부분만 취득
            each_rtn = each_rtn[match_COMP]

            '''
            if self.bm_rtn_arr is None:
                cov_rtn = each_rtn
                cov_bm_rtn = None
            else:
                each_bm_rtn = self.bm_rtn_arr[(self.bm_rtn_arr.index <= eachdate_dt) & (self.bm_rtn_arr.index > windowdate_dt)]
                cov_rtn = each_rtn
                cov_bm_rtn = each_rtn - each_bm_rtn.values
            '''

            if custom_corr_dir is None:
                cov_rtn = each_rtn
                cov = cov_rtn.cov()
            else:
                custom_corr = pd.read_csv(custom_corr_dir+'/{}.csv'.format(updatedate), index_col=0)

                # return array와 portfolio의 종목 개수가 다르므로 통일시켜주는 작업 시행
                match_COMP = list(set(custom_corr.columns) & set(each_comp.CODE))
                each_comp = each_comp[each_comp.CODE.isin(match_COMP)]
                each_rtn = each_rtn[match_COMP]
                cov_rtn = each_rtn

                #현재 포트폴리오 순서에 맞게 custom 상관관계 재정렬
                custom_corr = custom_corr[each_comp.CODE].loc[each_comp.CODE]

                #스프레드를 잘 정의하는 것이 중요
                self.factor = (custom_corr - cov_rtn.corr()).mean().sort_values(ascending=False)
                self.factor = self.factor - self.factor.min()

                #if cov_bmrtn is not None:
                #    self.spread = self.spread[((cov_bmrtn+1).product()>1)==True]

                #custom 상관관계에 맞추어 custom 공분산 재계산
                diagstd = np.diag(cov_rtn.std())
                cov = pd.DataFrame(np.matmul(np.matmul(diagstd, custom_corr), diagstd))
                cov.index = each_comp.index
                cov.columns = each_comp.index

            if custom_factor_dir is not None:
                factor_score = pd.read_csv(custom_factor_dir+'/{}.csv'.format(updatedate), index_col=0)

                # return array와 portfolio의 종목 개수가 다르므로 통일시켜주는 작업 시행
                match_COMP = list(set(factor_score.index) & set(each_comp.CODE))
                each_comp = each_comp[each_comp.CODE.isin(match_COMP)]
                each_rtn = each_rtn[match_COMP]
                cov_rtn = each_rtn

                # 현재 포트폴리오 순서에 맞게 factor array 재정렬
                factor_score = factor_score.loc[each_comp.CODE]

                if len(factor_score.columns) > 1:
                    factor_score = factor_score[factor_score.columns[-1]].sort_values(ascending=False)
                else:
                    factor_score = factor_score.sort_values(ascending=False, by=factor_score.columns[-1])

                self.factor = factor_score

            if quantile is not None:
                nth_q = int(quantile.split('q')[0])
                tot_q = int(quantile.split('q')[1])
                drop_unit = int(len(self.factor)/tot_q)

                factor_top = self.factor.iloc[int(drop_unit*(nth_q-1)):int(drop_unit*nth_q)] #
                factor_bottom = self.factor.iloc[-int(drop_unit*1):]
                self.factor_bottom = self.factor.loc[factor_bottom.index]
                self.factor = self.factor.loc[factor_top.index]

                each_comp = each_comp[each_comp.CODE.isin(factor_top.index)]
                each_rtn = each_rtn[factor_top.index]
                cov = cov_rtn.cov()
                cov = cov[factor_top.index].loc[factor_top.index]


            #다음 비중 계산을 위하여 self로 값 전달: 좋은 방법인지는 잘 모르겠음. 어차피 포트 크기가 별로 안크니까...
            self.temp_comp = each_comp
            self.temp_rtn = each_rtn.fillna(0)
            self.temp_cov = cov.fillna(0)
            if self.mktcap_arr is not None:
                try:
                    self.temp_mkt = self.mktcap_arr[each_comp.CODE].loc[eachdate_dt].fillna(0)
                except:
                    self.temp_mkt = self.mktcap_arr[each_comp.CODE].iloc[-1].fillna(0)

            weight = eval('self.{}'.format(method))()
            weight.index = [eachdate_dt]
            weights.append(weight)

        print('{} weight building complete'.format(method))
        weights = pd.concat(weights, axis=0)

        return weights

    def mkt(self):
        weight = pd.DataFrame(self.temp_mkt/self.temp_mkt.sum()).T

        return weight

    def equ(self):
        weight = [1 / len(self.temp_comp)] * len(self.temp_comp)
        weight = pd.DataFrame([weight], columns=self.temp_comp.CODE)

        return weight

    def equLS(self):
        weight = [1 / len(self.temp_comp)] * len(self.temp_comp)
        weight = pd.DataFrame([weight], columns=self.temp_comp.CODE)

        weightbot = [1 / len(self.factor_bottom)] * len(self.factor_bottom)
        weightbot = -pd.DataFrame([weightbot], columns=self.factor_bottom.index)

        weight = pd.concat([weight, weightbot],axis=1)

        return weight

    def mvo(self):
        port = rp.Portfolio(returns=self.temp_rtn, sht=True)
        port.mu = mean_vector(self.temp_rtn)
        port.cov = self.temp_cov
        model = 'Classic'
        rm = 'MV'
        obj = 'Sharpe'
        hist = True

        weight = pd.DataFrame(port.optimization(model=model, rm=rm, obj=obj, hist=hist)).T

        return weight

    def rpo(self):
        port = rp.Portfolio(returns=self.temp_rtn)
        port.mu = mean_vector(self.temp_rtn)
        port.cov = self.temp_cov
        model = 'Classic'
        rm = 'MV'
        hist = True

        weight = port.rp_optimization(model=model, rm=rm, hist=hist).T

        return weight

    def hrp(self):
        port = rp.HCPortfolio(returns=self.temp_rtn)
        model = 'HRP'  # HRP, HERC, HERC2, NCO
        #codependence = 'custom_cov'
        covariance = 'custom_cov' #'hist' or 'custom_cov'
        rm = 'MV'
        linkage = 'average'

        weight = port.optimization(model=model, covariance=covariance
                                   , rm=rm, custom_cov=self.temp_cov, linkage=linkage).T

        return weight

    def cosp(self):
        self.factor = self.factor - self.factor.min()
        weight = pd.DataFrame(self.factor / self.factor.sum()).T

        return weight

    def cospLS(self):
        #self.spread = self.spread - self.spread.min()
        weight = pd.DataFrame(self.factor / self.factor.sum()).T

        #self.spreadbot = self.spreadbot - self.spreadbot.min()
        weightbot = -pd.DataFrame(self.factor_bottom / self.factor_bottom.sum()).T

        weight = pd.concat([weight, weightbot], axis=1)

        return weight

    def onlyOne(self):
        onestock = self.factor[0:1]
        weight = pd.DataFrame(onestock/onestock).T

        return weight
