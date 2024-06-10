import pandas as pd
import numpy as np
import OpenDartReader
from glob import glob
import time
import urllib3
import urllib.request as urlreq
urllib3.disable_warnings()
from tqdm import tqdm
from dateutil.relativedelta import relativedelta
from datetime import datetime as dt
from bs4 import BeautifulSoup
import re


def Text_output(url_link):

    url_open = urlreq.urlopen((url_link)).read()
    soup = BeautifulSoup(url_open, 'html5lib')
    wording = soup.select('body')[0].get_text().replace("\n","").replace("\xa0","")
    return wording

def proc_xml(xml_doc):

    starttag = '<?xml version="1.0" encoding="utf-8"?>'

    excepttag = re.compile(r'(<(DOCUMENT|FORMULA|SUMMARY|COMPANY|EXTRACTION|BODY|COVER|P|COVER|'
                           r'TABLE|COLGROUP|COL|TBODY|TR|TD|TU|TH|PGBRK|SECTION|TITLE|IMAGE|IMG|'
                           r'LIBRARY|THEAD|TE|SPAN|INSERTION|COMMENT|'
                           r'/DOCUMENT|/FORMULA|/SUMMARY|/COMPANY|/EXTRACTION|/BODY|/COVER|/P|/COVER|'
                           r'/TABLE|/COLGROUP|/COL|/TBODY|/TR|/TD|/TU|/TH|/PGBRK|/SECTION|/TITLE|/IMAGE|/IMG|'
                           r'/LIBRARY|/THEAD|/TE|/SPAN|/INSERTION|/COMMENT)(.*?)>)')

    xml_temp = xml_doc
    find_keyword = '<SECTION-1'
    for _ in range(0, 2):
        xml_doc_start = xml_temp.find(find_keyword)
        xml_temp = xml_temp[xml_doc_start+len(find_keyword):]
    xml_doc_start_sub = xml_temp.find(find_keyword)
    xml_doc_start = xml_temp.find('II. 사업의 내용')
    #in case we have more <section-1>
    if abs(xml_doc_start - xml_doc_start_sub) > 150:
        xml_doc_start = xml_temp.find(find_keyword)
        xml_temp = xml_temp[xml_doc_start+len(find_keyword):]

        #그럼에도 여전히 하나가 더 남은 경우
        xml_doc_start_sub2 = xml_temp.find('I. 회사의 개요')
        if xml_doc_start_sub2 < 150:
            xml_doc_start = xml_temp.find(find_keyword)
            xml_temp = xml_temp[xml_doc_start + len(find_keyword):]

        xml_doc_start = xml_temp.find('II. 사업의 내용')

    xml_temp = xml_temp[xml_doc_start:]
    xml_doc_end = xml_temp.find('</LIBRARY>')
    xml_temp = xml_temp[:xml_doc_end]
    xml_temp = xml_temp.replace('&cr;', '')
    xml_temp = re.sub(excepttag, '', xml_temp)
    xml_temp = xml_temp.replace('\xa0','')
    xml_temp = re.sub(re.compile(r'(\n)+'), '\n', xml_temp)
    xml_temp = re.sub(re.compile(r'( )+'), ' ', xml_temp)
    xml_temp = re.sub(re.compile(r'(-)+'), '-', xml_temp)
    xml_temp = xml_temp.replace('\n', '. ')

    return xml_temp

def get_buss_detail(arr):

    temp_df = pd.DataFrame(columns=['report_nm', 'rcept_no', 'Srch_Date'], dtype=object)
    buss_detail_list = []
    buss_detail = None
    pbar = tqdm(total=len(arr))
    for i in arr.iterrows():
        fullcode = i[1][0]
        enddate = i[1][4]
        report_nm = i[1][1]
        rcept_no = i[1][2]
        try:
            code = fullcode[1:]
            startdate = dt.strftime(dt.strptime(enddate, '%Y-%m-%d') - relativedelta(months=6), '%Y-%m-%d')

            result_list = dart.list(code, start=startdate, end=enddate, kind='A', final=False)
            try:
                result_list['Rpt_Date'] = result_list.report_nm.apply(lambda x: str.split(x, "(")[1].replace(')', ''))
                result_list = result_list.sort_values(by='Rpt_Date', ascending=False)
            except:
                pass

            #공시 검색 결과가 없는 경우, 예외 처리
            if len(result_list) == 0:
                result_temp = pd.DataFrame({'report_nm':['조회 데이터 없음'],
                                            'rcept_no':['조회 데이터 없음'],
                                            'Srch_Date':[enddate]})
                buss_detail = '조회 데이터 없음'
                raise ValueError('조회 데이터 없음')

            #첫번째로 검색된 최신 리포트를 기준으로 함
            result_list = result_list[result_list.report_nm.apply(lambda x: '첨부정정' not in x)]
            result_temp = result_list.iloc[0][['report_nm', 'rcept_no']]
            result_temp = pd.DataFrame([result_temp.values], columns=result_temp.index)
            result_temp['Srch_Date'] = enddate

            #사업의 내용 부분 텍스트 추출
            buss_detail = proc_xml(dart.document(result_temp.values[0][1]))

            #in case we scrap other report
            if '감사대상업무' in buss_detail[:100] or '내부회계관리제도' in buss_detail[:100]:
                raise ValueError('Scraped wrong report.')
            elif buss_detail == '':
                raise ValueError('Scraped wrong report.')
        except:
            try:
                if buss_detail == '조회 데이터 없음':
                    raise ValueError('조회 데이터 없음')

                code = fullcode[1:]
                result_list = dart.list(code, end=enddate, kind='A', final=False)

                # 공시 검색 결과가 없는 경우, 예외 처리
                if len(result_list) == 0:
                    result_temp = pd.DataFrame({'report_nm': ['조회 데이터 없음'],
                                                'rcept_no': ['조회 데이터 없음'],
                                                'Srch_Date': [enddate]})
                    buss_detail = '조회 데이터 없음'
                    raise ValueError('조회 데이터 없음')

                # 첫번째로 검색된 최신 리포트를 기준으로 함
                result_list = result_list[result_list.report_nm.apply(lambda x: '첨부정정' not in x)]
                result_temp = result_list.iloc[0][['report_nm', 'rcept_no']]
                result_temp = pd.DataFrame([result_temp.values], columns=result_temp.index)
                result_temp['Srch_Date'] = enddate

                # 추출한 결과값에 존재하는 보고서의 url 주소를 직접 가져오자
                dcmno = dart.sub_docs(result_temp.values[0][1], match='사업의 내용').iloc[0]['url'][:-4]

                buss_detail = Text_output(dcmno)
                time.sleep(np.random.randint(1, 5))

            except:
                pass

        if buss_detail == None:
            buss_detail_list.append('에러 발생')
        else:
            buss_detail_list.append(buss_detail)
        temp_df = pd.concat([temp_df, result_temp])
        pbar.set_description('{} done'.format(fullcode))
        pbar.update(1)

    temp_df['detail'] = buss_detail_list
    temp_df = temp_df.reset_index(drop=True)

    return temp_df

api_key = '97a20740d8e5f3ad251b0f7fa5c1ee0f9ef29114'
#my api key(naver) = '97a20740d8e5f3ad251b0f7fa5c1ee0f9ef29114'
#my api key = '16166465c8bcc04467fedc3f5ff3226d01baebf9'
#young hoon's api key1 = '63be3029d2da89a10a14778bfbb3cddfd0e10bf3'
#young hoon's api key2 = '609aa14de6f4e7f8934053574dc3e73d71c0aa9e'
dart = OpenDartReader(api_key)

report_list = glob('./data/report_list/*')
base_rpt = pd.read_csv('./data/report_list/2022-09-30.csv', index_col=0)

#일치하지 않는 종목 체크
base_rpt['rpt_chk'] = base_rpt.detail.apply(lambda x: 1 if (('사업의 내용' in str(x))*('참조' not in str(x)[:30])*('참고' not in str(x)[:30])
                                                              or ('조회 데이터 없음' in str(x))) else 0)
#일치하지 않는 종목 추가 검색을 위해 필터링된 행렬 제작
filtered_rpt = base_rpt[base_rpt.rpt_chk==0]
survived_rpt = base_rpt[base_rpt.rpt_chk==1]
survived_rpt = survived_rpt.drop(columns='rpt_chk', axis=1)

temp_df = get_buss_detail(filtered_rpt) #167

Dart_df = pd.DataFrame({'code': filtered_rpt.code.values})
Dart_df = pd.concat([Dart_df, temp_df], axis=1)

total_rpt = pd.concat([survived_rpt, Dart_df]).sort_values(by='Srch_Date').drop_duplicates()
total_rpt.to_csv('./data/report_list/total_rpt.csv')

#이상값, 결측값이 존재하는 지 확인하기 위한 절차
test_df = Dart_df.copy()
test_df.detail = test_df.detail.apply(lambda x: 'Error Occured' if pd.isna(x) else x)
test_df.detail = test_df.detail.apply(lambda x: 'Well Collected' if len(x) > 20 else x)

new_list = pd.DataFrame(columns=test_df.columns)
new_list = pd.concat([new_list, test_df[test_df.detail != 'Well Collected']])
print(new_list)