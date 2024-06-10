import pandas as pd
import OpenDartReader
from glob import glob
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

def get_buss_detail(Dart_df, enddate):

    temp_df = pd.DataFrame(columns=['report_nm', 'rcept_no'], dtype=object)
    buss_detail_list = []
    buss_detail = None

    for i in (pbar := tqdm(Dart_df['code'])):
        try:
            code = i[1:]
            startdate = dt.strftime(dt.strptime(enddate, '%Y-%m-%d') - relativedelta(months=6), '%Y-%m-%d')
            result_list = dart.list(code, start=startdate, end=enddate, kind='A', final=False)
            try:
                result_list['Rpt_Date'] = result_list.report_nm.apply(lambda x: str.split(x, "(")[1].replace(')', ''))
                result_list = result_list.sort_values(by='Rpt_Date', ascending=False)
            except:
                pass

            #공시 검색 결과가 없는 경우, 예외 처리
            if len(result_list) == 0:
                result_temp = pd.DataFrame({'report_nm':['조회 데이터 없음'], 'rcept_no':['조회 데이터 없음']})
                buss_detail = '조회 데이터 없음'
                raise ValueError('조회 데이터 없음')

            #첫번째로 검색된 최신 리포트를 기준으로 함
            result_temp = result_list.iloc[0][['report_nm', 'rcept_no']]
            result_temp = pd.DataFrame([result_temp.values], columns=result_temp.index)

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

                code = i[1:]
                result_list = dart.list(code, end=enddate, kind='A', final=False)

                # 공시 검색 결과가 없는 경우, 예외 처리
                if len(result_list) == 0:
                    result_temp = pd.DataFrame({'report_nm': ['조회 데이터 없음'], 'rcept_no': ['조회 데이터 없음']})
                    buss_detail = '조회 데이터 없음'
                    raise ValueError('조회 데이터 없음')

                # 첫번째로 검색된 최신 리포트를 기준으로 함
                result_temp = result_list.iloc[0][['report_nm', 'rcept_no']]
                result_temp = pd.DataFrame([result_temp.values], columns=result_temp.index)

                # 추출한 결과값에 존재하는 보고서의 url 주소를 직접 가져오자
                listofsubdocs = dart.sub_docs(result_temp.values[0][1], match='사업의 내용')
                dcmno = listofsubdocs.iloc[0]['url'][:-4]

                buss_detail = Text_output(dcmno)
                #time.sleep(np.random.randint(1, 5))

            except:
                pass

        if buss_detail == None:
            buss_detail_list.append('에러 발생')
        else:
            buss_detail_list.append(buss_detail)
        temp_df = pd.concat([temp_df, result_temp])
        pbar.set_description('{} done'.format(i))

    temp_df['detail'] = buss_detail_list
    temp_df['Srch_Date'] = enddate
    temp_df = temp_df.reset_index(drop=True)

    return temp_df

api_key = 'dd'
dart = OpenDartReader(api_key)

port_list = pd.read_csv('./data/K200const.csv', index_col=0)
date_list = port_list.index.unique()

report_list = glob('./data/report_list/*')
report_list = [i.split('\\')[1].split('.')[0] for i in report_list]

for eachdate in date_list:

    if eachdate in report_list:
        continue

    if eachdate <'2022-09-30':
        continue

    print('=========={}============='.format(eachdate))

    current_port = port_list[port_list.index == eachdate].dropna()

    Dart_df = pd.DataFrame({'code':current_port.CODE.values})

    temp_df = get_buss_detail(Dart_df, eachdate)

    Dart_df = pd.concat([Dart_df, temp_df], axis=1)

    #이상값, 결측값이 존재하는 지 확인하기 위한 절차
    test_df = Dart_df.copy()
    test_df.detail = test_df.detail.apply(lambda x: 'Error Occured' if pd.isna(x) else x)
    test_df.detail = test_df.detail.apply(lambda x: 'Well Collected' if len(x) > 20 else x)

    new_list = pd.DataFrame(columns=test_df.columns)
    new_list = pd.concat([new_list, test_df[test_df.detail != 'Well Collected']])
    print(new_list)

    Dart_df.to_csv('./data/report_list/{}.csv'.format(eachdate))