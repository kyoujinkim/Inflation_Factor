import pandas as pd
from glob import glob
from konlpy.tag import Mecab
from gensim.models import doc2vec
from tqdm import tqdm
from multiprocessing import Pool
import re, html, os

'''
Initialization : 시작을 위한 기반 데이터 호출 및 정의
'''
#텍스트 cleansing을 위한 패턴 리스트
EMAIL_PATTERN = re.compile(r'''(([a-zA-Z0-9._%+-]+)@([a-zA-Z0-9.-]+)(\.[a-zA-Z]{2,4}))''')
URL_PATTERN = re.compile("(ftp|http|https)?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+")
#ACCOUNT_PATTERN = re.compile('@[a-zA-Z]*')
NUMBER_PATTERN = re.compile('\d+')
MULTIPLE_SPACES = re.compile(' +', re.UNICODE)
removal_list = "_$#@|‘ ’ ◇ ‘ ”  ’ · \“ · △ ➤ ●   ■ ( ) \" >> ` / -∼=ㆍ<> ? !【】 … ◆%"
PASS_POS = ['NNG','SL','VV','VA','MM']

# 형태소 분석을 위해 mecab Load - 딥 러닝을 이용한 자연어 처리 입문(https://wikidocs.net/book/2155) 참조
mecab = Mecab(dicpath=r"C:\mecab\mecab-ko-dic")

def get_clean_text(text):
    '''
    텍스트 클린징을 위한 알고리즘
    :param text: string 클린징이 요구되는 텍스트
    :return: string 클린징이 완료된 텍스트
    '''
    text = text.lower()
    text = text.replace("\n", ' ').replace("\'s", '')
    text = re.sub(EMAIL_PATTERN, ' ', text)   # 이메일 패턴 제거
    text = re.sub(URL_PATTERN, ' ', text)   # URL 패턴 제거
    text = re.sub(re.compile('<.*?>'), '', text)   # HTML 태그 제거 (e.g., <br>)
    text = html.unescape(text)   # HTML 코드 변환 (&#39; -> ')
    text = text.translate(str.maketrans(removal_list, ' '*len(removal_list)))   # 특수문자 제거
    text = re.sub(NUMBER_PATTERN, ' ', text) # 숫자 제거
    #text = re.sub(NUMBER_PATTERN, '[NUM]', text) # 숫자 비식별화 처리
    text = re.sub(MULTIPLE_SPACES, ' ', text)   # 무의미한 공백 제거
    return text

def get_morphs(df):
    '''
    코드가 태깅된 morphs 리스트를 반환하는 함수
    :param df: list multiprocessing을 위하여 list 형식 input으로 변경 및 함수화
    :return: doc2vec tagged document
    '''
    text = df[1]
    text = get_clean_text(text)
    code = df[0]

    # remove element by pos of morphs
    text_withPOS = mecab.pos(text)
    for el in text_withPOS[:]:  # as we iterate through list elements, should use copy version of list
        if el[1] not in PASS_POS:
            text_withPOS.remove(el)
    text_morphs = pd.DataFrame(text_withPOS).iloc[:, 0].values

    #Genism의 Doc2vec은 10,000 개 이상의 Token을 처리하지 못한다. 따라서 추가설정이 요구됨
    if len(text_morphs) > 6000:
        text_morphs_list = []
        for idx in range(0,len(text_morphs),2000):
            if len(text_morphs[idx:idx+6000])<4000:
                continue
            text_morphs_list.append(doc2vec.TaggedDocument(tags=[code], words=text_morphs[idx:idx+6000]))

        return text_morphs_list
    else:
        return [doc2vec.TaggedDocument(tags=[code], words=text_morphs)]

if __name__ == '__main__':

    # data base의 용량이 큰 관계로 mutliprocessing pool을 선제적으로 호출
    p = Pool(processes=os.cpu_count()-1)

    # 저장해 놓은 DART 사업의 내용 Loading
    total_rpt = pd.read_csv('./data/report_list/total_rpt4Q04_2Q22_adj.csv', index_col=0, encoding='UTF-8')

    # 인플레이션 관련 파일을 읽어온 후, 전처리 수행
    text_infl_list = glob('./data/text_inflation/*')
    text_infl_label = [str.split(str.split(str.split(x, '\\')[1], '_')[1], '.')[0] for x in text_infl_list]
    text_infl_arr = []
    for text_infl in text_infl_list:
        with open(text_infl, 'r', encoding='UTF-8') as file:
            temp_text = file.read()
        text_infl_arr.append(temp_text)

    text_infl_df = pd.DataFrame({'code': text_infl_label,
                                 'detail': text_infl_arr})

    tagged_corpus_list_infl = []
    for df in text_infl_df.values.tolist():
        tagged_corpus_list_infl.append(get_morphs(df))

    srch_date = total_rpt.Srch_Date.unique()

    #여기부터는 날짜별로 Loop를 돌려서 따로 학습 진행
    for currdate in srch_date:

        #skip if we already have model
        if './simil\\{}.csv'.format(currdate) in glob('./simil/*'):
            continue

        print('=========={}=========='.format(currdate))

        #screen dataset which match current date, also drop company which has no business detail
        temp_rpt = total_rpt[total_rpt.Srch_Date==currdate]
        temp_rpt = temp_rpt[['code', 'detail']]
        temp_rpt = temp_rpt[temp_rpt.detail.apply(lambda x: len(str(x)) > 20)]

        tagged_corpus_list_comp = []
        pbar = tqdm(total=len(temp_rpt))
        for i, res in enumerate(p.imap_unordered(get_morphs, temp_rpt.values.tolist())):
            pbar.update()
            tagged_corpus_list_comp.append(res)
        pbar.close()

        #append inflation text to company business text morphs
        tagged_corpus_list_comp.extend(tagged_corpus_list_infl)

        #flat list of tagged documents
        tagged_corpus_list_comp = [subx for x in tagged_corpus_list_comp for subx in x if subx!=None]

        model = doc2vec.Doc2Vec(dm=1,
                                vector_size=50,
                                alpha=0.025,
                                min_alpha=0.0001,
                                min_count=3,
                                epochs=5000,
                                workers=os.cpu_count()-1,
                                window=8,
                                seed=111)

        # Vocabulary 빌드
        model.build_vocab(tagged_corpus_list_comp)
        print(f"Tag Size: {len(model.dv.index_to_key)}", end=' / ')

        # Doc2Vec 학습
        model.train(tagged_corpus_list_comp, total_examples=model.corpus_count, epochs=50)
        print('total spent time : {} s'.format(round(model.total_train_time,2)))

        # 모델 저장
        model.save('./models/dart_{}.doc2vec'.format(currdate))

        simil_nmwiki = pd.DataFrame(model.dv.most_similar('nmwiki', topn=1000), columns=['code', 'nmwiki_simil']).set_index('code')
        simil_KDI = pd.DataFrame(model.dv.most_similar('KDI', topn=1000), columns=['code', 'kdi_simil']).set_index('code')
        simil_wiki = pd.DataFrame(model.dv.most_similar('wiki', topn=1000), columns=['code', 'wiki_simil']).set_index('code')
        simil_total = pd.concat([simil_nmwiki, simil_KDI, simil_wiki], axis=1)
        simil_total['avg_simil'] = simil_total.mean(axis=1)
        simil_total = simil_total.sort_values(by='avg_simil', ascending=False)

        simil_total.to_csv('./simil/{}.csv'.format(currdate))