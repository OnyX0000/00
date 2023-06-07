#!/usr/bin/env python
# coding: utf-8

# In[2]:


import csv
import sys
import concurrent.futures
import warnings
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from konlpy.tag import Okt
import streamlit as st
import requests
import jpype
from toml import load
import streamlit as st

# config.toml 파일 로드
config = load("config.toml")

# JVM 경로 설정
jvm_path = config["JVM"]["path"]
jpype.startJVM(jvm_path)

# JVM 클래스 경로 설정
classpath_path = config["JVMClasspath"]["path"]
jpype.addClassPath(classpath_path)

csv.field_size_limit(2**31 - 1)  # Set a smaller field size limit

csv_filename = "https://raw.githubusercontent.com/OnyX0000/00/main/allend_law_example.csv"
csv_filename2 = "https://raw.githubusercontent.com/OnyX0000/00/main/allend_cases_example.csv"

def calculate_sentence_similarity(word, sentence):
    okt = Okt()
    tokenizer = okt.morphs
    vectorizer = TfidfVectorizer(tokenizer=tokenizer)

    corpus = [word] + [sentence]
    tfidf = vectorizer.fit_transform(corpus).toarray()

    similarity_scores = cosine_similarity(tfidf[-1].reshape(1, -1), tfidf[:-1]).flatten()

    return similarity_scores

def load_data_from_csv(csv_url):
    with st.spinner("데이터를 로딩 중입니다..."):
        response = requests.get(csv_url)
        content = response.content.decode('utf-8')
        csv_reader = csv.reader(content.splitlines())
        headers = next(csv_reader)  # 헤더를 읽고 넘어갑니다.
        rows = list(csv_reader)

    return headers, rows

def calculate_sentence_similarities(word, sentences):
    okt = Okt()
    tokenizer = okt.morphs
    vectorizer = TfidfVectorizer(tokenizer=tokenizer)
    tfidf_matrix = vectorizer.fit_transform(sentences)
    word_tokens = okt.morphs(word)
    word_vector = vectorizer.transform([' '.join(word_tokens)])
    cosine_similarities = cosine_similarity(tfidf_matrix, word_vector)
    return cosine_similarities

def main():
    st.title("유사도기반 키워드검색")
    word = st.text_input("단어를 입력하세요(각각의 단어는 공백으로 구분)")

    if word:
        # 문장들을 저장할 리스트
        sentences = []

        # CSV 파일 읽기
        _, rows = load_data_from_csv(csv_filename)
        sentences = [row[1] for row in rows]

        # TfidfVectorizer로 문장들을 벡터화
        cosine_similarities = calculate_sentence_similarities(word, sentences)

        # 코사인 유사도가 가장 높은 상위 5개의 행 인덱스
        top_indices = cosine_similarities.argsort(axis=0)[-5:].flatten()[::-1]

        # 상위 5개의 행 저장
        output_data = []
        top_scores = []
        _, rows = load_data_from_csv(csv_filename)
        headers = rows[0]
        for i, row in enumerate(rows):
            if i in top_indices:
                output_data.append(row[1:])
                top_scores.append(cosine_similarities[i][0])

        # 결과를 DataFrame으로 변환
        output_df = pd.DataFrame(output_data, columns=headers[1:])
        output_df['유사도 점수'] = top_scores
        output_df = output_df.sort_values(by='유사도 점수', ascending=False).head(5)

        # 결과 출력
        st.subheader("법령")
        st.dataframe(output_df.rename(columns={headers[1]: '조문내용'}))

        # 판례 데이터 로딩 및 처리
        headers, rows = load_data_from_csv(csv_filename2)

        top_scores = []
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            with concurrent.futures.ThreadPoolExecutor() as executor:
                similarity_scores = []
                for row in rows[1:]:
                    sentences = row[1:6]
                    similarity_scores.clear()

                    sentence_similarity = executor.map(lambda s: calculate_sentence_similarity(word, s), sentences)
                    for scores in sentence_similarity:
                        similarity_scores.extend(scores)

                    if similarity_scores:
                        max_scores = sorted(similarity_scores, reverse=True)[:5]
                        if len(max_scores) >= 5:
                            extracted_row = row
                            top_scores.append((row[0], max_scores, extracted_row))

        top_scores_with_avg = []
        for id, scores, row in top_scores:
            avg_score = sum(scores) / len(scores)
            top_scores_with_avg.append((id, scores, avg_score, row))

        top_scores_with_avg.sort(key=lambda x: x[2], reverse=True)

        output_cases_df = pd.DataFrame(columns=headers)
        for _, _, avg_score, extracted_row in top_scores_with_avg[:5]:
            avg_score_rounded = round(avg_score, 10)
            extracted_row.append(avg_score_rounded)
            output_cases_df = pd.concat([output_cases_df, pd.Series(extracted_row[:len(headers)], index=headers)], ignore_index=True)

        # 비어있는 행 제거
        output_cases_df.dropna(axis=0, how='all', inplace=True)

        # 비어있는 열 제거
        output_cases_df.dropna(axis=1, how='all', inplace=True)

        st.subheader("판례")
        st.dataframe(output_cases_df)


if __name__ == '__main__':
    main()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




