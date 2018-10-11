#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct  2 16:22:04 2018

@author: pirl
"""

import lime
import sklearn
import numpy as np
import sklearn
import sklearn.ensemble
import sklearn.metrics
from __future__ import print_function

##fetching data, training a classifier
from sklearn.datasets import fetch_20newsgroups
    # Scikit-Learn 패키지는 분류(classification) 모형의 테스트를 위해 가상 데이터 생성 함수를 제공한다.
categories = ['alt.atheism', 'soc.religion.christian'] 
    # 카테고리 만들고 
newsgroups_train = fetch_20newsgroups(subset='train', categories=categories)
newsgroups_train.data
type(newsgroups_train)
    # train 데이터 가져옴
    '''
    # 데이터 타입 :sklearn.bunch
    scikit-learn에 포함된 데이터셋은 실제 데이터와 데이터셋 관련 정보를 담고 있는 Bunch 객체에 저장되어 있습니다. 
    Bunch 객체는 파이썬 딕셔너리dictionary와 비슷하지만 점 표기법을 사용할 수 있습니다
    (즉 bunch[‘key’] 대신 bunch.key를 사용할 수 있습니다).
    
    # 데이터 형태
        data : 한꺼번에 들어간 text data
        target_names = categories 
        target = 0,1 로 만들어진 array
    '''
    
newsgroups_test = fetch_20newsgroups(subset='test', categories=categories)
    # test 데이터 가져옴
    
class_names = ['atheism', 'christian']


vectorizer = sklearn.feature_extraction.text.TfidfVectorizer(lowercase=False)
    # TfidfVectorizer: tf-idf 하게 해주는 함수, 문서 집합으로부터 단어의 수를 세고 TF-IDF 방식으로 단어의 가중치를 조정한 BOW 벡터를 만듦
    # ? lowercase = False : 대문자를 무시하지 않음 
 
    
print(vectorizer)  
    
##############################벡터 생성 

train_vectors = vectorizer.fit_transform(newsgroups_train.data)
test_vectors = vectorizer.transform(newsgroups_test.data)
    # counter vector로 만듦 : fit_transform이랑 transform의 차이는 잘 모르겠음. 용도는 같음 
    # -> 결과는 동일한데 fit이 더 효율적이라 함 

train_vectors.toarray().shape


rf = sklearn.ensemble.RandomForestClassifier(n_estimators=500)
    # 앙상블 중 랜덤포레스트 분류를 실행함(의사결정나무 겁나 많이하는 거)
    # n_estimators = 트리의 개수 
    
rf.fit(train_vectors, newsgroups_train.target)
    # fit(X, y[, sample_weight]) : Build a forest of trees from the training set (X, y)


pred = rf.predict(test_vectors)
    # prediction 하고
    
sklearn.metrics.f1_score(newsgroups_test.target, pred, average='binary')
    # f1 점수 구하는 함수 
    # f1 점수 : 정밀도와 재현율의 조화 평균 

# Explaining predictions using lime 
from lime import lime_text
from sklearn.pipeline import make_pipeline
c = make_pipeline(vectorizer, rf) 
    #?? 파이프라인 만듦(?파이프라인?) :와 돌릴 모델

print(c.predict_proba([newsgroups_test.data[0]]))
    #?? predict_proba : 조건부확률 계산
    # 의사결정나무(랜덤포레스트)는 조건부 확률 모형 중 하나
        # 각 카테고리 혹은 클래스가 정답일 조건부 확률(conditional probability)를 계산함 

from lime.lime_text import LimeTextExplainer
explainer = LimeTextExplainer(class_names=class_names)
    #  ‘Explainer’ : 모델에서 가장 중요했던 증상을 강조해줌 


idx = 83
exp = explainer.explain_instance(newsgroups_test.data[idx], c.predict_proba, num_features=6)
    # explainer.explain_instance() : 특정 텍스트에 대한 설명 생성 
print('Document id: %d' % idx)
print('Probability(christian) =', c.predict_proba([newsgroups_test.data[idx]])[0,1])
    # 크리스천일 조건부 확률
    
print('True class: %s' % class_names[newsgroups_test.target[idx]])
        

exp.as_list()


print('Original prediction:', rf.predict_proba(test_vectors[idx])[0,1])
tmp = test_vectors[idx].copy()
tmp[0,vectorizer.vocabulary_['Posting']] = 0
tmp[0,vectorizer.vocabulary_['Host']] = 0
print('Prediction removing some features:', rf.predict_proba(tmp)[0,1])
print('Difference:', rf.predict_proba(tmp)[0,1] - rf.predict_proba(test_vectors[idx])[0,1])


# Visualinzing 
%matplotlib inline
fig = exp.as_pyplot_figure()
exp.show_in_notebook(text=False)
