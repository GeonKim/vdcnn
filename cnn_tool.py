import numpy as np
import pandas as pd
import re
import tensorflow as tf
import random
from konlpy.tag import Kkma

####################################################
# cut words function                               #
####################################################
def cut(contents):
    results = []
    for content in contents:
        words = content.split() 
        result = []
        for word in words: 
            result.append(word) 
        results.append(' '.join([token for token in result]))
    return results

####################################################
# token words function                             #
####################################################
def tokenize(contents):
    return contents.split(' ')

def get_token_id(token, vocab):
    if token in vocab:
        return vocab[token]
    else:
        return 0

####################################################
# divide train/test set function                   #
####################################################
def divide(x, y, train_prop):
    random.seed(1234)
    x = np.array(x)
    y = np.array(y)
    tmp = np.random.permutation(np.arange(len(x)))
    x_tr = x[tmp][:round(train_prop * len(x))]
    y_tr = y[tmp][:round(train_prop * len(x))]
    x_te = x[tmp][-(len(x)-round(train_prop * len(x))):]
    y_te = y[tmp][-(len(x)-round(train_prop * len(x))):]
    return x_tr, x_te, y_tr, y_te


####################################################
# making input function                            #
####################################################
def make_input(documents, max_document_length):
    
    # tensorflow.contrib.learn.preprocessing 내에 VocabularyProcessor라는 클래스를 이용
    # 모든 문서에 등장하는 단어들에 인덱스를 할당
    # 길이가 다른 문서를 max_document_length로 맞춰주는 역할
    
    
    vocab_processor = tf.contrib.learn.preprocessing.VocabularyProcessor(max_document_length)  # 객체 선언
    x = np.array(list(vocab_processor.fit_transform(documents)))
    ### 텐서플로우 vocabulary processor
    # Extract word:id mapping from the object.
    # word to ix 와 유사
    vocab_dict = vocab_processor.vocabulary_._mapping
    # Sort the vocabulary dictionary on the basis of values(id).
    sorted_vocab = sorted(vocab_dict.items(), key=lambda x: x[1])
    # Treat the id's as index into list and create a list of words in the ascending order of id's
    # word with id i goes at index i of the list.
    # print(list(zip(*sorted_vocab)))


    vocabulary = list(list(zip(*sorted_vocab))[0])
    return x, vocabulary, len(vocab_processor.vocabulary_)

####################################################
# save vocabulary                                 #
####################################################
def save_vocab(filename, documents, max_document_length):
    # tensorflow.contrib.learn.preprocessing 내에 VocabularyProcessor라는 클래스를 이용
    # 모든 문서에 등장하는 단어들에 인덱스를 할당
    # 길이가 다른 문서를 max_document_length로 맞춰주는 역할
    vocab_processor = tf.contrib.learn.preprocessing.VocabularyProcessor(max_document_length)  # 객체 선언
    x = np.array(list(vocab_processor.fit_transform(documents)))
    ### 텐서플로우 vocabulary processor
    # Extract word:id mapping from the object.
    # word to ix 와 유사
    vocab_dict = vocab_processor.vocabulary_._mapping
    # Sort the vocabulary dictionary on the basis of values(id).
    sorted_vocab = sorted(vocab_dict.items(), key=lambda x: x[1])
    # Treat the id's as index into list and create a list of words in the ascending order of id's
    # word with id i goes at index i of the list.

    with open(filename, 'w', encoding='utf-8') as f:
        for i in vocab_dict:
            f.write('%s,%d\n' %(i, vocab_dict[i]))

    # vocabulary = list(list(zip(*sorted_vocab))[0])
    # return x, vocabulary, len(vocab_processor.vocabulary_)

####################################################
# make output function                             #
####################################################
def make_output(points, threshold):
    results = np.zeros((len(points),2))
    for idx, point in enumerate(points):
        if point > threshold:
            results[idx,0] = 1
        else:
            results[idx,1] = 1
    return results

####################################################
# check maxlength function                         #
####################################################
def check_maxlength(contents):
    max_document_length = 0
    for document in contents:
        document_length = len(document.split())
        if document_length > max_document_length:
            max_document_length = document_length
    return max_document_length

####################################################
# loading function                                 #
####################################################
#data_path = 'SK_news_data.csv'
#import pandas as pd
#import numpy as np
def loading_rdata(data_path):
    # R에서 title과 contents만 csv로 저장한걸 불러와서 제목과 컨텐츠로 분리
    # write.csv(corpus, data_path, fileEncoding='utf-8', row.names=F)
    corpus = pd.read_table(data_path, sep=",", encoding="utf-8")
    corpus = np.array(corpus)
    contents = []
    points = []
    for idx,doc in enumerate(corpus): #전체 데이터 셋을 읽어들여서 반복한다.
        print(idx,doc)
        # idx = 뉴스 개수
        # doc[0] = 뉴스 입력 날짜
        # doc[1] = 뉴스 기사
        # doc[2] = 뉴스 업/다운
        #print(idx+2, len(doc[1]))
        if len(doc[1]) > 100: #뉴스 본문의 길이가 100을 넘는다면
            contents.append(doc[1]) #contents에 본문 추가하고
            points.append(doc[2]) #points에 스코어를 추가한다
    return contents, points      #contents, points 리스트를 반환한다.

def load_vocab(filename):
    result = dict()
    with open(filename, 'r', encoding='utf-8') as f:
        for line in f.readlines():
            ls = line.split(',')
            result[ls[0]] = int(ls[1])

    return result

def isNumber(s):
  try:
    float(s)
    return True
  except ValueError:
    return False