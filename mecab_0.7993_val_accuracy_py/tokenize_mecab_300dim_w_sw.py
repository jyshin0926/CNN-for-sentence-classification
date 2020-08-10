#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np


# In[2]:


train_txt = pd.read_csv('ratings_train.txt',sep='\t')
train_txt


# In[3]:


from konlpy.tag import *


# In[4]:


train_txt = train_txt.dropna().reset_index(drop=True)


# In[5]:


train_txt['document'] = train_txt['document'].apply(lambda x:x.replace('.',' '))


# In[6]:


hannanum = Hannanum()
kkma = Kkma()
komoran = Komoran()
mecab = Mecab()
okt = Okt()


# In[7]:


# 데이터에 존재하는 모든 n-grams 이용하면 차원 커지므로 
# 빈도수 카운팅으로 분석에 유의미한 n-gram 추출

from collections import defaultdict
def get_ngram_counter(docs, min_count=10):
    ngram_counter = defaultdict(int)
    for doc in docs:
        words = mecab.pos(doc, join=True) # 형태소와 품사 태그 set return
        for ngram in words:
            ngram_counter[ngram] += 1

    ngram_counter = {
        ngram:count for ngram, count in ngram_counter.items()
        if count >= min_count
    }

    return ngram_counter

ngram_counter = get_ngram_counter(train_txt.document)


# In[8]:


class NgramTokenizer:

    def __init__(self, ngrams, base_tokenizer):
        self.ngrams = ngrams
        self.base_tokenizer = base_tokenizer

    def __call__(self, sent):
        return self.tokenize(sent)

    def tokenize(self, sent):
        if not sent:
            return []
        unigrams = self.base_tokenizer.pos(sent, join=True)
        return unigrams

ngram_tokenizer = NgramTokenizer(ngram_counter, mecab)


# In[9]:


from tqdm import tqdm

token_list = []
for doc in tqdm(train_txt.document):
    token_list.append(ngram_tokenizer(doc))


# In[10]:


train_txt['token'] = token_list
train_txt


# In[13]:


for item in train_txt['token'].tolist():
    if len(item) != 0:
        item = ''.join(str(e)+',' for e in item)
        item = item.replace('[','').replace(']','')
    else:
        item = ''


# In[12]:


for i in range(len(train_txt['token'].tolist())):
    if len(item) != 0:
        train_txt['token'][i] = ''.join(str(e)+',' for e in train_txt['token'][i])
        train_txt['token'][i] = train_txt['token'][i].replace('[','').replace(']','')
    else:
        train_txt['token'][i] = ''


# In[14]:


train_txt['tokens_'] = train_txt['token'].apply(lambda x: ' '.join([w.split('/')[0] for w in x.split(',')]).replace('[','').replace("'",""))


# In[15]:


train_txt


# In[69]:


train_df = pd.read_pickle('train_mecab_df.pkl')
train_df


# In[16]:


tokens_list = [tokens.split() for tokens in train_txt['tokens_'].tolist()]


# In[17]:


from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize


# ### Word2Vec 모델 생성

# In[18]:


from gensim.models import Word2Vec

word2vec = Word2Vec(
    sentences = tokens_list,
    size = 300,
    min_count = 2,
    window = 10,
    sg = 1,
    hs = 1,
    iter = 10
)


# In[19]:


word2vec.save('NaverMovie_mecab300.model')


# In[20]:


model = Word2Vec.load('NaverMovie_mecab300.model')


# In[22]:


print(model.most_similar(positive=['최고']))


# In[23]:


print(model.most_similar(positive=['재미']))


# In[91]:


word_embedding = word2vec.wv.__getitem__('평점')
word_embedding

