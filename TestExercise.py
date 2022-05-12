# -*- coding: utf-8 -*-
"""
Created on Wed May 11 16:21:46 2022

@author: user
"""
#Needed packages

from pprint import pprint
import re
import gensim
import pandas as pd
import numpy as np
import gensim.corpora as corpora
from gensim.utils import simple_preprocess
from gensim.models import CoherenceModel
import spacy
import warnings
warnings.filterwarnings('ignore', category=DeprecationWarning)
from nltk.corpus import stopwords
import nltk
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import json
import sys
import os

#File input

file_path = input("Please input the path of file>> ")
num_of_topics = input("Please input number of topics>> ")
print("You are input number successfully!")


#Data cleaning and Tokenize
stop_words = stopwords.words('english')
stop_words.extend(['from','edu','subject','re','use'])

df = pd.read_csv(file_path)
df = df.dropna(subset=['text'])
df1 = df.head(10000)
data = df1.text.values.tolist()
data = [re.sub(r'\S*@\S*\s?', '', sent) for sent in data]
data = [re.sub(r'\s+', ' ', sent) for sent in data]
data = [re.sub(r"\'", "", sent) for sent in data]

#Steaming
def sent_to_words(sentences):
    for sentence in sentences:
        yield(gensim.utils.simple_preprocess(str(sentence), deacc=True)) 
data_words = list(sent_to_words(data))
#print(data_words[:1])
def lemmatization(texts, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV']): #'NOUN', 'ADJ', 'VERB', 'ADV'
    texts_out = []
    for sent in texts:
        doc = nlp(" ".join(sent)) 
        texts_out.append(" ".join([token.lemma_ if token.lemma_ not in ['-PRON-'] else '' for token in doc if token.pos_ in allowed_postags]))
    return texts_out
nlp = spacy.load(r'C:\Users\user\anaconda3\envs\job\Lib\site-packages\en_core_web_sm\en_core_web_sm-3.3.0', disable=["parser", "ner"])
data_lemmatized = lemmatization(data_words, allowed_postags=['NOUN','VERB'])

#Create Document
vectorizer = CountVectorizer(analyzer = 'word', min_df = 10, stop_words = 'english', lowercase = True, token_pattern = '[a-zA-z0-9]{3,}', max_features = 50000)

data_vectorized = vectorizer.fit_transform(data_lemmatized)

#Build a model
lda_model = LatentDirichletAllocation(n_components=int(num_of_topics), max_iter=10, learning_method= 'online', random_state=100, batch_size=128, evaluate_every=-1, n_jobs=-1)

lda_output = lda_model.fit_transform(data_vectorized)

print(lda_model)

#Document-Topic Matrix
topicnames = ['Topic' + str(i) for i in range(lda_model.n_components)]
document = ['Score' + str(i) for i in range(len(data))]

#Topic - Score Dataframe
df_document_topic = pd.DataFrame(np.round(lda_output,2), columns=topicnames, index = document)
#print(df_document_topic.head())

dominant_topic = np.argmax(df_document_topic.values, axis=1)
df_document_topic['dominant_topic'] = dominant_topic
#print(df_document_topic.head())

#Topic Keyword matrix

df_topic_keywords = pd.DataFrame(lda_model.components_)
df_topic_keywords.columns = vectorizer.get_feature_names()
df_topic_keywords.index = topicnames
#print(df_topic_keywords.head())


# Show n keywords for each topic
def show_topics(vectorizer=vectorizer, lda_model=lda_model, n_words=20):
    keywords = np.array(vectorizer.get_feature_names())
    topic_keywords = []
    for topic_weights in lda_model.components_:
        top_keyword_locs = (-topic_weights).argsort()[:n_words]
        topic_keywords.append(keywords.take(top_keyword_locs))
    return topic_keywords
topic_keywords = show_topics(vectorizer=vectorizer, lda_model=lda_model, n_words=15)

# Topic - Keywords Dataframe
df_topic_keywords = pd.DataFrame(topic_keywords)
df_topic_keywords.columns = ['Trend '+str(i) for i in range(df_topic_keywords.shape[1])]
df_topic_keywords.index = ['Topic '+str(i) for i in range(df_topic_keywords.shape[0])]
#print(df_topic_keywords)
df_document_topic.drop('dominant_topic', inplace = True, axis = 1)
df_document_topic1 = df_document_topic
data_trend = df_topic_keywords['Trend 1']

#JSON format
{
 "topics":[
     print(df_document_topic1.to_json(orient=columns))
     
     ],
 "trends":
     print(data_trend.to_json(orient = columns))
 } 

