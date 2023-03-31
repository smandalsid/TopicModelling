import pandas as pd
import numpy as np
import re
import string
import gensim
from nltk.stem import WordNetLemmatizer, SnowballStemmer
from nltk.stem.porter import *
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer

stemmer = SnowballStemmer('english')
stop_words = stopwords.words('english')
stop_words.extend(['from', 'subject', 're', 'edu', 'use'])

def lemmatize_stemming(text):
    return stemmer.stem(WordNetLemmatizer().lemmatize(text, pos='v'))

def preprocess(text):
    result = []
    for token in gensim.utils.simple_preprocess(text):
        regex = re.compile('[' + re.escape(string.punctuation) + '0-9\\r\\t\\n]')
        text = regex.sub(" ", text.lower())
        words = text.split(" ")
        words = [re.sub('\S*@\S*\s?', '', sent) for sent in words]
        words = [re.sub('\s+', ' ', sent) for sent in words]
        words = [re.sub("\'", "", sent) for sent in words]
        if token not in stop_words and len(token) > 3:
            result.append(lemmatize_stemming(token))
    return result


df=pd.read_csv('/home/siddharth/Documents/NLP/topicmodelling/modeldata/docs.csv')
docs=[x for x in df["doc"]]
titles=[x for x in df["title"]]
data=pd.read_csv('/home/siddharth/Documents/NLP/topicmodelling/modeldata/articles.csv')
articles=[x for x in data['text']]
# print(docs[0].strip("[]").replace("'", '').split(", "))

for i in range(len(docs)):
  docs[i]=docs[i].strip("[]").replace("'", '').split(", ")

# Cosine similarity
def cosine_sim(text1, text2):
    tfidf_score = TfidfVectorizer().fit_transform([text1, text2])
    return ((tfidf_score * tfidf_score.T).A)[0, 1]

# Most similar article
def closest_doc_name(sentence):
    cos = []
    for i in range(len(docs)):
        cos.append(cosine_sim(', '.join(sentence.split(' ')),', '.join(docs[i])))
    return [titles[x] for x in np.argsort(cos)[-10:][::-1]], [articles[x] for x in np.argsort(cos)[-10:][::-1]]

# print(closest_doc_name('news'))
