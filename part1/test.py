import pandas as pd
import re
import string
import gensim
from nltk.stem import WordNetLemmatizer, SnowballStemmer
from nltk.stem.porter import *
import pickle
from nltk.corpus import stopwords

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

loaded_model = pickle.load(open('/home/siddharth/Documents/NLP/finalized_model.sav', 'rb'))
loaded_vectorizer=pickle.load(open('/home/siddharth/Documents/NLP/finalized_tfidf_vectorizer.sav', 'rb'))
topic_df=pd.read_csv("topics_data.csv")
while(1):
    doc=input("Enter doc: ")
    doc=[preprocess(doc)]
    testweight = loaded_model.transform(loaded_vectorizer.transform(doc))
    print("Topic: ",topic_df["topics"][testweight.argmax(axis=1)[0]], "\n")
