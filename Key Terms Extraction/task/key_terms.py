# Write your code here
import string
from collections import Counter
import nltk;
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer()

import xml.etree.ElementTree as ET
root = ET.parse('news.xml').getroot()

stories = []
clean_stories = []
english_stopwords = stopwords.words('english')
puncts = list(string.punctuation)
lemmatizer = WordNetLemmatizer()



def lemmatize(text):
    return [lemmatizer.lemmatize(word) for word in text]

def remove_stopwords(text):
    return [word for word in text if word not in nltk.corpus.stopwords.words('english') and word not in puncts]

def remove_non_nouns(text):
   return [word for word in text if nltk.pos_tag([word])[0][1] == "NN"]

def calc_TF(text):
    vectorizer.fit_transform(text)
    print(vectorizer.get_feature_names())


for news in root[0]:
    headline = news[0].text
    text = nltk.word_tokenize(news[1].text.lower())
    text = lemmatize(text)
    text = remove_stopwords(text)
    text = remove_non_nouns(text)
    calc_TF(text)
    text.sort(reverse=True)
    story = (headline, Counter(text))
    stories.append(story)


for story in stories:
    print(f'{story[0]}:\n{" ".join([x[0] for x in story[1].most_common(5)])}\n')


corpus = [
    'This is the first document.',
    'This document is the second document.',
    'And this is the third one.',
    'Is this the first document?',
]

calc_TF(corpus)
