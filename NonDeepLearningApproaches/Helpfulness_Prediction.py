# -*- coding: utf-8 -*-
"""
Created on Sat Apr 21 14:33:55 2018

@author: Enri
"""

# Author: Matt Terry <matt.terry@gmail.com>
#
# License: BSD 3 clause
from __future__ import print_function

import numpy as np

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.datasets.twenty_newsgroups import strip_newsgroup_footer
from sklearn.datasets.twenty_newsgroups import strip_newsgroup_quoting
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.svm import LinearSVC
from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.metrics import classification_report
from sklearn.pipeline import FeatureUnion
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
import pickle
from sklearn import metrics
from sklearn.model_selection import train_test_split
import pandas as pd
from nltk.stem import SnowballStemmer
from nltk.tokenize import RegexpTokenizer
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold
import nltk


def pickleRead(filename):
    data_f = open("pickled_vars/%s.pickle" % filename, "rb")
    data = pickle.load(data_f)
    data_f.close()
    return data

def pickleWrite(variable, filename):
    saveData = open("pickled_vars/%s.pickle" % filename,"wb")
    pickle.dump(variable, saveData)
    saveData.close()

def getGlobalNLTKVars():
    #Global NLTK Variables
    ret = RegexpTokenizer('[a-zA-Z0-9\']+')
    sw = set(stopwords.words('english'))
    #lemmaTokenizer
    wnl = WordNetLemmatizer()
    #stemTokenizer
    ess = SnowballStemmer('english', ignore_stopwords=True)
    #POSTokenizer
    allowed_word_types = ["ADJ", "ADV", "NOUN", "VERB"]
    return ret, sw, wnl, ess, allowed_word_types

def lemmaTokenizer(sentence):
    tokens= ret.tokenize(sentence)
    #return [wnl.lemmatize(t) for t in tokens if t not in sw]
    return [wnl.lemmatize(t) for t in tokens]

def stemTokenizer(sentence):
    tokens= ret.tokenize(sentence)
    return [ess.stem(t) for t in tokens if t not in sw]

def POSTokenizer(sentence, ret):
    tokens= ret.tokenize(sentence)
    pos = nltk.pos_tag(tokens, tagset = "universal")
    return [wnl.lemmatize(pos[i][0]) for i in range(0, len(tokens)) 
            if pos[i][1] in allowed_word_types]


class ItemSelector(BaseEstimator, TransformerMixin):
    def __init__(self, key):
        self.key = key

    def fit(self, x, y=None):
        return self

    def transform(self, data_dict):
        return data_dict[self.key]

def count_uppercase(word_array):
    counter = 0
    word_array = [word_array]
    count_vect = CountVectorizer(lowercase = False)
    X_train_counts = count_vect.fit_transform(word_array)
    dict_vect = count_vect.vocabulary_
    #print(dict_vect)
    if len(dict_vect) != 0:
        dictlist = []
        for key, value in dict_vect.items():
            dictlist.append(key)
        for word in dictlist:
            if word == word.upper():
                counter += 1

    return counter

def count_occurences(character, word_array):
    counter = 0
    for j, word in enumerate(word_array):
        for char in word:
            if char == character:
                counter += 1
    #print("? Counter", counter)
    return counter


class TextStats(BaseEstimator, TransformerMixin):
    """Extract features from each document for DictVectorizer"""

    def fit(self, x, y=None):
        return self

    def transform(self, posts):
        return [{'length': len(text)}  for text in posts]
        #return [{'length': len(text.split(' '))}  for text in posts]
        #return [{'number_?': count_occurences('?', text), 
        #         'length': len(text)} for text in posts]
        #return [{'uppercase_num': count_uppercase(text)}  for text in posts]


pipeline = Pipeline([
    # Use FeatureUnion to combine the features from subject and body
    ('union', FeatureUnion(
        transformer_list=[

            # Pipeline for standard bag-of-words model for body
            ('body_bow', Pipeline([
                
                ('vect', CountVectorizer(tokenizer = stemTokenizer,
                                         
                                ngram_range=(1, 2),
                                binary = True, min_df=2, max_df=0.8)),
                        ('tfidf', TfidfTransformer()),
                
            ])),
            
            # Pipeline for pulling ad hoc features from post's body
            ('body_stats', Pipeline([
                ('stats', TextStats()),  # returns a list of dicts
                ('vect', DictVectorizer()),  # list of dicts -> feature matrix
            ])),
        ],
        # weight components in FeatureUnion
        transformer_weights={
            'body_bow': 0.85,
            'body_stats': 0.15,
        },
    )),
    ('chi2', SelectKBest(chi2, k=5000)),
    # Use a SVC classifier on the combined features
    #('clf', SGDClassifier( penalty='elasticnet')),
    ('clf', MultinomialNB(fit_prior=False)),
    #('clf', LogisticRegression()),
    #('clf', RandomForestClassifier(n_estimators=100))
])

def preprocess(data):
    balancedData = []
    for i in range(0, len(data)):
        
        helpful = data[i].groupby('helpful').size()
        print("Helpful/Unhelpful distribution: \n", helpful)
        
        helpfulDistribution = data[i].helpful.value_counts().sort_index()
        helpfulDistribution = helpfulDistribution.min(axis = 0)
        
        unhelpfulRev = data[i][data[i].helpful == -1]
        
        helpfulRev = data[i][data[i].helpful == 1]
        helpfulRevLimited = helpfulRev.iloc[:helpfulDistribution ,:]
        
        reviews = pd.concat([unhelpfulRev, helpfulRevLimited])
        
        balancedData.append(reviews)
    return balancedData


data = pickleRead("helpfulBiningDataWithout0")
cleanedDataAll = pickleRead("helpfulBiningDataWithout0All")

helpful = cleanedDataAll.groupby('helpful').size()
print("Helpful/Unhelpful distribution: \n", helpful)

#Balanced Data
cleanedData = preprocess(data)

#Global NLTK Variables
ret, sw, wnl, ess, allowed_word_types = getGlobalNLTKVars()

"""
#Unique Train Test Split
X = cleanedData[2].reviewText
y = cleanedData[2].helpful 

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.4, 
                                                    random_state=1)
pipeline.fit(X_train, y_train)
y = pipeline.predict(X_test)
print("Model accuracy: %.2f%%" % (100 * metrics.accuracy_score(y_test, y)))
print(classification_report(y, y_test))
"""

#KFold CrossValidation aprroach

#cleanedData = data[2]
#Balanced Data
cleanedData = preprocess(data)

#Balanced Data All Reviews UnHelpful
unhelpfulRev0 = cleanedData[0][cleanedData[0].helpful == -1]
unhelpfulRev1 = cleanedData[1][cleanedData[1].helpful == -1]
unhelpfulRev2 = cleanedData[2][cleanedData[2].helpful == -1]

currentReviews = pd.concat([unhelpfulRev0, unhelpfulRev1])

unhelpfulReviews = pd.concat([currentReviews, unhelpfulRev2])

        
#Balanced Data All Reviews Helpful
helpfulRev0 = cleanedData[0][cleanedData[0].helpful == 1]
helpfulRev1 = cleanedData[1][cleanedData[1].helpful == 1]
helpfulRev2 = cleanedData[2][cleanedData[2].helpful == 1]

currentReviews = pd.concat([helpfulRev0, helpfulRev1])

helpfulReviews = pd.concat([currentReviews, helpfulRev2])

helpfulReviewsAll = pd.concat([unhelpfulReviews, helpfulReviews])

pickleWrite(helpfulReviewsAll, 'helpfulReviewsAll')

helpfulReviewsAll = pickleRead('helpfulReviewsAll')

#cleanedData = cleanedData[2]

#cleanedData = cleanedDataAll
cleanedData = helpfulReviewsAll


cleanedData = cleanedData.reset_index(drop=True)

kf = KFold(n_splits=3, shuffle=True)

scores = np.array([])

iterator = 1

for train, test in kf.split(cleanedData):
    
    X_train = cleanedData.loc[train ,"reviewText"]
    y_train = cleanedData.loc[train, "helpful"]
    
    X_test = cleanedData.loc[test ,"reviewText"]
    y_test = cleanedData.loc[test, "helpful"]
    
    X_train = X_train.reset_index(drop=True)
    X_test = X_test.reset_index(drop=True)
    y_train = y_train.reset_index(drop=True)
    y_test = y_test.reset_index(drop=True)
    
    print("Split number %d" % iterator)
    print("Train first 10 indexes: ", train[0:10])
    print("Test first 10 indexes: ", test[0:10])
    print("Train set size: %d // Test set size: %d" % (len(train), len(test)))

    #Make sure that has been read correctly
    print ("The first review is:")
    #print (X_train.iloc[0, "reviewText"])
    
    pipeline.fit(X_train, y_train)
    y = pipeline.predict(X_test)
    
    score = accuracy_score(y_test, y)
    print(score)
    
    scores = np.append(scores, score)
    
    iterator += 1

scores_mean = np.mean(scores) 
scores_std = scores.std() * 2  

print("Accuracy: %0.2f%% (+/- %0.3f)" % (scores_mean*100, scores_std))


rev1 = X_test[0]
rev2 = X_test[10]

reviewsList = [rev1, rev2]

txt_clf.predict(X_test)
