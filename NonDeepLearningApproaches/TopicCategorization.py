# -*- coding: utf-8 -*-
"""
Created on Tue Mar 20 18:49:30 2018

@author: Enri
"""

#Dependencies 
import time
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.svm import LinearSVC
from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import re
import nltk
from collections import OrderedDict
from sklearn.feature_selection import SelectKBest, chi2

from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import metrics
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import matplotlib.pyplot as plt
import seaborn as sns
from nltk.stem import SnowballStemmer
from nltk.tokenize import RegexpTokenizer
from nltk.stem import WordNetLemmatizer
from sklearn.datasets import fetch_20newsgroups
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
import pickle


"""LOAD DATA"""
#Pickle method to load a stored and precomputed variable
def pickleRead(filename):
    data_f = open("pickled_vars/%s.pickle" % filename, "rb")
    data = pickle.load(data_f)
    data_f.close()
    return data

#Pickle method to store a variable
def pickleWrite(variable, filename):
    saveData = open("pickled_vars/%s.pickle" % filename,"wb")
    pickle.dump(variable, saveData)
    saveData.close()

#Load data
#Input [filenames]: String list of dataset filenames
#Output [data]: Dataframe list of n datasets
def loadData(filenames):
    filesNumber = len(filenames)
    data = []
    for i in range(0, filesNumber):
        newData = pd.read_json("json/%s.json" % filenames[i], lines = True)
        data.append(newData)
    return data

"""PREPROCESSING"""
#Limit the amount of data for each instance for non deep learning algorithms
def preprocess(data):
    filteredData = []
    for i in range(0, len(data)):
        newData = data[i].loc[:1000, ["reviewText", "category"]]
        filteredData.append(newData)
    cleanedData = pd.concat([filteredData[0], filteredData[1]])
    cleanedData = pd.concat([cleanedData, filteredData[2]])
    cleanedData = cleanedData.reset_index(drop=True)
    return cleanedData

#Limit the amount of data for each instance for deep learning algorithms
def preprocessDeep(data):
    filteredData = []
    for i in range(0, len(data)):
        newData = data[i].loc[:10000, ["reviewText", "category"]]
        filteredData.append(newData)
    cleanedData = pd.concat([filteredData[0], filteredData[1]])
    cleanedData = pd.concat([cleanedData, filteredData[2]])
    cleanedData = cleanedData.reset_index(drop=True)
    return cleanedData

"""LEARNING ALGORITHM TRAINING"""
def learnModel(cleanedData):
    X = cleanedData.reviewText
    y = cleanedData.category 
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.4, 
                                                            random_state=1)
    
    txt_clf = Pipeline([('vect', TfidfVectorizer(stop_words="english", binary = True
                                                  )),
                         ('chi2', SelectKBest(chi2, k=5000)),
                          ('clf', MultinomialNB(fit_prior=False))
                          #('clf', RandomForestClassifier())
                          #('clf', LogisticRegression())
                          #('clf', SGDClassifier( penalty='elasticnet'))
                          ,])
    
    txt_clf.fit(X_train, y_train)  
    
    predicted = txt_clf.predict(X_test)
    
    scores = cross_val_score(txt_clf, X_test, y_test, cv=3)
    print("Accuracy CV: %.2f%%" % (scores.mean()*100))   
    """
    vect = txt_clf.steps[0][1]    
    #Plot k Most Predictive Tokens
    nb = txt_clf.steps[2][1]
    tokens = getMostPredictiveTokens(vect, nb, X_train, y_train)
    kMostPredictiveTokens = 10
    plotMostPredictiveTokens(tokens, kMostPredictiveTokens)
    """
    #return y_test, predicted
    return txt_clf

"""MODEL EVALUATION"""
def evaluateModel(y_test, predicted):
    print("Model accuracy: %.2f%%" % (100 * metrics.accuracy_score(y_test, predicted)))

    print("Classification Report\n", metrics.classification_report(y_test, predicted))
    
    confusion_matrix = metrics.confusion_matrix(y_test, predicted)
    print("Confusion Matrix\n", confusion_matrix)
    
    plt.figure(figsize=(9,9))
    sns.heatmap(confusion_matrix, annot=True, fmt=".2f", linewidths=.5,
                square = True, cmap = 'Blues_r');
    plt.ylabel('Actual label');
    plt.xlabel('Predicted label');
    #all_sample_title = 'Accuracy Score: {0}'.format(score)
    all_sample_title = 'Confusion matrix'
    plt.title(all_sample_title, size = 15);
    

def plotBoW(df_sortedByValue, kMostFrequentWords):
    df_sortedByValue = df_sortedByValue.reset_index(drop=True)
    df_sortedByValue = df_sortedByValue.head(kMostFrequentWords)
    vocab = df_sortedByValue.loc[:, 'vocab']
    freq = df_sortedByValue.loc[:, 'freq']
    
    #Matplotlib order String arrays in alphabetical order by default
    #In order to respect the original order:
    #Use values 0,1,2,3,4,... as x and assign vocab with tick_label=
    plt.bar(range(len(vocab)), freq , tick_label=vocab)
    
    plt.xticks(rotation = 45)
    plt.xlabel('Tokens')
    plt.ylabel('Frequency')
    plt.title('%d Most frequent words' % kMostFrequentWords , fontweight='bold')
    
    plt.show()
    

 # Take a look at the words in the vocabulary
def printBoW(vect, X_train):
    # fit and transform X_train into X_train_dtm
    X_train_dtm = vect.fit_transform(X_train)
    X_train_dtm = X_train_dtm.toarray()
    vocab = vect.get_feature_names()
    #print(vocab)
    #•print the counts of each word in the vocabulary
    # Sum up the counts of each vocabulary word
    freq = np.sum(X_train_dtm, axis=0)
    # For each, print the vocabulary word and the number of times it 
    # appears in the training set
    freqList = list(freq)
    bowDict = {'vocab': vocab, 'freq': freqList}
    bowDf = pd.DataFrame(data=bowDict)
    df_sortedByValue = bowDf.sort_values("freq", ascending=0)
    #print(df_sortedByValue.head(50))
    #plotBoW(df_sortedByValue)
    return df_sortedByValue
    
def plotMostPredictiveTokens(tokens, kMostPredictiveTokens ): 
    kindle_tokens = tokens.sort_values('kindle_review_ratio', ascending=False)
    kindle_tokens = kindle_tokens.reset_index(drop=True)
    kindle_vocab = kindle_tokens.loc[:kMostPredictiveTokens-1,'token']
    kindle_ratio = kindle_tokens.loc[:kMostPredictiveTokens-1,'kindle_review_ratio']
    #print(positive_tokens_filtered.head(10))
    
    plt.bar(range(len(kindle_vocab)), kindle_ratio ,color= "yellowgreen",
            tick_label=kindle_vocab)
    
    plt.xticks(rotation = 70, size='small')
    plt.xlabel('Tokens')
    plt.ylabel('Ebooks Review Ratio')
    plt.title('%d Most predictive tokens for ebooks reviews' % kMostPredictiveTokens , fontweight='bold')
    
    plt.show()
    
    
    toys_tokens = tokens.sort_values('toys_review_ratio', ascending=False)
    toys_tokens = toys_tokens.reset_index(drop=True)
    toys_vocab = toys_tokens.loc[:kMostPredictiveTokens-1,'token']
    toys_ratio = toys_tokens.loc[:kMostPredictiveTokens-1,'toys_review_ratio']
    #print(positive_tokens_filtered.head(10))
    
    plt.bar(range(len(toys_vocab)), toys_ratio ,color= "lightskyblue",
            tick_label=toys_vocab)
    
    plt.xticks(rotation = 70, size='small')
    plt.xlabel('Tokens')
    plt.ylabel('Toys Review Ratio')
    plt.title('%d Most predictive tokens for Toys reviews' % kMostPredictiveTokens , fontweight='bold')
    
    plt.show()
    
    videogames_tokens = tokens.sort_values('videogames_review_ratio', ascending=False)
    videogames_tokens = videogames_tokens.reset_index(drop=True)
    videogames_vocab = videogames_tokens.loc[:kMostPredictiveTokens-1,'token']
    videogames_ratio = videogames_tokens.loc[:kMostPredictiveTokens-1,'videogames_review_ratio']
    #print(positive_tokens_filtered.head(10))
    
    plt.bar(range(len(videogames_vocab)), videogames_ratio ,color= "lightcoral",
            tick_label=videogames_vocab)
    
    plt.xticks(rotation = 70, size='small')
    plt.xlabel('Tokens')
    plt.ylabel('Videogames Review Ratio')
    plt.title('%d Most predictive tokens for Videogames reviews' % kMostPredictiveTokens , fontweight='bold')
    
    plt.show()
    

#Calculate which 10 tokens are the most predictive of positive reviews, 
#and which 10 tokens are the most predictive of negative reviews.
def getMostPredictiveTokens(vect, nb, X_train, y_train):
    
    # Feature Extraction
    X_train = vect.fit_transform(X_train) #fit_transform on training data
    feature_names = vect.get_feature_names()
    ch2 = SelectKBest(chi2, k=5000)
    X_train = ch2.fit_transform(X_train, y_train)
    X_train_tokens = [feature_names[i] for i in ch2.get_support(indices=True)]
    
    #X_train_tokens = vect.get_feature_names()
    #len(X_train_tokens)
    # first row is one-star reviews, second row is five-star reviews
    nb.feature_count_.shape
    
    # store the number of times each token appears across each class
    kindle_token_count = nb.feature_count_[0, :]
    toys_token_count = nb.feature_count_[1, :]
    videogames_token_count = nb.feature_count_[2, :]
    
    
    # create a DataFrame of tokens with their separate one-star and five-star counts
    #tokens = pd.DataFrame({'token':X_train_tokens, 'negative_rev':one_star_token_count, 'positive_rev':five_star_token_count}).set_index('token')
    tokens = pd.DataFrame({'token':X_train_tokens, 'kindle_rev':kindle_token_count,
                           'toys_rev':toys_token_count, 'videogames_rev':videogames_token_count})
    
    # add 1 to one-star and five-star counts to avoid dividing by 0
    tokens['kindle_rev'] = tokens.kindle_rev + 1
    tokens['toys_rev'] = tokens.toys_rev + 1
    tokens['videogames_rev'] = tokens.videogames_rev + 1
    
    
    # first number is one-star reviews, second number is five-star reviews
    nb.class_count_
    
    # convert the one-star and five-star counts into frequencies
    tokens['kindle_rev'] = tokens.kindle_rev / nb.class_count_[0]
    tokens['toys_rev'] = tokens.toys_rev / nb.class_count_[1]
    tokens['videogames_rev'] = tokens.videogames_rev / nb.class_count_[2]
    
    
    
    # calculate the ratio of five-star to one-star for each token
    tokens['kindle_review_ratio'] = round(tokens.kindle_rev / 
                                      (tokens.toys_rev + tokens.videogames_rev), 2)
    tokens['toys_review_ratio'] = round(tokens.toys_rev / 
                                      (tokens.kindle_rev + tokens.videogames_rev), 2)
    tokens['videogames_review_ratio'] = round(tokens.videogames_rev / 
                                      (tokens.toys_rev + tokens.kindle_rev), 2)
    
    return tokens
    
     
#filenames = ["kindle_reviews", "Toys_and_Games_5", "Video_Games_5" ]
#data = loadData(filenames)
data = pickleRead("dataTextClf")
cleanedData = preprocess(data)
#cleanedDataDeep = preprocessDeep(data)
#pickleWrite(cleanedDataDeep, "cleanedDataTextClfDeep")

#cleanedDataDeep = pickleRead("cleanedDataTextClfDeep")

#y_test, predicted = learnModel(cleanedData)
txt_clf = learnModel(cleanedData)

#evaluateModel(y_test, predicted)









