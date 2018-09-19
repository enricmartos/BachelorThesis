"""
Created on Thu Mar 22 17:04:07 2018

@author: Enric Martos
"""

#Dependencies for each method
import time
import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.svm import LinearSVC
from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import make_scorer
from sklearn.metrics import accuracy_score
import re
import nltk
from collections import OrderedDict
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import metrics
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from nltk.stem import SnowballStemmer
from nltk.tokenize import RegexpTokenizer
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet as wn
import pickle
from sklearn.feature_selection import SelectKBest, chi2

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

"ERROR CORRECTION"""
#Overview of the raw data
def checkData(data):
    for i in range(0, len(data)):
        print("First five entries of each Dataframe:\n", data[i].head())
        print("Check if there is some missing data:\n", data[i].info())

"""PREPROCESSING"""
"""DATA BINING"""
#Performs the Data Bining Phase for the Multiclass classification problem 2
#Input:
#[data]: Dataframe list of n datasets
#[overallSegmentSize]: Size of each overall field possible value
#Output [cleanedData]: Dataframe list of n datasets
def overallSegmentMulticlass(data, overallSegmentSize):
    negReviews1 = data[data.overall == 1]
    negReviews1Limited = negReviews1.iloc[:overallSegmentSize ,:]
    
    negReviews2 = data[data.overall == 2]
    negReviews2Limited = negReviews2.iloc[:overallSegmentSize, :]
    
    negReviews = pd.concat([negReviews1Limited, negReviews2Limited])
    
    neutReviews = data[data.overall == 3]
    neutReviewsLimited = neutReviews.iloc[:overallSegmentSize, :]
    
    totalReviews = pd.concat([negReviews, neutReviewsLimited])
    #Select and limit Positives Reviews
    posReviews4 = data[data.overall == 4]
    posReviews4Limited = posReviews4.iloc[:overallSegmentSize, :]
    
    posReviews5 = data[data.overall == 5]
    posReviews5Limited = posReviews5.iloc[:overallSegmentSize, :]

    posReviews = pd.concat([posReviews4Limited, posReviews5Limited])
    
    cleanedData = pd.concat([totalReviews, posReviews])
    #Reset indexes to iterate cleanedData Dataframe
    cleanedData = cleanedData.reset_index(drop=True)
    
    return cleanedData

#Performs the Data Bining of Overall field Phase for the Multiclass
# classification problem 1
#Input:
#[data]: Dataframe list of n datasets
#[overallSegmentSize]: Size of each overall field possible value
#Output [cleanedData]: Dataframe list of n datasets
def overallBiningNeutral(data, overallSegmentSize):
    #Select and limit Ngeative Reviews
    #data = data[data.overall != 3]
    #data['overall'] = data['overall'] >= 4
    negReviews1 = data[data.overall == 1]
    negReviews1Limited = negReviews1.iloc[:overallSegmentSize ,:]
    
    negReviews2 = data[data.overall == 2]
    negReviews2Limited = negReviews2.iloc[:overallSegmentSize, :]
    
    negReviews = pd.concat([negReviews1Limited, negReviews2Limited])
    
    neutReviews = data[data.overall == 3]
    neutReviewsLimited = neutReviews.iloc[:overallSegmentSize, :]
    
    totalReviews = pd.concat([negReviews, neutReviewsLimited])
    #Select and limit Positives Reviews
    posReviews4 = data[data.overall == 4]
    posReviews4Limited = posReviews4.iloc[:overallSegmentSize, :]
    
    posReviews5 = data[data.overall == 5]
    posReviews5Limited = posReviews5.iloc[:overallSegmentSize, :]

    posReviews = pd.concat([posReviews4Limited, posReviews5Limited])
    
    cleanedData = pd.concat([totalReviews, posReviews])
    #Reset indexes to iterate cleanedData Dataframe
    cleanedData = cleanedData.reset_index(drop=True)
    
    #Data bining assignation
        #Data bining assignation
    for i in range(0, len(cleanedData)):
        if cleanedData.loc[i,'overall'] == 1 or cleanedData.loc[i,'overall'] == 2:
            cleanedData.loc[i,'overall'] = -1
        elif cleanedData.loc[i,'overall'] == 3:
            cleanedData.loc[i,'overall'] = 0
        else:
            cleanedData.loc[i,'overall'] = 1  
    return cleanedData
    
#Performs the Data Bining of Overall field Phase for the Binary
# classification problem 1
#Input:
#[data]: Dataframe list of n datasets
#[overallSegmentSize]: Size of each overall field possible value
#Output [cleanedData]: Dataframe list of n datasets

def overallBining1and5(data, overallSegmentSize):
    #Select and limit Ngeative Reviews
    negReviews1 = data[data.overall == 1]
    negReviews = negReviews1.iloc[:overallSegmentSize ,:]
    #Select and limit Positives Reviews    
    posReviews5 = data[data.overall == 5]
    posReviews = posReviews5.iloc[:overallSegmentSize, :]
    
    cleanedData = pd.concat([negReviews, posReviews])
    #Reset indexes to iterate cleanedData Dataframe
    cleanedData = cleanedData.reset_index(drop=True)
    
    #Data bining assignation
    cleanedData['overall'] = cleanedData['overall'] == 5
    return cleanedData

#Performs the Data Bining of Overall field Phase for the Binary
# classification problem 2
#Input:
#[data]: Dataframe list of n datasets
#[overallSegmentSize]: Size of each overall field possible value
#Output [cleanedData]: Dataframe list of n datasets
def overallBining(data, overallSegmentSize):
    #Select and limit Ngeative Reviews
    #data = data[data.overall != 3]
    #data['overall'] = data['overall'] >= 4
    negReviews1 = data[data.overall == 1]
    negReviews1Limited = negReviews1.iloc[:overallSegmentSize ,:]
    
    negReviews2 = data[data.overall == 2]
    negReviews2Limited = negReviews2.iloc[:overallSegmentSize, :]
    
    negReviews = pd.concat([negReviews1Limited, negReviews2Limited])
    #Select and limit Positives Reviews
    posReviews4 = data[data.overall == 4]
    posReviews4Limited = posReviews4.iloc[:overallSegmentSize, :]
    
    posReviews5 = data[data.overall == 5]
    posReviews5Limited = posReviews5.iloc[:overallSegmentSize, :]

    posReviews = pd.concat([posReviews4Limited, posReviews5Limited])
    
    cleanedData = pd.concat([negReviews, posReviews])
    #Reset indexes to iterate cleanedData Dataframe
    cleanedData = cleanedData.reset_index(drop=True)
    
    #Data bining assignation
    cleanedData['overall'] = cleanedData['overall'] >= 4
    return cleanedData

#Performs the Data Bining phase of the Overall Field but without the overallSegmentSize
#parameter. Then, the unbalanced raw dataset is preserved.
def overallBiningUnbalanced(data):    
    #Data bining assignation
    negReviews1 = data[data.overall == 1]
    
    negReviews2 = data[data.overall == 2]
    
    negReviews = pd.concat([negReviews1, negReviews2])
    #Select and limit Positives Reviews
    posReviews4 = data[data.overall == 4]
    
    posReviews5 = data[data.overall == 5]

    posReviews = pd.concat([posReviews4, posReviews5])
    
    cleanedData = pd.concat([negReviews, posReviews])
    #Reset indexes to iterate cleanedData Dataframe
    cleanedData = cleanedData.reset_index(drop=True)
    
    #Data bining assignation
    cleanedData['overall'] = cleanedData['overall'] >= 4
    return cleanedData


#Clean data
#Input [data]: Dataframe list of n datasets
#Output [cleanedData]: Preprocessed Dataframe list of n datasets
def preprocess(data):
    cleanedData = []
    for i in range(0, len(data)):
        #Filter Data
        filteredData = data[i].loc[:, ["asin", "reviewText", "overall",]]
        # 2. Remove non-letters
        
        # examine overall class distribution
        overallDistribution = filteredData.overall.value_counts().sort_index()
        print(overallDistribution)
    
        # Perform the data bining of the overall field for all four
        #classification problems
        #overallSegmentSize = overallDistribution.min(axis = 0)
        #newCleanedData = overallBining(filteredData, overallSegmentSize)
        #newCleanedData = overallBining1and5(filteredData, overallSegmentSize)
        #newCleanedData = overallSegmentMulticlass(filteredData, overallSegmentSize)
        #newCleanedData = overallBiningNeutral(filteredData, overallSegmentSize)
        #cleanedData.append(newCleanedData)
    return cleanedData

"""NLP TRANSFORMATIONS"""
def getGlobalNLTKVars():
    #Global NLTK Variables
    ret = RegexpTokenizer('[a-zA-Z0-9\']+')
    sw = set(stopwords.words('english'))
    #lemmaTokenizer
    wnl = WordNetLemmatizer()
    #stemTokenizer
    ess = SnowballStemmer('english')
    #POSTokenizer
    allowed_word_types = ["ADJ", "ADV", "NOUN", "VERB"]
    return ret, sw, wnl, ess, allowed_word_types

#Stemmer Tokenizer
def stemTokenizer(sentence):
    tokens= ret.tokenize(sentence)
    #return [ess.stem(t) for t in tokens if t not in sw]
    return [ess.stem(t) for t in tokens]

#return WORDNET POS compliance to WORDENT lemmatization (a,n,r,v)
def get_wordnet_pos(pos):
    pos_wn = [] 
    for i in range(0, len(pos)):
        if pos[i][1] == 'ADJ':
            pos_wn.append(wn.ADJ)
        elif pos[i][1] == 'VERB':
            pos_wn.append(wn.VERB)
        elif pos[i][1] == 'ADV':
            pos_wn.append(wn.ADV)
        else:
            # As default pos in lemmatization is Noun
            pos_wn.append(wn.NOUN)
    return pos_wn

#Part of Speech Tokenizer
def POSTokenizer(sentence):
    tokens= ret.tokenize(sentence)
    pos = nltk.pos_tag(tokens, tagset = "universal")
    return [wnl.lemmatize(pos[i][0] ) for i in range(0, len(tokens)) 
            if pos[i][1] in allowed_word_types]

#Lemmatization Tokenizer
def lemmaTokenizer(sentence):
    tokens= ret.tokenize(sentence)
    pos = nltk.pos_tag(tokens, tagset = "universal")
    pos_wn = get_wordnet_pos(pos)
    return [wnl.lemmatize(pos[i][0], pos_wn[i]) for i in range(0, len(tokens))]


    

   

 #Building of a Pipeline
def buildPipeline(pipelineId):
    if pipelineId == 1:
        #Pipeline 1
        
        text_clf = Pipeline([
                            ('vect', CountVectorizer(
                                                     max_features = 5000)),
                          ('clf', MultinomialNB()),])
    elif pipelineId == 2:
        #Pipeline 2
        text_clf = Pipeline([('vect', CountVectorizer(
                                                     max_features = 5000)),
                          ('clf', LogisticRegression()),])
    elif pipelineId == 3:
        #Pipeline 3
        text_clf = Pipeline([('vect', CountVectorizer(
                                                     max_features = 5000)),
                          ('clf', SGDClassifier()),])
    elif pipelineId == 4:
        #Pipeline 3
        text_clf = Pipeline([('vect', CountVectorizer(
                                                     max_features = 5000)),
                          ('clf', RandomForestClassifier()),])
    elif pipelineId == 5:
        #Pipeline 5
        text_clf = Pipeline([('vect', CountVectorizer(
                                    max_features = 5000, ngram_range=(1, 2),
                                    binary = True, min_df=2, max_df=0.8)),
                            ('tfidf', TfidfTransformer()),
                            ('clf', MultinomialNB()),])
    elif pipelineId == 6:
        #Pipeline 5
        text_clf = Pipeline([('vect', CountVectorizer(tokenizer = lemmaTokenizer,
                                    max_features = 5000, ngram_range=(1, 2),
                                    binary = True, min_df=2, max_df=0.8)),
                            ('tfidf', TfidfTransformer()),
                            ('clf', DecisionTreeClassifier()),])
    elif pipelineId == 7:
        #Pipeline 5
        text_clf = Pipeline([('vect', CountVectorizer(max_features = 5000)),
                            ('clf', RandomForestClassifier())])
    elif pipelineId == 8:
        #Pipeline 5
        text_clf = Pipeline([('vect', TfidfVectorizer(tokenizer = stemTokenizer,
                                    max_features = 5000, ngram_range=(1, 2),
                                    binary = True, min_df=2, max_df=0.8)),
                            ('clf',  RandomForestClassifier()),])
    elif pipelineId == 9:
        #Pipeline 5
        text_clf = Pipeline([('vect', TfidfVectorizer(tokenizer = stemTokenizer,
                                    max_features = 5000, ngram_range=(1, 2),
                                    binary = True, min_df=2, max_df=0.8)),
                            ('clf',  LogisticRegression()),])
    elif pipelineId == 10:
        #Pipeline 5
        text_clf = Pipeline([('vect', TfidfVectorizer(tokenizer = stemTokenizer,
                                    max_features = 5000, ngram_range=(1, 2),
                                    binary = True, min_df=2, max_df=0.8)),
                            ('clf',  SGDClassifier()),])
    elif pipelineId == 11:
        #Pipeline 5
        text_clf = Pipeline([('vect', TfidfVectorizer(tokenizer = stemTokenizer,
                                    ngram_range=(1, 2),
                                    binary = True, min_df=2, max_df=0.8)),
                     ('chi2', SelectKBest(chi2, k=5000)),
                     ('clf', MultinomialNB(fit_prior=False))])
    elif pipelineId == 12:
        #Pipeline 5
        text_clf = Pipeline([('vect', TfidfVectorizer(tokenizer = stemTokenizer,
                                    ngram_range=(1, 2), 
                                    binary = True, min_df=2, max_df=0.8)),
                     ('chi2', SelectKBest(chi2, k=5000)),
                     ('clf', LogisticRegression())])
    elif pipelineId == 13:
        #Pipeline 5
        text_clf = Pipeline([('vect', TfidfVectorizer(tokenizer = stemTokenizer,
                                    ngram_range=(1, 2),
                                    binary = True, min_df=2, max_df=0.8)),
                     ('chi2', SelectKBest(chi2, k=5000)),
                     ('clf', SGDClassifier(n_iter= 10, penalty='elasticnet'))])
    elif pipelineId == 14:
        #Pipeline 5
        text_clf = Pipeline([('vect', TfidfVectorizer(tokenizer = stemTokenizer,
                                    ngram_range=(1, 2), 
                                    binary = True, min_df=2, max_df=0.8)),
                     ('chi2', SelectKBest(chi2, k=5000)),
                     ('clf', RandomForestClassifier(n_estimators=100))])
    else:
        #Pipeline 4
        text_clf = Pipeline([('vect', CountVectorizer(tokenizer = lemmaTokenizer,
                                    max_features = 5000, ngram_range=(1, 2),
                                    binary = True, min_df=2, max_df=0.8)),
                            ('tfidf', TfidfTransformer()),
                            ('clf',  MultinomialNB()),])
    
    return text_clf


"""LEARNING ALGORITHM TRAINING"""

#Parameter tuning using grid search
def checkFirstApproach(cleanedData):
    X = cleanedData.reviewText
    y = cleanedData.overall 

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.4, 
                                                        random_state=1)
    parameters = {'vect__tokenizer': (lemmaTokenizer, None)}
    
    pipelineId = 1
    text_clf = buildPipeline(pipelineId)
    
    gs_clf = GridSearchCV(text_clf, parameters, n_jobs=1)
    gs_clf = gs_clf.fit(X_train, y_train)
    print("Best score:  %.2f%%" % float(100*gs_clf.best_score_))
    #print("Model accuracy: %.2f%%" % (100 * metrics.accuracy_score(y_test, y_pred_class)))
    
    print("Best params:\n")
    for param_name in sorted(parameters.keys()):
                 print("%s: %r" % (param_name, gs_clf.best_params_[param_name]))
    
    print("Mean test score of all params:\n")
    results = pd.DataFrame.from_dict(gs_clf.cv_results_)
    print(results.loc[:, ('params','mean_test_score')])
    
    #The pipeline already does the transformation using CountVectorizer
    X_test = X_test.reset_index(drop=True)
    X_test = X_test[:300]
    y_test = y_test.reset_index(drop=True)
    y_test = y_test[:300]
    y_pred = gs_clf.best_estimator_.predict(X_test)
    y_pred = pd.Series(data = y_pred)
    
    vect = gs_clf.estimator.steps[0][1]
    
    printBoW(vect, X_train)
    #printClfReport(y_test, y_pred)
    #printConfusionMatrix(y_test, y_pred)
    #getFPandFN(X_test, y_test, y_pred)
    
#Train the model using only one specific dataset
def learnModel(cleanedData, datasetId, pipelineId):
    X = cleanedData[datasetId].reviewText
    y = cleanedData[datasetId].overall 

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.4, 
                                                        random_state=1)
    
    
    txt_clf = buildPipeline(pipelineId)
    txt_clf.fit(X_train, y_train)
    
    scores = cross_val_score(txt_clf, X_test, y_test, cv=3)
    print("Accuracy CV: %.2f%%" % (scores.mean()*100))    
    
    # calculate null accuracy
    nullAcucracy = y_test.value_counts().head(1) / y_test.shape
    print("Null Accuracy CV: %.2f%%" % (nullAcucracy*100)) 
    
    #vect = txt_clf.steps[0][1]
    #df_sortedByValue = printBoW(vect, X_train)
    #return df_sortedByValue
    #return  text_clf, X_test, y_test, y_pred_class

#Train the model using all the datasets
def learnModelAll(cleanedData, pipelineId):
    X = cleanedData.reviewText
    y = cleanedData.overall 

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.4, 
                                                        random_state=1)
    
    txt_clf = buildPipeline(pipelineId)
    txt_clf.fit(X_train, y_train)
    
    scores = cross_val_score(txt_clf, X_test, y_test, cv=3)
    print("Accuracy CV: %.2f%%" % (scores.mean()*100))  
    
    # calculate null accuracy
    nullAcucracy = y_test.value_counts().head(1) / y_test.shape
    print("Null Accuracy CV: %.2f%%" % (nullAcucracy*100))

    return txt_clf
    

"""MODEL EVALUATION"""

def computeExecutionTime(start_time, end_time):
    elapsed_sec = end_time - start_time
    elapsed_min = elapsed_sec/60
    if elapsed_min > 0:
        elapsed_sec = elapsed_sec % 60
    print("Execution time: %.0f min : %0.f sec" % (elapsed_min, elapsed_sec))

def printClfReport(y_test, y_pred_class):
    print("Classification Report\n",metrics.classification_report(y_test, y_pred_class))

# print the confusion matrix
def printConfusionMatrix(y_test, y_pred_class):
    print("Confusion matrix\n", metrics.confusion_matrix(y_test, y_pred_class))
    
def getFPandFN(X_test, y_test, y_pred_class):
    # first 10 false positives
    #(Negative reviews incorrectly classified as Positive reviews)
    FP = X_test[y_test < y_pred_class]
    # first 10 false negatives
    #(Positives reviews incorrectly classified as Negatives reviews)
    FN = X_test[y_test > y_pred_class]
    print("False Positive Reviews: \n")
    for i in range(0, 10):
        print(FP.iloc[i])
        print("\n\n")
    print("False Negative Reviews: \n")
    for i in range(0, 10):
        print(FN.iloc[i])
        print("\n\n")
    return FP, FN

def plotROCCurve(y_test, y_pred_class):
    fpr, tpr, thresholds = metrics.roc_curve(y_test, y_pred_class)
    
    roc_auc = metrics.roc_auc_score(y_test, y_pred_class)

    #Print ROC Curve
    plt.plot(fpr, tpr, color='darkorange',
              label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic example')
    plt.legend(loc="lower right")   
    plt.show()

    
#Compute accuracy and other evalution parameters of the model
#Input
#[y_test]: Correct output of the test set
#[y_pred_class]: Trained classifer predicitions on the test
#Output: Print evaluation results
def evaluateModel(X_test, y_test, y_pred_class):
    # calculate accuracy of class predictions
    print("Model accuracy: %.2f%%" % (100 * metrics.accuracy_score(y_test, y_pred_class)))
    
    printClfReport(y_test, y_pred_class)
    
    # print the confusion matrix
    printConfusionMatrix(y_test, y_pred_class)
    
    plotROCCurve(y_test, y_pred_class)
    
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
    # sort the DataFrame by five_star_ratio (descending order), and examine the first 10 rows
    # note: use sort() instead of sort_values() for pandas 0.16.2 and earlier
    
    positive_tokens = tokens.sort_values('positive_review_ratio', ascending=False)
    positive_tokens = positive_tokens.reset_index(drop=True)
    positive_vocab = positive_tokens.loc[:kMostPredictiveTokens-1,'token']
    positive_ratio = positive_tokens.loc[:kMostPredictiveTokens-1,'positive_review_ratio']
    #print(positive_tokens_filtered.head(10))
    
    plt.bar(range(len(positive_vocab)), positive_ratio ,color= "yellowgreen",
            tick_label=positive_vocab)
    
    plt.xticks(rotation = 70, size='small')
    plt.xlabel('Tokens')
    plt.ylabel('Positive Review Ratio')
    plt.title('%d Most predictive tokens for positive reviews' % kMostPredictiveTokens , fontweight='bold')
    
    plt.show()

    
    # sort the DataFrame by five_star_ratio (ascending order), and examine the first 10 rows
    negative_tokens = tokens.sort_values('negative_review_ratio', ascending=False)
    negative_tokens = negative_tokens.reset_index(drop=True)
    negative_vocab = negative_tokens.loc[:kMostPredictiveTokens-1,'token']
    negative_ratio = negative_tokens.loc[:kMostPredictiveTokens-1,'negative_review_ratio']  
    #print(negative_tokens_filtered.head(10))
    
    plt.bar(range(0, len(negative_vocab)), negative_ratio , color="lightcoral",
            tick_label=negative_vocab)
    
    plt.xticks(rotation = 70, size='small')
    plt.xlabel('Tokens')
    plt.ylabel('Negative Review Ratio')
    plt.title('%d Most predictive tokens for negative reviews' % kMostPredictiveTokens , fontweight='bold')
    
    plt.show()
    

#Calculate which 10 tokens are the most predictive of positive reviews, 
#and which 10 tokens are the most predictive of negative reviews.
def getMostPredictiveTokens(vect, nb):
    
    # store the vocabulary of X_train
    X_train_tokens = vect.get_feature_names()
    #len(X_train_tokens)
    # first row is one-star reviews, second row is five-star reviews
    nb.feature_count_.shape
    
    # store the number of times each token appears across each class
    one_star_token_count = nb.feature_count_[0, :]
    five_star_token_count = nb.feature_count_[1, :]
    
    # create a DataFrame of tokens with their separate one-star and five-star counts
    #tokens = pd.DataFrame({'token':X_train_tokens, 'negative_rev':one_star_token_count, 'positive_rev':five_star_token_count}).set_index('token')
    tokens = pd.DataFrame({'token':X_train_tokens, 'negative_rev':one_star_token_count, 'positive_rev':five_star_token_count})
    
    # add 1 to one-star and five-star counts to avoid dividing by 0
    tokens['negative_rev'] = tokens.negative_rev + 1
    tokens['positive_rev'] = tokens.positive_rev + 1
    
    # first number is one-star reviews, second number is five-star reviews
    nb.class_count_
    
    # convert the one-star and five-star counts into frequencies
    tokens['negative_rev'] = tokens.negative_rev / nb.class_count_[0]
    tokens['positive_rev'] = tokens.positive_rev / nb.class_count_[1]
    
    # calculate the ratio of five-star to one-star for each token
    tokens['positive_review_ratio'] = round(tokens.positive_rev / tokens.negative_rev, 2)
    
    tokens['negative_review_ratio'] = round(tokens.negative_rev / tokens.positive_rev, 2)
    
    return tokens

def BoWSwitchExperiment(cleanedData, BoWId, targetDatasetId, pipelineId):
    X_BoW = cleanedData[BoWId].reviewText
    y_BoW = cleanedData[BoWId].overall 

    X_train_BoW, X_test_BoW, y_train_BoW, y_test_BoW = train_test_split(X_BoW,
                                                    y_BoW, test_size = 0.4, random_state=1)
    
    txt_clf = buildPipeline(pipelineId)
    txt_clf = txt_clf.fit(X_train_BoW, y_train_BoW)
    
    X = cleanedData[targetDatasetId].reviewText
    y = cleanedData[targetDatasetId].overall 
    
    X_train, X_test, y_train, y_test = train_test_split(X,
                                                    y, test_size = 0.4, random_state=1)
    pred = txt_clf.predict(X_test)
    
    accuracy = metrics.accuracy_score(y_test, pred) 
    print("Accuracy: %.2f%%\n" % (accuracy*100))   
    

"""HYPERPARAMETER OPTIMIZATION"""
#Parameter tuning using grid search
def tuneParamsAndEvaluate(cleanedData):
    #Preprocessing hyperparameter tuning
    X = cleanedData.reviewText
    y = cleanedData.overall 
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.4, 
                                                random_state=1)
    parameters = { 'vect__stop_words': ("english", None),
                    'vect__ngram_range': [(1, 1), (1, 2)],
                    'vect__min_df': (1, 2, 3),
                    'vect__max_df': (0.8, 0.9, 1.0),
                    'vect__binary': (True, False),
                   'tfidf__use_idf': (True, False)
    }
    
    parameters = {'vect__ngram_range': [(1, 1), (1, 2)],
                   'tfidf__use_idf': (True, False),
                   'clf__alpha': (1e-2, 1e-3),
    }
    
    #Classifier hyperparameter tuning
    #Random Forest Clf
    parameters = {'clf__n_estimators': (10, 100, 500),
                  'clf__max_features': ('auto', 0.2)
    }
    
    #Logisitc Reg Clf
    parameters = {'clf__C':  [0.01, 0.1, 1, 10, 100],
                  'clf__penalty': ('l1', 'l2')
    }
    #SGD
    #Logisitc Reg Clf
    parameters = {'clf__C':  [0.01, 0.1, 1, 10, 100],
                  'clf__penalty': ('l1', 'l2')
    }
    #SGD
    parameters = {'clf__alpha': (1e-3, 1e-4, 1e-5),
                  'clf__penalty': ('l2', 'elasticnet'),
                  'clf__n_iter': (None, 10, 100, 1000)
    }
    
    pipelineId = 10
    text_clf = buildPipeline(pipelineId)
    
    gs_clf = GridSearchCV(text_clf, parameters, n_jobs=1)
    gs_clf = gs_clf.fit(X_train, y_train)
    print("Best score: ", gs_clf.best_score_)
    for param_name in sorted(parameters.keys()):
                 print("%s: %r" % (param_name, gs_clf.best_params_[param_name]))
   
    #evaluateModel(X_test, y_test, y_pred_class)
    #y_pred = gs_clf.predict(X_test)
    
    #print(np.mean(y_pred == y_test))
    
    #print(metrics.classification_report(y_test, y_pred))
    
    
    

#start_time = time.clock();
#filenames = ["kindle_reviews", "Toys_and_Games_5", "Video_Games_5" ]
#data = loadData(filenames)

#SET YOUR DIRECTORY IN ORDER TO PROPERLY READ THE PICKLED VARIABLES.
#For instance
#directory: cd "Documents/Python Spyder Projects"
#data = pickleRead("data")

#Check if there are missing fields all over the three datasets
#checkData(data)

#Unbalanceded Data 
#data[0], data[1], data[2]
#checkFirstApproach(data[0])
#Balanced Data
#cleanedData = preprocess(data)
#Binary Classification 2 Problem
#cleanedData = pickleRead("cleanedDataOverallBinary")
#Binary Classification 1 Problem
#cleanedData = pickleRead("cleanedDataOverallBinary1and5")
#Multi class Classification 2 Problem
#cleanedData = pickleRead("cleanedDataOverallMulticlass5")
#Multi class Classification 1 Problem
#cleanedData = pickleRead("cleanedDataOverallMulticlassNeutral")
#cleanedData[0], cleanedData[1], cleanedData[2]

#allReviews = pickleRead("cleanedDataOverallBinaryAll")


#allReviewsMulti = pickleRead("cleanedDataOverallMulticlassNeutralAll")

#allReviews = pickleRead("cleanedDataOverallMulticlassNeutralAll")

#checkFirstApproach(cleanedData[0])

#Global NLTK Variables
ret, sw, wnl, ess, allowed_word_types = getGlobalNLTKVars()

#text_clf, X_test, y_test, y_pred_class = learnModel(cleanedData[2])
#Experiment 1 - No feature Bining- Bias to Positive
#text_clf, X_test, y_test, y_pred_class = learnModel(cleanedData[1])

#FINAL MODEL ACCURACY RESULTS
"""
pipelineId = 11 -> MNB | pipelineId = 12 -> LogReg
pipelineId = 13 -> SGD | pipelineId = 14 -> RF
"""
#datasetId = 0
#pipelineId = 11
#learnModel(cleanedData, datasetId, pipelineId)
#learnModel(cleanedDataMulti, datasetId, pipelineId)
#txt_clf = learnModelAll(allReviews, pipelineId)

#Test reviews
#rev1 = "This is great"
#rev2 = "I like this book"
#rev3 = "My kid love this game"
#rev4 = "I totally hate this book"
#rev5 = "This is disappointing"
#rev6 = "Shame on this game"

#testReviewList = [rev1, rev2, rev3, rev4, rev5, rev6]

#BoW Experiment Test
#BoWId = 2
#targetDatasetId = 1
#pipelineId = 13
#BoWSwitchExperiment(cleanedData, BoWId, targetDatasetId, pipelineId)

#evaluateModel(X_test, y_test, y_pred_class)
#FP, FN = getFPandFN(X_test, y_test, y_pred_class)


#end_time = time.clock();
#computeExecutionTime(start_time, end_time)
  
"""TEST PROGRAM LOGIC"""  
#Test Program Flow: Interaction with user input

inputClfTask = 0
while (inputClfTask != 'E'):
    inputClfTask = input(' \n \
                     Choose a classification task (Press E to close the program):\n \n \
                     1. Overall Sentiment of a review\n \
                     2. Helpfulness of a review  \n \
                     3. Topic categorization of a review \n')
    
    if inputClfTask == '1':
        print('You have chosen the "Overall Sentiment of a review" classification task.')
        
        inputClfTaskType = 0
        while (inputClfTaskType != "B"):
            inputClfTaskType = input(' \n \
                         Choose a classification problem type (Press B to go to the previous step):\n \n \
                         1. Binary Classification (Positive or Negative) \n \
                         2. Multiclass Classification (Positive, Negative or Neutral)  \n ')
            if inputClfTaskType == '1':
                print('You have chosen the "Binary Classification" task.')
                
                inputClfOverall = 0
                
                while (inputClfOverall != "B"):
                    inputClfOverall = input(' \n \
                                 Choose your classifier (Press B to go to the previous step):\n \n \
                                 1. Random Forest (RF) \n \
                                 2. Multinomial Naive Bayes (MNB)  \n \
                                 3. Support Vector Machines (SVM) \n \
                                 4. Logistic Regression (LR) \n ')
                    
                    if inputClfOverall == '1':
                        print('You have chosen the RF classifier')
                        #Pickle read 
                        txtClfRF = pickleRead('RFClfOverallBinary')
                        
                        inputReview = 0
                        while (inputReview != "B"):
                            inputReview = input('\n \
                                                Introduce your own review and check the prediction of the classifier (Press B to go to the previous step): \n ')
                            #Convert str to list
                            testReview = [inputReview]
                            result = txtClfRF.predict(testReview)

                            if result[0] == True:
                                print('This review has been classified as positive.')
                            else:
                                print('This review has been classified as negative.')
                        
                    elif inputClfOverall == '2':
                        print('You have chosen the MNB classifier')
                        
                        #Pickle read 
                        txtClfMNB = pickleRead('MNBClfOverallBinary')
                        
                        inputReview = 0
                        while (inputReview != "B"):
                            inputReview = input('\n \
                                                Introduce your own review and check the prediction of the classifier (Press B to go to the previous step): \n ')
                            #Convert str to list
                            testReview = [inputReview]
                            result = txtClfMNB.predict(testReview)
                            
                            if result[0] == True:
                                print('This review has been classified as positive.')
                            else:
                                print('This review has been classified as negative.')
                            
                    elif inputClfOverall == '3':
                        print('You have chosen the SVM classifier')
                        
                        #Pickle read 
                        txtClfSVM = pickleRead('SVMClfOverallBinary')
                        
                        inputReview = 0
                        while (inputReview != "B"):
                            inputReview = input('\n \
                                                Introduce your own review and check the prediction of the classifier (Press B to go to the previous step): \n ')
                            #Convert str to list
                            testReview = [inputReview]
                            result = txtClfSVM.predict(testReview)
                            
                            if result[0] == True:
                                print('This review has been classified as positive.')
                            else:
                                print('This review has been classified as negative.')
                    elif inputClfOverall == '4':
                        print('You have chosen the LR classifier')
                        
                        #Pickle read 
                        txtClfLR = pickleRead('LRClfOverallBinary')
                        
                        inputReview = 0
                        while (inputReview != "B"):
                            inputReview = input('\n \
                                                Introduce your own review and check the prediction of the classifier (Press B to go to the previous step): \n ')
                            #Convert str to list
                            testReview = [inputReview]
                            result = txtClfLR.predict(testReview)
                            
                            if result[0] == True:
                                print('This review has been classified as positive.')
                            else:
                                print('This review has been classified as negative.')
                    elif inputClfOverall == "B":
                        print('Back to the previous step')
                    elif inputClfOverall != 'B':
                        print('Please choose a valid option')
                
            elif inputClfTaskType == '2':
                print('You have chosen the "Multiclass Classification" task.')
                
                inputClfOverall = 0
                
                while (inputClfOverall != "B"):
                    inputClfOverall = input(' \n \
                                 Choose your classifier (Press B to go to the previous step):\n \n \
                                 1. Random Forest (RF) \n \
                                 2. Multinomial Naive Bayes (MNB)  \n \
                                 3. Support Vector Machines (SVM) \n \
                                 4. Logistic Regression (LR) \n ')
                    
                    if inputClfOverall == '1':
                        print('You have chosen the RF classifier')
                        #Pickle read 
                        txtClfRF = pickleRead('RFClfOverallMulti')
                        
                        inputReview = 0
                        while (inputReview != "B"):
                            inputReview = input('\n \
                                                Introduce your own review and check the prediction of the classifier (Press B to go to the previous step): \n ')
                            #Convert str to list
                            testReview = [inputReview]
                            result = txtClfRF.predict(testReview)

                            if result[0] == 1:
                                print('This review has been classified as positive.')
                            elif result[0] == -1:
                                print('This review has been classified as negative.')
                            else:
                                print('This review has been classified as neutral.')
                        
                    elif inputClfOverall == '2':
                        print('You have chosen the MNB classifier')
                        
                        #Pickle read 
                        txtClfMNB = pickleRead('MNBClfOverallMulti')
                        
                        inputReview = 0
                        while (inputReview != "B"):
                            inputReview = input('\n \
                                                Introduce your own review and check the prediction of the classifier (Press B to go to the previous step): \n ')
                            #Convert str to list
                            testReview = [inputReview]
                            result = txtClfMNB.predict(testReview)
                            
                            if result[0] == 1:
                                print('This review has been classified as positive.')
                            elif result[0] == -1:
                                print('This review has been classified as negative.')
                            else:
                                print('This review has been classified as neutral.')
                            
                    elif inputClfOverall == '3':
                        print('You have chosen the SVM classifier')
                        
                        #Pickle read 
                        txtClfSVM = pickleRead('SVMClfOverallMulti')
                        
                        inputReview = 0
                        while (inputReview != "B"):
                            inputReview = input('\n \
                                                Introduce your own review and check the prediction of the classifier (Press B to go to the previous step): \n ')
                            #Convert str to list
                            testReview = [inputReview]
                            result = txtClfSVM.predict(testReview)
                            
                            if result[0] == 1:
                                print('This review has been classified as positive.')
                            elif result[0] == -1:
                                print('This review has been classified as negative.')
                            else:
                                print('This review has been classified as neutral.')
                    elif inputClfOverall == '4':
                        print('You have chosen the LR classifier')
                        
                        #Pickle read 
                        txtClfLR = pickleRead('LRClfOverallMulti')
                        
                        inputReview = 0
                        while (inputReview != "B"):
                            inputReview = input('\n \
                                                Introduce your own review and check the prediction of the classifier (Press B to go to the previous step): \n ')
                            #Convert str to list
                            testReview = [inputReview]
                            result = txtClfLR.predict(testReview)
                            
                            if result[0] == 1:
                                print('This review has been classified as positive.')
                            elif result[0] == -1:
                                print('This review has been classified as negative.')
                            else:
                                print('This review has been classified as neutral.')
                    elif inputClfOverall == "B":
                        print('Back to the previous step')
                    elif inputClfOverall != 'B':
                        print('Please choose a valid option')
                        
            elif inputClfTaskType == "B":
                print('Back to the previous step')
            elif inputClfTaskType != 'B':
                print('Please choose a valid option')
            
            
        
    elif inputClfTask == '2':
        print('You have chosen the "Helpfulness of a review" classification task.')
        
        inputClfHelpful = 0
        while (inputClfHelpful != "B"):
            inputClfHelpful = input(' \n \
                         Choose your classifier (Press B to go to the previous step):\n \n \
                         1. Random Forest (RF) \n \
                         2. Multinomial Naive Bayes (MNB)  \n \
                         3. Support Vector Machines (SVM) \n \
                         4. Logistic Regression (LR) \n ')
            
            if inputClfHelpful == '1':
                print('You have chosen the RF classifier')
                #Pickle read 
                txtClfRF = pickleRead('RFClfHelpful')
                
                inputReview = 0
                while (inputReview != "B"):
                    inputReview = input('\n \
                                        Introduce your own review and check the prediction of the classifier (Press B to go to the previous step): \n ')
                    #Convert str to list
                    testReview = [inputReview]
                    result = txtClfRF.predict(testReview)

                    if result[0] == True:
                        print('This review has been classified as helpful.')
                    else:
                        print('This review has been classified as unhelpful.')
                
            elif inputClfHelpful == '2':
                print('You have chosen the MNB classifier')
                
                #Pickle read 
                txtClfMNB = pickleRead('MNBClfHelpful')
                
                inputReview = 0
                while (inputReview != "B"):
                    inputReview = input('\n \
                                        Introduce your own review and check the prediction of the classifier (Press B to go to the previous step): \n ')
                    #Convert str to list
                    testReview = [inputReview]
                    result = txtClfMNB.predict(testReview)
                    
                    if result[0] == True:
                        print('This review has been classified as helpful.')
                    else:
                        print('This review has been classified as unhelpful.')
                    
            elif inputClfHelpful == '3':
                print('You have chosen the SVM classifier')
                
                #Pickle read 
                txtClfSVM = pickleRead('SVMClfHelpful')
                
                inputReview = 0
                while (inputReview != "B"):
                    inputReview = input('\n \
                                        Introduce your own review and check the prediction of the classifier (Press B to go to the previous step): \n ')
                    #Convert str to list
                    testReview = [inputReview]
                    result = txtClfSVM.predict(testReview)
                    
                    if result[0] == True:
                        print('This review has been classified as helpful.')
                    else:
                        print('This review has been classified as unhelpful.')
            elif inputClfHelpful == '4':
                print('You have chosen the LR classifier')
                
                #Pickle read 
                txtClfLR = pickleRead('LRClfHelpful')
                
                inputReview = 0
                while (inputReview != "B"):
                    inputReview = input('\n \
                                        Introduce your own review and check the prediction of the classifier (Press B to go to the previous step): \n ')
                    #Convert str to list
                    testReview = [inputReview]
                    result = txtClfLR.predict(testReview)
                    
                    if result[0] == True:
                        print('This review has been classified as helpful.')
                    else:
                        print('This review has been classified as unhelpful.')
            elif inputClfHelpful == "B":
                print('Back to the previous step')
            elif inputClfHelpful != 'B':
                print('Please choose a valid option')
                        
    elif inputClfTask == '3':
        print('You have chosen the "Topic categorization of a review" classification task.')
        
        inputClfCat = 0
        while (inputClfCat != "B"):
            inputClfCat = input(' \n \
                         Choose your classifier (Press B to go to the previous step):\n \n \
                         1. Random Forest (RF) \n \
                         2. Multinomial Naive Bayes (MNB) \n \
                         3. Support Vector Machines (SVM) \n \
                         4. Logistic Regression (LR) \n ')
            
            if inputClfCat == '1':
                print('You have chosen the RF classifier')
                #Pickle read 
                txtClfRF = pickleRead('RFClfCat')
                
                inputReview = 0
                while (inputReview != "B"):
                    inputReview = input('\n \
                                        Introduce your own review and check the prediction of the classifier (Press B to go to the previous step): \n ')
                    #Convert str to list
                    testReview = [inputReview]
                    result = txtClfRF.predict(testReview)

                    if result[0] == 0:
                        print('This predicted topic of this review is "Kindle ebooks"')
                    elif result[0] == 1:
                        print('This predicted topic of this review is "Toys"')
                    else: 
                        print('This predicted topic of this review is "Videogames"')
                        
            elif inputClfCat == '2':
                print('You have chosen the MNB classifier')
                
                #Pickle read 
                txtClfMNB = pickleRead('MNBClfCat')
                
                inputReview = 0
                while (inputReview != "B"):
                    inputReview = input('\n \
                                        Introduce your own review and check the prediction of the classifier (Press B to go to the previous step): \n ')
                    #Convert str to list
                    testReview = [inputReview]
                    result = txtClfMNB.predict(testReview)
                    
                    if result[0] == 0:
                        print('This predicted topic of this review is "Kindle ebooks"')
                    elif result[0] == 1:
                        print('This predicted topic of this review is "Toys"')
                    else: 
                        print('This predicted topic of this review is "Videogames"')
                    
            elif inputClfCat == '3':
                print('You have chosen the SVM classifier')
                
                #Pickle read 
                txtClfSVM = pickleRead('SVMClfCat')
                
                inputReview = 0
                while (inputReview != "B"):
                    inputReview = input('\n \
                                        Introduce your own review and check the prediction of the classifier (Press B to go to the previous step): \n ')
                    #Convert str to list
                    testReview = [inputReview]
                    result = txtClfSVM.predict(testReview)
                    
                    if result[0] == 0:
                        print('This predicted topic of this review is "Kindle ebooks"')
                    elif result[0] == 1:
                        print('This predicted topic of this review is "Toys"')
                    else: 
                        print('This predicted topic of this review is "Videogames"')
            elif inputClfCat == '4':
                print('You have chosen the LR classifier')
                
                #Pickle read 
                txtClfLR = pickleRead('LRClfCat')
                
                inputReview = 0
                while (inputReview != "B"):
                    inputReview = input('\n \
                                        Introduce your own review and check the prediction of the classifier (Press B to go to the previous step): \n ')
                    #Convert str to list
                    testReview = [inputReview]
                    result = txtClfLR.predict(testReview)
                    
                    if result[0] == 0:
                        print('This predicted topic of this review is "Kindle ebooks"')
                    elif result[0] == 1:
                        print('This predicted topic of this review is "Toys"')
                    else: 
                        print('This predicted topic of this review is "Videogames"')
            elif inputClfCat == "B":
                print('Back to the previous step')
            elif inputClfCat != 'B':
                print('Please choose a valid option')
        
    elif inputClfTask == "E":
        print('See you soon!')
    elif inputClfTask != "E":
        print('Please choose a valid option')
    

