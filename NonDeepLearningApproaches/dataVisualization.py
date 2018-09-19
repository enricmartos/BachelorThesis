# -*- coding: utf-8 -*-
"""
Created on Sat Feb 24 21:10:23 2018

@author: Enric
"""

#Data Visualization

#Sort datasets by Date time (Timestamp field - Unix time)
#Videogames

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import seaborn as sns
import pickle

def loadData(filenames):
    filesNumber = len(filenames)
    data = []
    for i in range(0, filesNumber):
        newData = pd.read_json("json/%s.json" % filenames[i], lines = True)
        data.append(newData)
    return data

def plotOverallPieChart(data, productTypes, colors):
    #Unique values in Overall Column
    bins = pd.Series(data[0].loc[:,"overall"].unique())
    #Sort by overall ascending value
    bins = bins.sort_values(axis=0, ascending=True)
    labels = ['Positive', 'Neutral', 'Negative']
        
    fig = plt.figure()
    fig.suptitle("Overall Field Distribution", fontsize=14, fontweight = "bold")
    # Make square figures and axes
    the_grid = GridSpec(1, 3)    
        
    for i in range(0, len(data)):
        #Count how many rows has it's unique Overall value
        y = data[i].groupby('overall').size()
        neg_reviews_frac = (y[1] + y[2]) / len(data[i])
        neutral_reviews_frac = y[3] / len(data[i])
        pos_reviews_frac = (y[4] + y[5]) / len(data[i])
        fracs = [ pos_reviews_frac, neutral_reviews_frac, neg_reviews_frac ]
        #aspect=1 to make it a circle
        plt.subplot(the_grid[0, i], aspect=1)
        #autopct to show the percentage distribution
        plt.pie(fracs, autopct='%.1f%%', shadow=True, colors = colors)
        plt.title(productTypes[i])
        
    plt.legend(labels = labels, loc="upper right", bbox_to_anchor=(2, 1))
    plt.show()
    
    return None

def plotReviewsPerUser(data, productTypes, colors):
    
    reviewsPerUser = []
    for i in range(0, len(data)):
        #Count how many rows has it's unique Overall value
        y = data[i].groupby('reviewerID').size()
        
        reviewsNumber = len(data[i])
        usersNumber = len(y)
        newReviewsPerUser = reviewsNumber / usersNumber
        reviewsPerUser.append(newReviewsPerUser)
    
    plt.bar(productTypes, reviewsPerUser, color = colors, width = 0.5)
    
    plt.xlabel('Product Type')
    plt.ylabel('Number of Reviews per User')
    plt.title('Mean Reviews Per User', fontweight='bold')
    
    plt.show()
    
    return None
    

def checkFirstAndLastReviewDate(data):
    
    firstLastReviewTimes = []
    for i in range(0, len(data)):
        idxmin = data[i]["unixReviewTime"].idxmin()
        
        firstReviewTime = data[i].loc[idxmin, "reviewTime"]
        
        idxmax = data[i]["unixReviewTime"].idxmax()
        
        lastReviewTime = data[i].loc[idxmax, "reviewTime"]
        
        firstLastReviewTimes.append([firstReviewTime, lastReviewTime])

    return firstLastReviewTimes

def getReviewsTemporalDistribution(data):
    
    totalReviews = []
    
    for i in range(0, len(data)):
        #Sort data to make computations faster
        sortedData = data[i].sort_values("unixReviewTime", ascending=1)
        sortedData = sortedData.reset_index(drop=True)
        
        reviewsPerPeriod = np.zeros(14)
        includedYears = np.arange(2000, 2014)
        
        for j in range(0, len(data[i])):
            for z in range(0, len(reviewsPerPeriod)):
                currentYear = str(includedYears[z])
                if sortedData.loc[j,'reviewTime'][7:11] == currentYear:
                    reviewsPerPeriod[z] += 1
        
        totalReviews.append(reviewsPerPeriod)
    
    return totalReviews
        
def plotReviewsTemporalDistribution(totalReviews, productTypes, colors):    
    X = np.arange(2000, 2014)
    
    fig = plt.figure()
    fig.suptitle("Reviews Temporal Distribution", fontsize=14, fontweight = "bold")
    
    ax1 = plt.subplot(211)
    ax2 = plt.subplot(212, sharex = ax1)
    #Plotting linestyle, color and legend
    ax1.plot(X, totalReviews[0], \
             c=colors[0], label= productTypes[0])
    
    plt.setp(ax1.get_xticklabels(), visible=False)
    
    for i in range(1, len(totalReviews)):
    
        #Plotting linestyle, color and legend
        ax2.plot(X, totalReviews[i], \
                 c=colors[i], label= productTypes[i])
 
    # Set common labels
    fig.text(0.5, 0.04, 'Time (Years)', ha='center', va='center')
    fig.text(0.0, 0.5, 'Number of Reviews', ha='center', va='center', rotation='vertical')
    
    fig.legend(loc="upper right", bbox_to_anchor=(1.25, 1))
    
    plt.savefig('dataVisualizationFigures/reviewsTemporalDistribution.png')
    plt.show()
    
    return None

def getHelpfulDistribution(data):
    helpfulList = []
    helpfulRate = []
    helpfulNull = [0, 0]
    nonEmptyhelpfulCtr = 0
    emptyHelpfulCtr = 0
    helpfulCtr = 0
    nonHelpfulCtr = 0
    for i in range(0, len(data)):    
        for j in range(0, len(data[i])):
            if data[i].helpful[j] != helpfulNull or \
                data[i].helpful[j][1] != 0:
                nonEmptyhelpfulCtr += 1
                if data[i].helpful[j][0] / data[i].helpful[j][1] >= 0.5:
                    helpfulCtr += 1
        emptyHelpfulCtr = len(data[i]) - nonEmptyhelpfulCtr
        helpfulList.append([nonEmptyhelpfulCtr, emptyHelpfulCtr])
        nonHelpfulCtr = nonEmptyhelpfulCtr - helpfulCtr
        helpfulRate.append([helpfulCtr, nonHelpfulCtr])
        nonEmptyhelpfulCtr = 0 
        helpfulCtr = 0
    return helpfulList, helpfulRate

def getHelpfulDistribution2(data):
    helpfulList = []
    helpfulRate = []
    helpfulNull = [0, 0]
    nonEmptyhelpfulCtr = 0
    emptyHelpfulCtr = 0
    helpfulCtr = 0
    nonHelpfulCtr = 0
    for i in range(0, len(data)):    
        for j in range(0, len(data[i])):
            if data[i].helpful[j] != helpfulNull or \
                data[i].helpful[j][1] != 0:
                nonEmptyhelpfulCtr += 1
                if data[i].helpful[j][1] >= 4 and data[i].helpful[j][0] / data[i].helpful[j][1] >= 0.5:
                    helpfulCtr += 1
        emptyHelpfulCtr = len(data[i]) - nonEmptyhelpfulCtr
        helpfulList.append([nonEmptyhelpfulCtr, emptyHelpfulCtr])
        nonHelpfulCtr = nonEmptyhelpfulCtr - helpfulCtr
        helpfulRate.append([helpfulCtr, nonHelpfulCtr])
        nonEmptyhelpfulCtr = 0 
        helpfulCtr = 0
    return helpfulList, helpfulRate

def helpfulBining(cleanedData):
    for i in range(0, len(cleanedData)):
        #data[i]['helpfulBin'] = data[i]['helpful'] != 0 and \
         #    (data[i]['helpful'][0] / data[i]['helpful'][1]) >= 0.5 
        print("Processing Set number", i )
        for j in range(0, len(cleanedData[i])):
            #NonEmpty Helpful Field
            if cleanedData[i].helpful[j][1] >= 4:
                if cleanedData[i].helpful[j][0] / cleanedData[i].helpful[j][1] >= 0.5:
                    cleanedData[i].loc[j,'helpful'] = 1
                else:
                    cleanedData[i].loc[j,'helpful'] = -1
            else:
                cleanedData[i].loc[j,'helpful'] = 0
                
    return cleanedData

def filterBiningData(helpfulBiningData):
    filteredBiningData = []
    
    for i in range(0, len(helpfulBiningData)):
        newfilteredBiningData = pd.DataFrame(columns=["reviewText", "helpful", "overall"])
        for j in range(0, len(helpfulBiningData[i])):
            if helpfulBiningData[i].helpful[j] != 0 :
                newfilteredBiningData.loc[j, 'reviewText'] = helpfulBiningData[i].loc[j,'reviewText']
                newfilteredBiningData.loc[j, 'helpful'] = helpfulBiningData[i].loc[j,'helpful']
                newfilteredBiningData.loc[j, 'overall'] = helpfulBiningData[i].loc[j,'overall']
                newfilteredBiningData.loc[j, 'textLength'] = helpfulBiningData[i].loc[j,'textLength']
        newfilteredBiningData = newfilteredBiningData.reset_index(drop=True)
        filteredBiningData.append(newfilteredBiningData)
    return filteredBiningData
                
    

def plotHelpfulListDistribution(helpfulList, productTypes, colors):
    labels = ['Non-empty Helpful Field', 'Empty Helpful Field']
    # Make square figures and axes
    the_grid = GridSpec(1, 3)    
    
    colors = [colors[0], colors[2]]
    
    fig = plt.figure()
    fig.suptitle("Empty and Non Empty Helpful Field Distribution",
                 fontsize=14, fontweight = "bold")
    
    for i in range(0, len(data)):
        nonEmptyHelpfulFrac = helpfulList[i][0] / len(data[i])
        emptyHelpfulFrac = helpfulList[i][1] / len(data[i])
        fracs = [ nonEmptyHelpfulFrac, emptyHelpfulFrac ]
        #aspect=1 to make it a circle
        plt.subplot(the_grid[0, i], aspect=1)
        #autopct to show the percentage distribution
        plt.pie(fracs, autopct='%.0f%%', colors = colors, shadow=True)
        
        plt.title(productTypes[i])
    
    plt.legend(labels = labels, loc="upper right", bbox_to_anchor=(3, 1))
    #plt.title("How many reviews have been evaluated by users? [Helpful Field]")
    plt.show()
    

def plotHelpfulRateDistribution(helpfulRate, helpfulList,
                                productTypes, colors):
    labels = ['Positive Helpful Field', 'Negative Helpful Field']
    # Make square figures and axes
    the_grid = GridSpec(1, 3)    
    
    colors = [colors[0], colors[2]]
    
    fig = plt.figure()
    fig.suptitle("Positive and Negative Helpful Field Distribution",
                 fontsize=14, fontweight = "bold")
    
    for i in range(0, len(data)):
        posHelpfulFrac = helpfulRate[i][0] / helpfulList[i][0]
        negHelpfulFrac = helpfulRate[i][1] / helpfulList[i][0]
        fracs = [ posHelpfulFrac, negHelpfulFrac ]
        #aspect=1 to make it a circle
        plt.subplot(the_grid[0, i], aspect=1)
        #autopct to show the percentage distribution
        plt.pie(fracs, autopct='%.0f%%', colors = colors, shadow=True)
        
        plt.title(productTypes[i])
    
    plt.legend(labels = labels, loc="upper right", bbox_to_anchor=(3, 1))
    #plt.title("How many reviews have been evaluated by users? [Helpful Field]")
    plt.show()
    
def plotHelpfulRateDistribution(data,
                                productTypes, colors):
    for i in range(0, len(helpfulBiningData)):
        helpfulBiningData[i].helpful.value_counts().sort_index()
    

def addTextLength(data):
    for i in range(0, len(data)):
        data[i]['textLength'] = data[i]['reviewText'].apply(len)
    return data

#Data Bining of Overall field
#1 and 2 stars -> 0 (Negative)
#4 and 5 stars -> 1 (Positive)
#Input:
#[data]: Dataframe list of n datasets
#[overallSegmentSize]: Size of each overall field possible value
#Output [cleanedData]: Dataframe list of n datasets
def overallBining(data, overallSegmentSize):
    #Select and limit Ngeative Reviews
    negReviews1 = data[data.overall == 1]
    negReviews1Limited = negReviews1.iloc[:overallSegmentSize ,:]
    
    negReviews2 = data[data.overall == 2]
    negReviews2Limited = negReviews2.iloc[:overallSegmentSize, :]
    
    negReviews3= data[data.overall == 3]
    negReviews3Limited = negReviews3.iloc[:overallSegmentSize, :]
    
    negReviews = pd.concat([negReviews1Limited, negReviews2Limited])
    
    allReviews = pd.concat([negReviews, negReviews3Limited])
    #Select and limit Positives Reviews
    posReviews4 = data[data.overall == 4]
    posReviews4Limited = posReviews4.iloc[:overallSegmentSize, :]
    
    posReviews5 = data[data.overall == 5]
    posReviews5Limited = posReviews5.iloc[:overallSegmentSize, :]

    posReviews = pd.concat([posReviews4Limited, posReviews5Limited])
    
    cleanedData = pd.concat([allReviews, posReviews])
    #Reset indexes to iterate cleanedData Dataframe
    cleanedData = cleanedData.reset_index(drop=True)
     
    return cleanedData

def preprocess(data):

    # examine overall class distribution
    #overallDistribution = data.overall.value_counts().sort_index()
    #print(overallDistribution)

    # filter the DataFrame 
    cleanedData = []
    for i in range(0, len(data)):
        data[i] = data[i].loc[:, ["asin", "reviewText", "overall", "helpful", 
                                "textLength"]]
        overallDistribution = data[i].overall.value_counts().sort_index()
        overallSegmentSize = overallDistribution.min(axis = 0)
        newCleanedData = overallBining(data[i], overallSegmentSize)
        cleanedData.append(newCleanedData)
    #cleanedData = overallBining1and5(filteredData, overallSegmentSize)
    return cleanedData

def pickleRead(filename):
    data_f = open("pickled_vars/%s.pickle" % filename, "rb")
    data = pickle.load(data_f)
    data_f.close()
    return data

def pickleWrite(variable, filename):
    saveData = open("pickled_vars/%s.pickle" % filename,"wb")
    pickle.dump(variable, saveData)
    saveData.close()
    
productTypes = ["Kindle", "Toys and Games", "Videogames"]
colors = ["yellowgreen", "lightskyblue", "lightcoral"]
filenames = ["kindle_reviews", "Toys_and_Games_5", "Video_Games_5" ]

#directory: cd "Documents/Python Spyder Projects"
#data = loadData(filenames)
data = pickleRead("dataTextLength")

#cleanedData = preprocess(data)
cleanedData = pickleRead("preprocessedDataOverallSegmentSizeVis")

#helpfulBiningData = helpfulBining(cleanedData)
helpfulBiningData = pickleRead("helpfulBiningData")

filteredBiningData = filterBiningData(helpfulBiningData)
pickleWrite(filteredBiningData, "helpfulBiningDataWithout0")
helpfulBiningDataWithout0 = pickleRead("helpfulBiningDataWithout0")


g = sns.FacetGrid(data=cleanedData, col='overall').map(plt.hist, 
                 "textLength", y_var="Number of reviews", x_var = "Text length (Chars)", bins = 50)
#g = g.map(plt.hist, "textLength")

sns.boxplot(x='overall', y='textLength', data=cleanedData)

overall = helpfulBiningDataWithout0[2].groupby('overall').mean()
print(overall)
#overall = filteredBiningData[0].groupby('overall').mean()
overall.corr()
sns.heatmap(data=overall.corr(), annot=True)

plotOverallPieChart(data, productTypes, colors)

plotReviewsPerUser(data, productTypes, colors)

#firstLastReviewTimes = checkFirstAndLastReviewDate(data)
#totalReviews = getReviewsTemporalDistribution(data)
dataVtotalReviews = pickleRead("dataVtotalReviews")

#plotReviewsTemporalDistribution(totalReviews, productTypes, colors)
plotReviewsTemporalDistribution(dataVtotalReviews, productTypes, colors)

#helpfulList, helpfulRate = getHelpfulDistribution(data)
helpfulList = pickleRead("dataVhelpfulList")
helpfulRate = pickleRead("dataVhelpfulRate")

plotHelpfulListDistribution(helpfulList, productTypes, colors)

plotHelpfulRateDistribution(helpfulRate, helpfulList, productTypes, colors)
plotHelpfulRateDistributionRestrictive(helpfulBiningData, productTypes, colors)




