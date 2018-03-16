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
    plt.show()
    
    return None

def getHelpfulDistribution(data):
    helpfulList = []
    helpfulNull = [0, 0]
    helpfulCtr = 0
    nonHelpfulCtr = 0
    for i in range(0, len(data)):    
        for j in range(0, len(data[i])):
            if data[i].helpful[j] != helpfulNull:
                helpfulCtr += 1
        nonHelpfulCtr = len(data[i]) - helpfulCtr
        helpfulList.append([helpfulCtr, nonHelpfulCtr])
        helpfulCtr = 0  
    return helpfulList

def plotHelpfulDistribution(helpfulList, productTypes, colors):
    labels = ['Non-empty Helpful Field', 'Empty Helpful Field']
    # Make square figures and axes
    the_grid = GridSpec(1, 3)    
    
    colors = [colors[0], colors[2]]
    
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
    
    return None
    
productTypes = ["Kindle", "Toys and Games", "Videogames"]
colors = ["yellowgreen", "lightskyblue", "lightcoral"]
filenames = ["kindle_reviews", "Toys_and_Games_5", "Video_Games_5" ]

data = loadData(filenames)

plotOverallPieChart(data, productTypes, colors)

plotReviewsPerUser(data, productTypes, colors)

#firstLastReviewTimes = checkFirstAndLastReviewDate(data)
totalReviews = getReviewsTemporalDistribution(data)

plotReviewsTemporalDistribution(totalReviews, productTypes, colors)
helpfulList = getHelpfulDistribution(data)
plotHelpfulDistribution(helpfulList, productTypes, colors)

