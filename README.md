# BachelorThesis - Sentiment analysis on Amazon product reviews

This repository will contain the main Python scripts of my Bachelor Thesis (Sentiment analysis on Amazon product reviews).

# Abstract

Text mining has proved to be a crucial tool for companies in order to know their customers opinion. By deriving high-quality information from large volumes of text, it is possible to understand how the market preferences evolve beyond sales statistics. For this reason, the goal of this bachelor thesis is to perform an accurate sentiment analysis on Amazon product reviews.
Three different review datasets (ebooks, toys and video games) configure the starting point to extract and quantify affective states by applying natural language processing techniques. The aforementioned datasets are provided by Kaggle, a collaborative data science platform. 
Thus, supervised learning algorithms, including deep learning approaches, have been employed to predict the overall sentiment and the usefulness behind a product review. In addition, a topic-based categorization has been also carried out in order to classify unseen reviews into one specific product type.

# Goals

1. Prediction of the dominant sentiment behind each review: Considering that each review has its own overall field, we can use it to evaluate the accuracy of our model by comparing our output with the real value.

2. Prediction of the helpfulness of a review: This is a similar case as the previous one, but now we are going to predict how helpful a given review can be based on its body of text. We can use the helpful field to evaluate our model accuracy as well.

3. Topic categorization of a review:  Its goal is to determine the topic, i.e. the product type, of unseen reviews.

# Code

- OverallSentimentPrediction: This script lets the user test several precomputed classifiers in the three different classification tasks. First of all, the user chooses the classification task along with the desired classifier. Then, the program expects a text review that will be accordingly classified. Finally, the user can check if the predicted results is the correct one or not. 

