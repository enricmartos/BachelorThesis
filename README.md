# BachelorThesis - Sentiment analysis on Amazon product reviews

This repository contains the main Python scripts of my Bachelor Thesis (Sentiment analysis on Amazon product reviews).

# Abstract

Text mining has proved to be a crucial tool for companies in order to know their customers opinion. By deriving high-quality information from large volumes of text, it is possible to understand how the market preferences evolve beyond sales statistics. For this reason, the goal of this bachelor thesis is to perform an accurate sentiment analysis on Amazon product reviews.
Three different review datasets (ebooks, toys and video games) configure the starting point to extract and quantify affective states by applying natural language processing techniques. The aforementioned datasets are provided by Kaggle, a collaborative data science platform. 
Thus, supervised learning algorithms, including deep learning approaches, have been employed to predict the overall sentiment and the usefulness behind a product review. In addition, a topic-based categorization has been also carried out in order to classify unseen reviews into one specific product type.

# Goals

1. Prediction of the dominant sentiment behind each review: Considering that each review has its own overall field, we can use it to evaluate the accuracy of our model by comparing our output with the real value.

2. Prediction of the helpfulness of a review: This is a similar case as the previous one, but now we are going to predict how helpful a given review can be based on its body of text. We can use the helpful field to evaluate our model accuracy as well.

3. Topic categorization of a review:  Its goal is to determine the topic, i.e. the product type, of unseen reviews.

# Usage

The structure of the code is organized as follows:

- Datasets folder

- Pickled (or prestored) variables folder

- Non deep learning approaches folder:

-- Data Visualization.py

-- OverallPrediction.py

-- HelpfulnessPrediction.py

-- TopicPrediction.py

Deep learning approaches folder:

-- MLP folder: Overall, Helpfulness and Topic Prediction Python files based on MLP

-- CNN folder: Overall, Helpfulness and Topic Prediction Python files based on MLP

Please note that most of this code is not still ready to be executed in a friendly way,
and it may output errors depending on your set up. However, it contains all the
methods and resources that have been employed in order to accomplish the goals 
of this project. 

However, the most important script here is "OverallSentimentPrediction":

OverallSentimentPrediction: This script lets the user test several precomputed classifiers 
in the three different classification tasks. First of all, the user chooses the classification 
task along with the desired classifier. Then, the program expects a text review that will be accordingly classified.
Finally, the user can check if the predicted results is the correct one or not. The pickled variables inside pickled_vars folder can be downloaded through [this](https://drive.google.com/file/d/1_MBrMkOGYufBgYR7hbErjLDS4qGduHxB/view) Google Drive folder. [This](https://drive.google.com/drive/folders/1AcRAppxsPzAeFOCV3ZwWl7ylms2soMw6?usp=sharing) second link also includes the final report of the project, and a summarized presentation. Finally, the video demo below shows the main features of the program.

<a href="http://www.youtube.com/watch?feature=player_embedded&v=hO7awUAFZyU
" target="_blank"><img src="https://github.com/enricmartos/BachelorThesis/blob/master/NonDeepLearningApproaches/Thumbnail_Demo.png" 
width="600" height="350" border="10"/></a>



