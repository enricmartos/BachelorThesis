# CNN for the IMDB problem
import numpy as np
from keras.preprocessing.text import Tokenizer
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM, Convolution1D, Flatten, Dropout
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D
from keras.layers.embeddings import Embedding
from keras.preprocessing.sequence import pad_sequences
import re
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import pickle
from keras.utils import np_utils
import pandas as pd
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from keras.wrappers.scikit_learn import KerasClassifier


def pickleRead(filename):
    data_f = open("pickled_vars/%s.pickle" % filename, "rb")
    data = pickle.load(data_f)
    data_f.close()
    return data

data = pickleRead("cleanedDataOverallMulticlassNeutralAll")

#Convert reviews text to lowercase
data['text'] = data['reviewText'].apply(lambda x: x.lower())
#Remove all charactares appart from a-zA-z0-9 (for instance: punctuation)
data['text'] = data['text'].apply((lambda x: re.sub('[^a-zA-z0-9\s]', '', x)))

#overallDistribution = data.overall.value_counts().sort_index()
#print(overallDistribution)

#Vocabulary size - Most 2000 common words
#Convert input text to integer sequences
max_features = 5000
tokenizer = Tokenizer(num_words=max_features, split=' ')
tokenizer.fit_on_texts(data['text'].values)
X = tokenizer.texts_to_sequences(data['text'].values)
#cap the maximum review length at 500 words,
#truncating reviews longer than that and
# padding reviews shorter than that with 0 values.
max_review_length = 500
X = pad_sequences(X,  maxlen=max_review_length)

# Using embedding from Keras
embedding_vector_length = 32
def baseline_model():
    model = Sequential()
    model.add(Embedding(max_features, embedding_vector_length, input_length=max_review_length))
    model.add(Flatten())
    model.add(Dense(250, activation='relu'))
    model.add(Dense(3, activation='sigmoid'))
    # Loss function:  how the network will be able to measure how good a job it is doing on its training data,
    # and thus how it will be able to steer itself in the right direction
    # Optimizer:  the mechanism through which the network will update itself based on the data it sees and its loss function.
    # Metrics: Metrics to monitor during training and testing. Here we will only care about accuracy
    # (the fraction of the reviews that were correctly classified).
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    print(model.summary())
    return model

#TRAIN TEST SPLIT

Y = data['overall']
encoder = LabelEncoder()
encoder.fit(Y)
encoded_Y = encoder.transform(Y)
# convert integers to dummy variables (i.e. one hot encoded)
dummy_y = np_utils.to_categorical(encoded_Y)

# Fit the model
epochs_num = 2
batchSize = 128
model = KerasClassifier(build_fn=baseline_model, epochs=epochs_num, batch_size=batchSize, verbose=0)

kfold = KFold(n_splits=3, shuffle=True, random_state=1)
results = cross_val_score(model, X, dummy_y, cv=kfold)
print("Baseline: %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))