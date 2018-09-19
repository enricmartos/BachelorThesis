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
from matplotlib import pyplot as plt
from keras.utils import plot_model


def pickleRead(filename):
    data_f = open("pickled_vars/%s.pickle" % filename, "rb")
    data = pickle.load(data_f)
    data_f.close()
    return data

data = pickleRead("cleanedDataOverallBinary")
#data = data[1][['reviewText','overall']]
#data = data[2][['reviewText','overall']]
data = data[0][['reviewText','overall']]
#data = pickleRead("cleanedDataOverallBinaryAll")
#data = data.loc[:, ['reviewText','overall']]
#data = pickleRead("cleanedDataOverallMulticlassNeutral")
#data = data[0][['reviewText','overall']]
#data = pickleRead("cleanedDataTextClfDeep")

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
#tokenizer = Tokenizer( split=' ')
tokenizer.fit_on_texts(data['text'].values)
X = tokenizer.texts_to_sequences(data['text'].values)
"""
# Summarize number of words
print("Number of words: ")
print(len(np.unique(np.hstack(X))))

# Summarize review length
print("Review length: ")
result = [len(x) for x in X]
print("Mean %.2f words (%f)" % (np.mean(result), np.std(result)))
# plot review length
plt.xlabel('All reviews')
plt.ylabel('Reviews Text Length (Words)')
plt.title('Box plot of the review text length', fontweight='bold')
plt.boxplot(result)

plt.savefig("test.png")
"""

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
    model.add(Dense(50, activation='relu'))
    model.add(Dense(50, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    print(model.summary())
    return model

#TRAIN TEST SPLIT

Y = data['overall']
#Y = data['category']

# Fit the model
epochs_num = 2
batchSize = 128

model = KerasClassifier(build_fn=baseline_model, epochs=epochs_num, batch_size=batchSize, verbose=0)

#model.fit(X_train, y_train, epochs=epochs_num, batch_size=batchSize, verbose=2)
#model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=epochs_num, batch_size=128, verbose=2)
# Final evaluation of the model
#scores = model.evaluate(X_test, y_test, verbose=0)
#print("Accuracy: %.2f%%" % (scores[1]*100))

kfold = KFold(n_splits=3, shuffle=True, random_state=1)
results = cross_val_score(model, X, Y, cv=kfold)
print("Baseline: %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))