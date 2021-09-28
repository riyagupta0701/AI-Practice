#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 15 12:05:19 2021

@author: riyagupta
"""

import pandas as pd
import numpy as np

dataset = pd.read_csv("IMDB_Dataset.csv")

import re
import nltk
nltk.download("stopwords")
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
ps = PorterStemmer()

data = []

for i in range(0, 50000):
    info = dataset["review"][i]
    info = re.sub('[^a-zA-Z]', ' ', info)
    info = info.lower()
    info = info.split()
    
    #info = [ps.stem(word) for word in info if not word in set(stopwords.word('english'))]
    info = [ps.stem(word) for word in info if not word in set(stopwords.words('english'))]
    info = ' '.join(info)
    data.append(info)
    
from sklearn.feature_extraction.text import CountVectorizer

cv = CountVectorizer(max_features=5000)
x = cv.fit_transform(data).toarray()
y = dataset.iloc[:,1:2].values

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2,random_state=1)

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
y_train = le.fit_transform(y_train)
y_test = le.fit_transform(y_test)

print(y_test)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

model = Sequential()
model.add(Dense(units = 5000, kernel_initializer = "random_uniform", activation = "relu"))
model.add(Dense(units = 10000, kernel_initializer = "random_uniform", activation = "relu"))
model.add(Dense(units = 10000, kernel_initializer = "random_uniform", activation = "relu"))
model.add(Dense(units = 1, kernel_initializer = "random_uniform", activation = "sigmoid"))
model.compile(optimizer = "rmsprop", loss = "binary_crossentropy", metrics = ['accuracy'])
model.fit(x_train, y_train, epochs = 5)


    