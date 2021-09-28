#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 13 18:10:53 2021

@author: riyagupta
"""

import pandas as pd
import numpy as np

dataset = pd.read_csv("Restaurant_Reviews.tsv", delimiter = "\t")

import re # used to replace special characters
import nltk # natural language tool kit - all the necessary libraries for nlp
nltk.download("stopwords") # nltk corpus - data is stored
from nltk.corpus import stopwords # detect stopwords
from nltk.stem.porter import PorterStemmer # used to stem your word
ps = PorterStemmer()

data = []

df = pd.DataFrame([["a"], ["b"], ["c"]], columns = ["hey"])
print(df)
print(df["hey"][0])

a = [i*i for i in range(12) if(i%2==0)]

for i in range(0,1000):
    review = dataset["Review"][i]
    review = re.sub('[^a-zA-Z]', ' ', review)
    review = review.lower()
    review = review.split() # [wow, loved, this, place]

    review = [ps.stem(word) for word in review if not word in set(stopwords.words('english'))]
    review = ' '.join(review)
    data.append(review)
    
from sklearn.feature_extraction.text import CountVectorizer

b = ["wow love place", "awesome place", "wow love ambience"]
ct = CountVectorizer(max_features = 5)
c = ct.fit_transform(b).toarray()

# ambience awesome love place wow
#   0       0       1       1   1
#   0       1       0       1   0
#   1       0       1       0   1

cv = CountVectorizer(max_features = 2000)
x = cv.fit_transform(data).toarray()
y = dataset.iloc[:,1:2].values

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=0)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
model = Sequential()
model.add(Dense(units = 1565, kernel_initializer = "random_uniform", activation = "relu"))
model.add(Dense(units = 3000, kernel_initializer = "random_uniform", activation = "relu"))
model.add(Dense(units = 3000, kernel_initializer = "random_uniform", activation = "relu"))
model.add(Dense(units = 1, kernel_initializer = "random_uniform", activation = "sigmoid"))
model.compile(optimizer = "rmsprop", loss = "binary_crossentropy", metrics = ['accuracy'])
model.fit(x_train, y_train, epochs = 5)

pred = model.predict(x_test)
pred = pred>0.5

yp = model.predict(cv.transform(["bad"]))

text = "this is awesome.... i am in love with food"

text = re.sub('[^a-zA-Z]', ' ', text)
text = text.lower()
text = text.split()

text = [ps.stem(word) for word in text if not word in set(stopwords.words('english'))]
text = ' '.join(text)
yp = model.predict(cv.transform([text]))
yp>0.5










