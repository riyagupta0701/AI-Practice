#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 22 22:44:08 2021

@author: riyagupta
"""

from flask import Flask, request, render_template

import numpy as np
import os
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image


model = load_model("animal.h5")
app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods = ['POST', 'GET'])
def upload():
    if request.method == 'POST':
        file = request.files['image']
        basepath = os.path.dirname(__file__)
        filepath = os.path.join(basepath, 'uploads', file.filename)
        file.save(filepath)

        img = image.load_img(filepath, target_size = (64,64))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        pred = model.predict(x)
        pred = np.argmax(pred)

        index = ['Bear', 'Crow', 'Elephant', 'Raccoon', 'Rat']
        prediction = "The predicted animal is: " + str(index[pred])

        return render_template("index.html", value=prediction)

if __name__ == '__main__':
    app.run(debug = True)
