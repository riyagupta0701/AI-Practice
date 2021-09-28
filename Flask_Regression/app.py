#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 19 17:54:31 2021

@author: riyagupta
"""

from flask import Flask, render_template, request

from tensorflow.keras.models import load_model
import joblib
import numpy as np

model = load_model("profit.h5")
ct = joblib.load("column")
sc = joblib.load("scaler")

app = Flask(__name__)

@app.route('/')
def home():
    return render_template("index.html")

@app.route('/login', methods = ["POST", "GET"])
def predict():
    if request.method == "POST":
        ms = request.form["ms"]
        ad = request.form["as"]
        rd = request.form["rd"]
        s = request.form["s"]
        data = [[ms,ad,rd,s]]
        data = ct.transform(data)
        data = np.asarray(data).astype(np.float32)
        pred = model.predict(data)


        """ 1. model file, 2. scaler file, 3. column transform """

    return render_template("index.html", value = str(pred[0][0]))

if __name__ == "__main__":
    app.run(debug = True)
