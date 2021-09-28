#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 16 18:08:32 2021

@author: riyagupta
"""

from flask import Flask, render_template, request
app = Flask(__name__)
@app.route('/')
def hello_world():
    return render_template("index.html")

@app.route('/guest', methods = ["POST"])
def Guest():
    p = request.form["a"]
    print(p)
    return render_template("index.html", y = p)

@app.route('/user')
def User():
    return "Hello User! Welcome"

app.run(debug = True)
 