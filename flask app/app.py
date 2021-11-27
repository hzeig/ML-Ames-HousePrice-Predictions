import os
import pandas as pd
import numpy as np
import pickle
from flask import Flask, render_template, request, url_for

app = Flask(__name__)

@app.route('/home')
def home():
    return render_template('home.html')

@app.route('/')
def index():
    return render_template('index.html')

def ValuePredictor(to_predict_list):
    to_predict = np.array(to_predict_list).reshape(1,4)
    loaded_model = pickle.load(open('../fakeLinear.sav'))
    result = loaded_model.predict(to_predict)
    return result[0]

@app.route('/predict', methods = ['POST'])
def result():
    if request.method == 'POST':
        to_predict_list = request.form.to_dict()
        to_predict_list=list(to_predict_list.values())
        to_predict_list = list(map(float, to_predict_list))
        to_predict_list = list(map(float, to_predict_list))
        result = ValuePredictor(to_predict_list)
        prediction = str(result)
        return render_template('predict.html', prediction=prediction)

@app.route('/models')
def models():
    return render_template('models.html')

@app.route('/team')
def team():
    return render_template('team.html')

    if __name__ == '__main__':
        app.run(debug=True)
