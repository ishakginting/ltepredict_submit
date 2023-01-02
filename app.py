# from crypt import methods
from flask import Flask, render_template, request
import numpy as np
import pickle
import joblib
import os

app = Flask(__name__)

modelth = joblib.load('ModelRF')

@app.route("/")
def home():
    return render_template('home.html')

@app.route("/prediksi")
def util():
    return render_template('pred.html', th = 0,util = 0, cqi=0, rssi=0)

@app.route("/predict", methods = ["POST"])
def predict():
    dl_prb_utilization, cqi_avg, rssi = [x for x in request.form.values()]
    data = np.array([[float(dl_prb_utilization), float(cqi_avg), float(rssi)]])

    predicted1 = modelth.predict(data)
    hasil1 = np.round(predicted1, decimals=3)
    ka = np.array(float(hasil1))

    return render_template("pred.html", th = ka, dl_prb_utilization=dl_prb_utilization, cqi_avg=cqi_avg, rssi=rssi)


if __name__ == "__main__":
    app.run(debug=True)