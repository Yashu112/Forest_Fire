import pickle
from flask import Flask, request, jsonify, render_template
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import logging
logging.basicConfig(filename="model.log" , level=logging.INFO)

app = Flask(__name__)

## import ridge regressor model and standard scaler
try:
    ridge_model=pickle.load(open('models/ridge.pkl','rb'))
    standard_scaler_model=pickle.load(open('models/std_scaler.pkl','rb'))
except Exception as e:
    logging.info(e)

## Route for Homepage
@app.route('/')
def index():
    return render_template('index.html')    

@app.route('/predictdata',methods=['GET','POST'])
def predict_datapoint():
    if request.method=='POST':
        try:
            try:
                Temperature = float(request.form.get('Temperature'))
                RH = float(request.form.get('RH'))
                Ws = float(request.form.get('Ws'))
                Rain = float(request.form.get('Rain'))
                FFMC = float(request.form.get('FFMC'))
                DMC = float(request.form.get('DMC'))
                ISI = float(request.form.get('ISI'))
                Classes = float(request.form.get('Classes'))
                Region = float(request.form.get('Region'))
            except Exception as e:
                logging.info(e)

            try:
                new_data_scaled=standard_scaler_model.transform([[Temperature, RH, Ws, Rain, FFMC,DMC,ISI,Classes,Region]])
                result=ridge_model.predict(new_data_scaled)
            except Exception as e:
                logging.info(e)

            return render_template('home.html',result=result[0])

        except Exception as e:
            logging.info(e)
            return('Something went wrong')
    else:
        return render_template('home.html')

if __name__=="__main__":
    app.run(host="0.0.0.0")
