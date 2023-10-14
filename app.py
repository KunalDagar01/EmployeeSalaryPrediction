from flask import Flask, request, render_template
import pandas as pd
import pickle
from src.logger import logging
from src.exception import CustomException
from src.components.predict import prediction
import sys
import os

try:
    logging.info("In App.py")
    app = Flask(__name__)

    logging.info("Showing Index page")
    @app.route('/')
    def index():
        return render_template('index.html')

    
    @app.route('/submit',methods=['GET','POST'])
    def submit():
        logging.info("Submit command recieved")
        if request.method=="POST":
            
            age = int(request.form.get('age'))
            workclass = request.form.get('workclass')
            fnlwgt = int(request.form.get('fnlwgt'))
            education = request.form.get('education')
            education_num = int(request.form.get('education-num'))
            marital_status = request.form.get('marital-status')
            occupation = request.form.get('occupation')
            relationship = request.form.get('relationship')
            race = request.form.get('race')
            sex = request.form.get('sex')
            capital_gain = int(request.form.get('capital-gain'))
            capital_loss = int(request.form.get('capital-loss'))
            hours_per_week = int(request.form.get('hours-per-week'))
            country = request.form.get('country')

            inputs = [age,workclass,fnlwgt,education,education_num,marital_status,occupation,relationship,race,sex,capital_gain,capital_loss,hours_per_week,country]
            logging.info(f"recieved values {inputs}")
            result = prediction(inputs)
            if(result==0):
                logging.info("Done Prediction")
                return render_template('index.html',prediction="<50k")
            else:
                logging.info("Done Prediction")
                return render_template('index.html',prediction=">50k")
            
            
            
    if __name__ == "__main__":
        app.run()
except Exception as e:
    logging.info(CustomException(e,sys))
    raise CustomException(e,sys)