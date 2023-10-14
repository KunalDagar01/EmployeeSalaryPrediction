import pandas as pd
from src.logger import logging
from src.exception import CustomException
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
import sys
import os
import pickle



def prediction(inputs):
    try:
        logging.info("In Predict py importing models")
        preprocessor = pickle.load(open(os.path.join("models","preprocessor.pkl"),"rb"))
        model = pickle.load(open(os.path.join("models","tree.pkl"),"rb"))
        
        classNames = ['age', 'workclass', 'fnlwgt', 'education', 'education-num',
        'marital-status', 'occupation', 'relationship', 'race', 'sex',
        'capital-gain', 'capital-loss', 'hours-per-week', 'country']
        
        logging.info("Data Transoformation")
        data = pd.DataFrame([inputs],columns=classNames)
        logging.info("Predicting Results")
        data = preprocessor.transform(data)
        result = model.predict(data)
        if result[0]==0:
            logging.info(f"Result is {result[0]}")
            return 0
        else:
            logging.info(f"Result is {result[0]}")
            return 1
        

        
    except Exception as e:
        logging.info(CustomException(e,sys))
        raise CustomException(e,sys)

if __name__ == "__main__":
    inputs = [30,'Private',190040,'Bachelors',13,'Never-married','Machine-op-inspct','Not-in-family','White','Female',0,0,40,'United-States']
    result = prediction(inputs)
    print(result)
