import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from src.logger import logging
from src.components.data_ingestion import data_ingestion
import pickle
import sys
import os



def trainPipeline(train_data_path,test_data_path):
    try:
        logging.info("Initiating Pipeline")
    
        # train_data_path,test_data_path = data_ingestion(os.path.join("artifacts","raw.csv"))
        train_data = pd.read_csv(train_data_path)
        test_data = pd.read_csv(test_data_path)

        logging.info("Splitting data in train and test")
        X_train,y_train = train_data.drop('salary',axis=1),train_data['salary']
        X_test,y_test = test_data.drop('salary',axis=1),test_data['salary']

        numerical = X_train.select_dtypes(exclude="object")
        categorical = X_train.select_dtypes(include="object")

        num_features = list(numerical.columns)
        cat_features = list(categorical.columns)

        logging.info("Establishing Pipeline")
        num_pipeline = Pipeline(
            steps=[
                ("SimpleImputer",SimpleImputer(strategy='median')),
                ("scaler",StandardScaler())
            ]
        )
        cat_pipeline = Pipeline(
            steps=[
                ("imputer",SimpleImputer(strategy="most_frequent")),
                ("encoder",OneHotEncoder()),
                

            ]
        )
        
        preprocessor = ColumnTransformer([
            ('num_pipeline',num_pipeline,num_features),
            ('cat_pipeline',cat_pipeline,cat_features)
        ])

        logging.info("Pipeline creation done successfully")

        logging.info("Pipleine Successfully created saving pickle file of preprocessor")
        pickle.dump(preprocessor, open(os.path.join("models","preprocessor.pkl"), 'wb'))
        logging.info("Preprocessor saved in models")
        return preprocessor,X_train,y_train,X_test,y_test
        


    except Exception as e:
        logging.info("Some error occured")

if __name__ == "__main__":
    preprocessor = trainPipeline()
