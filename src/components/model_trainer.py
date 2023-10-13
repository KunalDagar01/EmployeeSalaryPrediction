import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score,classification_report,confusion_matrix
from sklearn.model_selection import GridSearchCV
from src.logger import logging
from src.components.data_ingestion import data_ingestion
from src.pipelines.train_pipeline import trainPipeline
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from src.exception import CustomException
import pickle
import os
import sys


try:
    logging.info("At model trainier")
    raw_data_path  = os.path.join("artifacts","raw.csv")
    train_data_path,test_data_path = data_ingestion(raw_data_path)
    preprocessor,X_train,y_train,X_test,y_test = trainPipeline(train_data_path,test_data_path)

    logging.info("Transforming data")

    X_train = preprocessor.fit_transform(X_train)
    X_test = preprocessor.transform(X_test)

    logging.info("Doing hyperparameter tuning for Decision Tree Classifier")

    parameters = {
        "criterion": ['gini','entropy','log_loss'],
        "splitter":['random','best'],
        "max_depth":[1,2,3,4,5]
    }

    clf = GridSearchCV(DecisionTreeClassifier(),param_grid=parameters,cv=5,verbose=1)
    clf.fit(X_train,y_train)
    logging.info("Done hyperpamater tuning")
    tree = DecisionTreeClassifier(
        criterion=clf.best_params_['criterion'],
        splitter= clf.best_params_['splitter'],
        max_depth=clf.best_params_['max_depth']

    )
    logging.info("Model Training")
    tree.fit(X_train,y_train)

    logging.info("Training Success")
    y_pred = tree.predict(X_test)
    score = accuracy_score(y_test,y_pred)
    cfr = classification_report(y_test,y_pred)
    cfm = confusion_matrix(y_test,y_pred)

    logging.info("Training success")
    logging.info(f"Accuracy is {score*100} %")
    logging.info(f"Classification report \n {cfr} ")
    logging.info(f"confusion matrix \n {cfm}")
    logging.info("saved model file at models")

    pickle.dump(tree, open(os.path.join("models","tree.pkl"), 'wb'))

except Exception as e:
    logging.info("Some Error has occured")
    raise CustomException(e,sys)






