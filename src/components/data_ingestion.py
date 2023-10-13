import pandas as  pd
import sys
import os
from src.logger import logging
from sklearn.model_selection import train_test_split

raw_data = os.path.join("artifacts","raw.csv")
# train_path = os.path.join("artifacts","train.csv")
# test_path = os.path.join("artifacts","test.csv")

def data_ingestion(raw_data):
    try:
        logging.info("Data Ingestion is started")
        
        df = pd.read_csv(raw_data)
        (train_data,test_data) = train_test_split(df,test_size=0.3,random_state=42)
        logging.info("Saving data")
        train_path = os.path.join("artifacts","train.csv")
        test_path = os.path.join("artifacts","test.csv")
        train_data.to_csv(train_path,index=False)
        test_data.to_csv(test_path,index=False)
        logging.info("Ingestion success")
        logging.info(f'Training data stored at {train_path}')
        logging.info(f'Test path stored at {test_path}')

        return train_path,test_path
    except Exception as e:
        logging.info("Some error has occured")
    
if __name__ == "__main__":
    train_path,test_path = data_ingestion(raw_data=raw_data)
    print(f'The train data path is {train_path}')
    print(f'The test data path is {test_path}')


