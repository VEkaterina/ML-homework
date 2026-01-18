import pandas as pd
import numpy as np
import pickle

import kagglehub
import os

import sklearn
from sklearn.feature_extraction import DictVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import make_pipeline

def load_data():
    path = kagglehub.dataset_download("csafrit2/maternal-health-risk-data")
    
    file_path = os.path.join(path, "Maternal Health Risk Data Set.csv")

    df = pd.read_csv(file_path)

    df.RiskLevel = df.RiskLevel.str.lower().str.replace(' ', '_')

    return df


def train_model(df):
    
    y_train = df.RiskLevel
    
    del df['RiskLevel']
    train_dict = df.to_dict(orient='records')

    pipeline = make_pipeline(
        DictVectorizer(),
        RandomForestClassifier(n_estimators=90, max_depth=10, random_state=1)
    )

    pipeline.fit(train_dict, y_train)

    return pipeline

def save_model(pipeline, output_file):
    with open(output_file, 'wb') as f_out:
        pickle.dump(pipeline, f_out)
    

df = load_data()
pipeline = train_model(df)
save_model(pipeline, 'MR_model.bin')

print('Model saved to MR_model.bin')