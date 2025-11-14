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
    path = kagglehub.dataset_download("rabieelkharoua/parkinsons-disease-dataset-analysis")
    
    file_path = os.path.join(path, "parkinsons_disease_data.csv")

    df = pd.read_csv(file_path)

    df = df.drop(['PatientID', 'DoctorInCharge'], axis = 1)

    yes_no = {
    0: 'no',
    1: 'yes'}

    df.Tremor = df.Tremor.map(yes_no)
    df.Rigidity = df.Rigidity.map(yes_no)
    df.Bradykinesia = df.Bradykinesia.map(yes_no)
    df.PosturalInstability = df.PosturalInstability.map(yes_no)

    return df


def train_model(df):
    
    cat_predictors = ['Tremor', 'Rigidity', 'Bradykinesia', 'PosturalInstability']
    num_predictors = ['UPDRS', 'FunctionalAssessment', 'MoCA', 'Age', 'SleepQuality', 'BMI', 'AlcoholConsumption', 'DietQuality']


    y_train = df.Diagnosis
    train_dict = df[cat_predictors + num_predictors].to_dict(orient='records')

    pipeline = make_pipeline(
        DictVectorizer(),
        RandomForestClassifier(n_estimators=57, max_depth=15, random_state=1)
    )

    pipeline.fit(train_dict, y_train)

    return pipeline

def save_model(pipeline, output_file):
    with open(output_file, 'wb') as f_out:
        pickle.dump(pipeline, f_out)
    

df = load_data()
pipeline = train_model(df)
save_model(pipeline, 'PD_model.bin')

print('Model saved to PD_model.bin')