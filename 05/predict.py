import pickle
import uvicorn
from fastapi import FastAPI
from typing import Dict, Any
import requests

app = FastAPI(title = "churn-prediction")

with open('pipeline_v2.bin', 'rb') as f_in: 
    model = pickle.load(f_in)

def predict_single(customer):
    result = model.predict_proba(customer)[0, 1]
    return float(result)

@app.post("/predict")
def predict(customer: Dict[str, Any]):
    churn = predict_single(customer)

    return {
       " churn_probability": churn,
       "churn": bool(churn >= 0.5)
    }


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=9696)
