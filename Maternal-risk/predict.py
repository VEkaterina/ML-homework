import pickle
from typing import Literal, Dict, Any
from pydantic import BaseModel, Field

from fastapi import FastAPI
import uvicorn

class Person(BaseModel):
    Age: float = Field(..., ge=0.0)
    SystolicBP: float = Field(..., ge=0.0)
    DiastolicBP: float = Field(..., ge=0.0)
    BS: float = Field(..., ge=0.0)
    BodyTemp: float = Field(..., ge=0.0)
    HeartRate: float = Field(..., ge=0.0)

class PredictResponse(BaseModel):
    risk: str

app = FastAPI(title="Maternal-risk-prediction")

with open('MR_model.bin', 'rb') as m:
    pipeline = pickle.load(m)


def predict_single(person):
    result = pipeline.predict(person)
    return str(result)


@app.post("/predict")
def predict(person: Dict[str, Any]):
    risk = predict_single(person)

    return {
        "Maternal risk": risk,
    }


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=10000)