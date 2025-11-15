import pickle
from typing import Literal, Dict, Any
from pydantic import BaseModel, Field

from fastapi import FastAPI
import uvicorn

class Person(BaseModel):
    Tremor: Literal["yes", "no"]
    Bradykinesia: Literal["yes", "no"]
    Rigidity: Literal["yes", "no"]
    PosturalInstability: Literal["yes", "no"]
    UPDRS: float = Field(..., ge=0.0)
    FunctionalAssessment: float = Field(..., ge=0.0)
    MoCA: float = Field(..., ge=0.0)
    Age: float = Field(..., ge=0.0)
    SleepQuality: float = Field(..., ge=0.0)
    BMI: float = Field(..., ge=0.0)
    AlcoholConsumption: float = Field(..., ge=0.0)
    DietQuality: float = Field(..., ge=0.0)

class PredictResponse(BaseModel):
    churn_probability: float
    churn: bool

app = FastAPI(title="PD-prediction")

with open('PD_model.bin', 'rb') as m:
    pipeline = pickle.load(m)


def predict_single(person):
    result = pipeline.predict_proba(person)[0, 1]
    return float(result)


@app.post("/predict")
def predict(person: Dict[str, Any]):
    prob = predict_single(person)

    return {
        "Probability of being diagnosed with PD": prob,
        "PD diagnosis": bool(prob >= 0.5)
    }


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=10000)