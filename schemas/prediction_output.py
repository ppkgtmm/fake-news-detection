from typing import List
from pydantic import BaseModel


class PredictionOutput(BaseModel):
    text: str
    probability: List[float]
    prediction: float
    label: str


class Predictions(BaseModel):
    predictions: List[PredictionOutput]
