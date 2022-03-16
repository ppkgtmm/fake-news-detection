import json
from fastapi import FastAPI
from schemas import PredictionInput, Predictions
from inference import do_prediction

app = FastAPI()


@app.post("/predict")
def predict(prediction_input: PredictionInput) -> Predictions:
    predictions = do_prediction(prediction_input.texts)
    return json.loads(predictions.to_json(orient="records"))
