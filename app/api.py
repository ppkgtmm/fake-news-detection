import json
from fastapi import FastAPI
from app.schemas import PredictionInput, Predictions
from inference import do_prediction
from hydra import compose, initialize
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/info")
def get_app_info():
    with initialize(config_path="config", job_name="inference_app"):
        config = compose(config_name="app.yaml")
    return {
        "app_name": config.app.name,
        "version": config.app.version,
    }


@app.post("/predict")
def predict(prediction_input: PredictionInput) -> Predictions:
    predictions = do_prediction(prediction_input.texts)
    return json.loads(predictions.to_json(orient="records"))
