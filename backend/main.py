# app/main.py

from fastapi import FastAPI, Depends
from schemas import PredictionRequest, PredictionResponse
from services import prediction_service, PredictionService
from scripts.database import init_db

app = FastAPI()

@app.on_event("startup")
def on_startup():
    init_db()

@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest):
    """
    Accepts a text query and returns the predicted ticket category and probabilities.
    """
    result = prediction_service.predict_ticket(request.text)
    return PredictionResponse(**result)

@app.get("/")
def read_root():
    return {"message": "Classification model API is running."}