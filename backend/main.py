# app/main.py

from fastapi import FastAPI, Depends
from services import prediction_service, answer_service
from schemas import (
    PredictionRequest, PredictionResponse, 
    AnswerRequest, TopAnswersResponse
)
from scripts.database import init_db, get_session
from sqlmodel import create_engine, Session
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

@app.post("/find-answer", response_model=TopAnswersResponse, tags=["Answer Retrieval"])
def find_top_answers(request: AnswerRequest, session: Session = Depends(get_session)):
    """
    Accepts a text query, finds the top 3 most similar historical tickets, 
    and returns their answer patterns.
    """
    # Call the updated service method
    results_list = answer_service.find_top_answers(request.text, session)
    
    # Wrap the list in the response model
    return TopAnswersResponse(answers=results_list)