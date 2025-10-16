# app/schemas.py

from pydantic import BaseModel
from typing import List

class PredictionRequest(BaseModel):
    text: str

class PredictionResponse(BaseModel):
    prediction: str
    parent: str
    confidence: float
    parent_confidence: float
    routed: bool
    top3: List[str]

class AnswerRequest(BaseModel):
    text: str

# 1. Define the schema for a single retrieved answer item
class RetrievedAnswerItem(BaseModel):
    retrieved_answer: str
    matched_query: str
    similarity_score: float

# 2. Define the main response schema which will contain a list of these items
class TopAnswersResponse(BaseModel):
    answers: List[RetrievedAnswerItem]