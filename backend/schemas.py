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