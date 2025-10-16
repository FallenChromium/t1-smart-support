# scripts/models.py

from typing import List, Optional
from sqlmodel import Field, SQLModel
from pgvector.sqlalchemy import Vector  # <-- IMPORT THE VECTOR TYPE
import sqlalchemy as sa                 # <-- IMPORT SQLALCHEMY for other column types

# CORRECTED TicketData model
class TicketData(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    
    # Query related fields
    text: str
    normalized_text: str = Field(sa_column=sa.Column(sa.Text, unique=False))
    
    # CORRECTED: Explicitly define the column type as Vector
    query_embedding: List[float] = Field(sa_column=sa.Column(Vector(1024)))

    # Answer related fields
    answer_pattern: str
    normalized_answer: str
    
    # CORRECTED: Explicitly define the column type as Vector
    answer_embedding: List[float] = Field(sa_column=sa.Column(Vector(1024)))

    # Category fields
    subcategory: str = Field(index=True)
    category: str = Field(index=True)


# CORRECTED LabelDescription model
class LabelDescription(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    subcategory: str = Field(unique=True, index=True)
    category: str = Field(index=True)
    generated_label: str
    
    # CORRECTED: Explicitly define the column type as Vector
    embedding: List[float] = Field(sa_column=sa.Column(Vector(1024)))