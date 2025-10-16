# scripts/models.py

from typing import List, Optional
from sqlmodel import Field, SQLModel
from pgvector.sqlalchemy import Vector
import sqlalchemy as sa # <-- Import sqlalchemy

class TicketData(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    text: str
    normalized_text: str = Field(sa_column=sa.Column(sa.Text, unique=False))
    subcategory: str = Field(index=True)
    category: str = Field(index=True)
    # Corrected Line: Wrap Vector in sa.Column
    embedding: List[float] = Field(sa_column=sa.Column(Vector(1024)))


class LabelDescription(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    subcategory: str = Field(unique=True, index=True)
    category: str = Field(index=True)
    generated_label: str
    # Corrected Line: Wrap Vector in sa.Column
    embedding: List[float] = Field(sa_column=sa.Column(Vector(1024)))