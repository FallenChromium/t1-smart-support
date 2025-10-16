import os
from typing import List

import numpy as np
import polars as pl
import torch
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from sqlalchemy import Column
from sqlmodel import Field, Session, SQLModel, create_engine, select

from pgvector.sqlalchemy import Vector

# --- Configuration and Initialization ---

load_dotenv()
DATABASE_URL = os.getenv("DATABASE_URL")
if not DATABASE_URL:
    raise ValueError("DATABASE_URL environment variable not set")

EMBEDDING_DIM = 1024  # Based on BAAI/bge-m3

# --- SQLModel Table Definitions ---


class Category(SQLModel, table=True):
    """A unified table for both parent categories and subcategories."""

    id: int = Field(default=None, primary_key=True)
    name: str = Field(unique=True, index=True)

    # Self-referencing FK. NULL for parent categories.
    parent_id: int | None = Field(default=None, foreign_key="category.id")

    # Embeddings only exist for subcategories (children).
    label_embedding: List[float] | None = Field(
        default=None, sa_column=Column(Vector(EMBEDDING_DIM))
    )


class Ticket(SQLModel, table=True):
    id: int = Field(default=None, primary_key=True)
    text: str
    # A ticket is always linked to a subcategory.
    subcategory_id: int = Field(foreign_key="category.id")
    text_embedding: List[float] = Field(sa_column=Column(Vector(EMBEDDING_DIM)))


# --- Database and Model Setup ---

engine = create_engine(DATABASE_URL)


def create_db_and_tables():
    """Initializes the database and creates all necessary tables."""
    print("Creating database tables...")
    with engine.connect() as connection:
        connection.exec_driver_sql("CREATE EXTENSION IF NOT EXISTS vector;")
        # Drop tables in reverse order of dependency for a clean slate
        SQLModel.metadata.drop_all(engine, tables=[Ticket.__table__, Category.__table__])
    SQLModel.metadata.create_all(engine)
    print("Tables created successfully.")


# --- Embedding Model ---
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using embedding device: {device}")
model_emb = SentenceTransformer("BAAI/bge-m3", device=device)


def get_embeddings_batch(texts: list[str]) -> np.ndarray:
    """Generates L2-normalized embeddings for a list of texts."""
    print(f"Generating embeddings for {len(texts)} texts...")
    return model_emb.encode(texts, normalize_embeddings=True)


# --- Data Migration Logic ---


def migrate_data():
    """Reads data from CSVs, generates embeddings, and populates the database."""
    print("Starting data migration with a unified category table...")

    data = pl.read_csv("./data_extended.csv", separator=";")
    label_desc = pl.read_csv("label_desc.csv")

    with Session(engine) as session:
        # 1. Insert Parent Categories
        print("Migrating parent categories...")
        parent_cat_names = sorted(label_desc["category"].unique().to_list())
        parent_categories = [Category(name=name) for name in parent_cat_names]
        session.add_all(parent_categories)
        session.commit()

        # Create a map from name to the new DB ID for parents
        parent_map = {cat.name: cat.id for cat in parent_categories}
        print(f"Added {len(parent_map)} parent categories.")

        # 2. Insert Subcategories
        print("Migrating subcategories...")
        subcat_rows = label_desc.to_dicts()

        labels = [row["generated_label"] for row in subcat_rows]
        embeddings = get_embeddings_batch(labels)

        subcategories = [
            Category(
                name=row["subcategory"],
                parent_id=parent_map[row["category"]],  # Link to parent ID
                label_embedding=embeddings[i].tolist(),
            )
            for i, row in enumerate(subcat_rows)
        ]
        session.add_all(subcategories)
        session.commit()

        # Create a map for all subcategories from the DB
        all_subcats_from_db = session.exec(
            select(Category).where(Category.parent_id != None)
        ).all()
        sub_map = {sub.name: sub.id for sub in all_subcats_from_db}
        print(f"Added {len(sub_map)} subcategories.")

        # 3. Insert Tickets
        print("Migrating tickets (this may take a while)...")
        ticket_texts = data["text"].to_list()
        text_embeddings = get_embeddings_batch(ticket_texts)

        tickets = [
            Ticket(
                text=row["text"],
                subcategory_id=sub_map[row["subcategory"]],
                text_embedding=text_embeddings[i].tolist(),
            )
            for i, row in enumerate(data.iter_rows(named=True))
        ]

        session.add_all(tickets)
        session.commit()
        print(f"Added {len(tickets)} tickets.")

    print("Data migration completed successfully.")


if __name__ == "__main__":
    create_db_and_tables()
    migrate_data()