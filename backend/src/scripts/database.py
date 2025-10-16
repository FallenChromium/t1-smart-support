# scripts/database.py

from sqlmodel import create_engine, Session, SQLModel
import os

DATABASE_URL = os.getenv("DATABASE_URL", "postgresql+psycopg2://postgres:password@localhost:5432/example_db")

engine = create_engine(DATABASE_URL)

def get_session():
    with Session(engine) as session:
        yield session

def init_db():
    print("Initializing database and creating tables...")
    SQLModel.metadata.create_all(engine)
    print("Database initialized.")
