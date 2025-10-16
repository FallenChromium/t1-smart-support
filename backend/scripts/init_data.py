import polars as pl
import unicodedata
import re
import openai
from sqlmodel import Session, select
from tqdm import tqdm

from database import engine, init_db
from models import TicketData, LabelDescription

# ========== CONFIG ==========
BATCH_SIZE = 32
EMBEDDING_MODEL = "BAAI/bge-m3"

# ========== TEXT NORMALIZATION ==========
def normalize_text(s: str) -> str:
    if s is None: return ""
    s = unicodedata.normalize("NFKC", s)
    s = s.lower()
    s = re.sub(r"\s+", " ", s).strip()
    return s

# ========== EMBEDDING MODEL SETUP ==========
client = openai.OpenAI(api_key=os.getenv("SCIBOX_API_KEY"), base_url="https://llm.t1v.scibox.tech/v1")

def get_embeddings_batch(texts: list[str], model: str = "bge-m3") -> list[list[float]]:
    resp = client.embeddings.create(
        model=model,
        input=texts
    )
    embs = [d.embedding for d in resp.data]
    return embs

# ========== DATA PROCESSING & DB INSERTION ==========
def process_and_insert_data(session: Session, filepath: str, is_extended: bool = False):
    print(f"Processing file: {filepath}...")
    
    # Load and normalize data using Polars
    df_raw = pl.read_csv(filepath, separator=";")
    if is_extended:
        df_raw = df_raw[262:]

    df = df_raw.with_columns(
        pl.col("text").map_elements(normalize_text, return_dtype=str).alias("normalized_text"),
        pl.col("subcategory").map_elements(normalize_text, return_dtype=str).alias("subcategory_norm"),
        pl.col("category").map_elements(normalize_text, return_dtype=str).alias("category_norm"),
    )

    # Check for existing entries to avoid duplicates
    existing_texts = set(session.exec(select(TicketData.normalized_text)).all())
    df = df.filter(~pl.col("normalized_text").is_in(list(existing_texts)))

    if df.height == 0:
        print(f"No new data to insert from {filepath}.")
        return

    # Generate embeddings in batches
    all_texts = df["normalized_text"].to_list()
    all_embeddings = []
    print("Generating text embeddings...")
    for i in tqdm(range(0, len(all_texts), BATCH_SIZE)):
        batch_texts = all_texts[i:i + BATCH_SIZE]
        batch_embeddings = get_embeddings_batch(batch_texts)
        all_embeddings.extend(batch_embeddings)
        
    df = df.with_columns(pl.Series(name="embedding", values=all_embeddings))

    # Insert data into the database
    print("Inserting data into the database...")
    for row in tqdm(df.to_dicts()):
        ticket = TicketData(
            text=row["text"],
            normalized_text=row["normalized_text"],
            subcategory=row["subcategory_norm"],
            category=row["category_norm"],
            embedding=row["embedding"]
        )
        session.add(ticket)
    session.commit()
    print(f"Successfully inserted {df.height} new records from {filepath}.")

def process_and_insert_labels(session: Session, filepath: str):
    print("Processing label descriptions...")
    df = pl.read_csv(filepath)
    df = df.with_columns(
        pl.col("subcategory").map_elements(normalize_text).alias("subcategory_norm"),
        pl.col("category").map_elements(normalize_text).alias("category_norm"),
        pl.col("generated_label").map_elements(normalize_text).alias("generated_label_norm"),
    )

    existing_labels = set(session.exec(select(LabelDescription.subcategory)).all())
    df = df.filter(~pl.col("subcategory_norm").is_in(list(existing_labels)))

    if df.height == 0:
        print("No new label descriptions to insert.")
        return

    labels_to_embed = df["generated_label_norm"].to_list()
    print("Generating label embeddings...")
    embeddings = get_embeddings_batch(labels_to_embed)
    df = df.with_columns(pl.Series(name="embedding", values=embeddings))

    print("Inserting label descriptions into the database...")
    for row in tqdm(df.to_dicts()):
        label_desc = LabelDescription(
            subcategory=row["subcategory_norm"],
            category=row["category_norm"],
            generated_label=row["generated_label_norm"],
            embedding=row["embedding"]
        )
        session.add(label_desc)
    session.commit()
    print(f"Successfully inserted {df.height} new label descriptions.")


if __name__ == "__main__":
    init_db()
    with Session(engine) as session:
        process_and_insert_labels(session, "./label_desc.csv")
        process_and_insert_data(session, "./data.csv", is_extended=False)
        process_and_insert_data(session, "./data_extended.csv", is_extended=True)
    print("Data initialization complete.")