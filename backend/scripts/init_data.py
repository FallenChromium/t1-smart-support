# scripts/init_data.py

import polars as pl
import unicodedata
import re
import os
import time
from openai import OpenAI, APIConnectionError, RateLimitError
from sqlmodel import Session, select
from tqdm import tqdm
from dotenv import load_dotenv
load_dotenv()

from database import engine, init_db
from models import TicketData, LabelDescription

# ========== CONFIGURATION ==========
BATCH_SIZE = 32  # Adjust batch size based on your API's capacity

# ========== OPENAI API CONFIGURATION ==========
# These must be set as environment variables.
# For a local model server (like vLLM or LiteLLM), the API key may not be required.
OPENAI_API_KEY = os.getenv("SCIBOX_API_KEY", "DUMMY_KEY")
OPENAI_BASE_URL = "https://llm.t1v.scibox.tech/v1"

if not OPENAI_BASE_URL:
    raise ValueError("The OPENAI_BASE_URL environment variable is not set. Please point it to your custom embedding endpoint's base URL (e.g., 'http://embedding-api:8000/v1').")

# Initialize the OpenAI client to point to your custom endpoint
try:
    client = OpenAI(api_key=OPENAI_API_KEY, base_url=OPENAI_BASE_URL)
    print(f"OpenAI client configured for custom endpoint: {OPENAI_BASE_URL}")
except Exception as e:
    raise RuntimeError(f"Failed to initialize OpenAI client: {e}")

# ========== TEXT NORMALIZATION ==========
def normalize_text(s: str) -> str:
    if s is None: return ""
    s = unicodedata.normalize("NFKC", s)
    s = s.lower()
    s = re.sub(r"\s+", " ", s).strip()
    return s

# ========== API-BASED EMBEDDING FUNCTION ==========
def get_embeddings_from_api(texts: list[str], model_name: str = "bge-m3", retries: int = 3, delay: int = 10) -> list[list[float]]:
    """
    Gets embeddings for a batch of texts from a custom OpenAI-compatible API
    with a simple retry mechanism.
    """
    for attempt in range(retries):
        try:
            response = client.embeddings.create(input=texts, model=model_name)
            # Ensure the embeddings are sorted in the same order as the input texts
            response_dict = {item.index: item.embedding for item in response.data}
            return [response_dict[i] for i in range(len(texts))]
        except (APIConnectionError, RateLimitError) as e:
            print(f"API Error: {e}. Retrying in {delay} seconds... (Attempt {attempt + 1}/{retries})")
            time.sleep(delay)
        except Exception as e:
            print(f"An unexpected API error occurred: {e}. Aborting for this batch.")
            # Return zero vectors to prevent crashing the entire process
            return [[0.0] * 1024 for _ in texts]
    
    print(f"Failed to get embeddings after {retries} retries for batch starting with: '{texts[0][:60]}...'")
    return [[0.0] * 1024 for _ in texts]

# ========== DATA PROCESSING & DB INSERTION (Modified to use API) ==========
def process_and_insert_data(session: Session, filepath: str, is_extended: bool = False):
    print(f"Processing file: {filepath}...")
    
    df_raw = pl.read_csv(filepath, separator=";")
    if is_extended: df_raw = df_raw[262:]

    df = df_raw.with_columns(
        pl.col("text").map_elements(normalize_text, return_dtype=str).alias("normalized_text"),
        pl.col("answer_pattern").map_elements(normalize_text, return_dtype=str).alias("normalized_answer"),
        pl.col("subcategory").map_elements(normalize_text, return_dtype=str).alias("subcategory_norm"),
        pl.col("category").map_elements(normalize_text, return_dtype=str).alias("category_norm"),
    )

    existing_texts = set(session.exec(select(TicketData.normalized_text)).all())
    df = df.filter(~pl.col("normalized_text").is_in(list(existing_texts)))

    if df.height == 0:
        print(f"No new data to insert from {filepath}.")
        return

    norm_queries = df["normalized_text"].unique().to_list()
    norm_answers = df["normalized_answer"].unique().to_list()
    unique_texts_to_embed = list(set(norm_queries + norm_answers))
    
    all_embeddings = []
    print(f"Generating embeddings for {len(unique_texts_to_embed)} unique texts via API...")
    for i in tqdm(range(0, len(unique_texts_to_embed), BATCH_SIZE)):
        batch_texts = unique_texts_to_embed[i:i + BATCH_SIZE]
        batch_embeddings = get_embeddings_from_api(batch_texts)
        all_embeddings.extend(batch_embeddings)
    
    text_to_embedding_map = {text: emb for text, emb in zip(unique_texts_to_embed, all_embeddings)}

    print("Inserting data into the database...")
    for row in tqdm(df.to_dicts()):
        ticket = TicketData(
            text=row["text"], normalized_text=row["normalized_text"],
            query_embedding=text_to_embedding_map.get(row["normalized_text"], [0.0] * 1024),
            answer_pattern=row["answer_pattern"], normalized_answer=row["normalized_answer"],
            answer_embedding=text_to_embedding_map.get(row["normalized_answer"], [0.0] * 1024),
            subcategory=row["subcategory_norm"], category=row["category_norm"]
        )
        session.add(ticket)
    session.commit()
    print(f"Successfully inserted {df.height} new records from {filepath}.")

def process_and_insert_labels(session: Session, filepath: str):
    print("Processing label descriptions...")
    df = pl.read_csv(filepath).with_columns(
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
    print(f"Generating embeddings for {len(labels_to_embed)} labels via API...")
    embeddings = get_embeddings_from_api(labels_to_embed) # Reuse the same API function
    df = df.with_columns(pl.Series(name="embedding", values=embeddings))

    print("Inserting label descriptions into the database...")
    for row in tqdm(df.to_dicts()):
        label_desc = LabelDescription(
            subcategory=row["subcategory_norm"], category=row["category_norm"],
            generated_label=row["generated_label_norm"], embedding=row["embedding"]
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