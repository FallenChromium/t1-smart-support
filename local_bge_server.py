import numpy as np
import polars as pl
import re
import torch
import unicodedata
from fastapi import FastAPI, Request
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
from sklearn.neighbors import NearestNeighbors

app = FastAPI()

def normalize_text(text: str | None) -> str:
    if text is None:
        return ""
    text = unicodedata.normalize("NFKC", text).lower()
    return re.sub(r"\s+", " ", text).strip()

def l2_normalize(vectors: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=1, keepdims=True) + 1e-9
    return vectors / norms

def load_model_and_data():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    model = SentenceTransformer("BAAI/bge-m3", device=device)

    df = pl.read_csv("./data.csv", separator=";").with_columns(
        pl.col("query_text").map_elements(normalize_text).alias("query_norm"),
        pl.col("category_fine").map_elements(normalize_text).alias("cat_fine_norm"),
    )

    unique_queries_df = df.select("query_norm").unique()
    unique_queries = unique_queries_df["query_norm"].to_list()
    unique_embeddings = model.encode(unique_queries)

    embeddings_df = pl.DataFrame({
        "query_norm": unique_queries,
        "query_emb": [list(emb) for emb in unique_embeddings]
    })

    df = df.join(embeddings_df, on="query_norm")
    
    train_embeddings = np.array(df["query_emb"].to_list())
    train_labels = np.array(df["cat_fine_norm"].to_list())

    normalized_embeddings = l2_normalize(train_embeddings)
    knn = NearestNeighbors(n_neighbors=1, metric="cosine")
    knn.fit(normalized_embeddings)
    
    return model, knn, train_labels

model, knn, train_labels = load_model_and_data()

class PredictRequest(BaseModel):
    input: str

@app.post("/predict")
async def predict(request: PredictRequest):
    normalized_input = normalize_text(request.input)
    input_embedding = model.encode([normalized_input])
    normalized_embedding = l2_normalize(input_embedding)
    
    distances, indices = knn.kneighbors(normalized_embedding, return_distance=True)
    predicted_label = train_labels[indices[0, 0]]
    
    return {"predict": predicted_label}