# app/services.py

import numpy as np
import pickle
import unicodedata
import re
from typing import Dict, Any
import openai
import os
from dotenv import load_dotenv
load_dotenv()

# ========== UTILITIES ==========
def l2(a):
    a = np.asarray(a)
    if a.ndim == 1: return a / (np.linalg.norm(a) + 1e-9)
    return a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-9)

# ========== EMBEDDING SERVICE (for inference) ==========
class EmbeddingService:
    def __init__(self, model_name="BAAI/bge-m3"):
        self.client = openai.OpenAI(api_key=os.getenv("SCIBOX_API_KEY"), base_url="https://llm.t1v.scibox.tech/v1")
        self.model_name = "bge-m3"

    def get_embedding(self, text: str) -> np.ndarray:
        """
        Takes a single string, calls the API, and returns a single,
        normalized NumPy array, matching the expected interface.
        """
        # The API expects a list, so we wrap our single text in a list
        resp = self.client.embeddings.create(
            model=self.model_name,
            input=[text]
        )
        
        # Extract the embedding from the response data
        embedding_list = resp.data[0].embedding
        
        # Convert to a NumPy array for compatibility with the model's logic
        return np.array(embedding_list)

    

embedding_service = EmbeddingService()

# ========== PREDICTION SERVICE ==========
class PredictionService:
    def __init__(self, model_path: str):
        with open(model_path, 'rb') as f:
            model_data = pickle.load(f)
        
        self.protos = model_data['protos']
        self.parent_clf = model_data['parent_clf']
        self.scaler = model_data['scaler']
        self.id2sub = model_data['id2sub']
        self.cat2id = model_data['cat2id']
        self.id2cat = model_data['id2cat']
        self.parent_map = model_data['parent_map']
        self.N_CAT = model_data['N_CAT']
        self.CONFIDENCE_THRESHOLD = 0.50

    def extract_parent_features(self, x_emb):
        # This function is identical to the one in train.py
        features = []
        for stat_fn in [np.max, np.mean]:
            for p in range(self.N_CAT):
                mask = self.parent_map == p
                if mask.any(): features.append(stat_fn(self.protos[mask] @ x_emb))
                else: features.append(0.0)
        parent_max_sims = np.array([(self.protos[self.parent_map == p] @ x_emb).max() if (self.parent_map == p).any() else 0 for p in range(self.N_CAT)])
        if len(parent_max_sims) >= 2:
            sorted_sims = np.sort(parent_max_sims)
            features.append(sorted_sims[-1] - sorted_sims[-2])
        else: features.append(0.0)
        return np.array(features, dtype=np.float32)

    def predict_ticket(self, query_text: str) -> Dict[str, Any]:
        # Normalize and embed the incoming query
        normalized_query = unicodedata.normalize("NFKC", query_text.lower())
        normalized_query = re.sub(r"\s+", " ", normalized_query).strip()
        query_emb = embedding_service.get_embedding(normalized_query)
        
        # The rest is your original classification logic
        parent_feat = self.extract_parent_features(query_emb)
        parent_prob = self.parent_clf.predict_proba(self.scaler.transform([parent_feat]))[0]
        
        parent_pred = parent_prob.argmax()
        parent_conf = parent_prob[parent_pred]
        
        default_parent = self.cat2id.get('частные клиенты', 0)
        routed = False
        if parent_conf < self.CONFIDENCE_THRESHOLD:
            parent_pred = default_parent
            routed = True
        
        children = np.where(self.parent_map == parent_pred)[0]
        if not len(children): # Fallback if a parent has no children in the filtered taxonomy
             best_child_idx = 0 # or some other default
             child_conf = 0.0
             top3_indices = [0]
        elif len(children) == 1:
            best_child_idx = children[0]
            child_conf = 1.0
            top3_indices = [best_child_idx]
        else:
            child_sims = self.protos[children] @ query_emb
            temperature = 0.5
            exp_sims = np.exp(child_sims / temperature)
            child_probs = exp_sims / exp_sims.sum()
            
            top3_local = child_probs.argsort()[-min(3, len(children)):][::-1]
            top3_indices = children[top3_local]
            best_child_idx = top3_indices[0]
            
            best_child_local_idx = np.where(children == best_child_idx)[0][0]
            child_conf = child_probs[best_child_local_idx]
        
        true_conf =child_conf
        
        return {
            "prediction": self.id2sub[best_child_idx],
            "parent": self.id2cat[parent_pred],
            "confidence": float(true_conf),
            "parent_confidence": float(parent_conf),
            "routed": routed,
            "top3": [self.id2sub[int(x)] for x in top3_indices]
        }

prediction_service = PredictionService(model_path="final_model.pkl")