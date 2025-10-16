# scripts/train.py

import numpy as np
import pandas as pd
import json
import pickle
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from collections import Counter
from sqlmodel import Session, select
import warnings

from database import engine
from models import TicketData, LabelDescription

warnings.filterwarnings('ignore')

# ... (Keep your original CONFIG and UTILITIES sections)
# ========== CONFIG ==========
MIN_CLASS_SUPPORT = 5
ALPHA = 0.20
BETA = 0.25
CONFIDENCE_THRESHOLD = 0.50

# ========== UTILITIES ==========
def l2(a):
    a = np.asarray(a)
    if a.ndim == 1: return a / (np.linalg.norm(a) + 1e-9)
    return a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-9)

def main():
    with Session(engine) as session:
        # ========== DATA LOADING from DB ==========
        print("="*80 + "\nLOADING DATA FROM DATABASE\n" + "="*80)
        
        stmt = select(TicketData.subcategory, TicketData.embedding)
        all_data = session.exec(stmt).all()
        y_all = [row[0] for row in all_data]
        X_all = [row[1] for row in all_data]

        X_train, X_test, y_train, y_test = train_test_split(
            X_all, y_all, test_size=0.3, random_state=42, stratify=y_all
        )
        print(f"Total data: {len(y_all)}, Train: {len(X_train)}, Test: {len(X_test)}")

        # ========== TAXONOMY from DB ==========
        print("\n" + "="*80 + "\nLOADING TAXONOMY FROM DATABASE\n" + "="*80)
        
        stmt = select(LabelDescription.subcategory, LabelDescription.category, LabelDescription.generated_label, LabelDescription.embedding)
        label_data = session.exec(stmt).all()
        
        sub_names_tax = [row[0] for row in label_data]
        cat_names_tax = [row[1] for row in label_data]
        label_texts = [row[2] for row in label_data] # Not used for embeddings anymore, just for reference
        desc_vecs = np.array([row[3] for row in label_data]) # Use pre-computed embeddings
        
        sub2cat = {s: c for s, c in zip(sub_names_tax, cat_names_tax)}
        
        # ... (The rest of your training script from "Drop toxic classes" onwards is almost identical)
        # Drop toxic classes
        class_counts = Counter(y_train)
        toxic = {cls for cls, cnt in class_counts.items() if cnt < MIN_CLASS_SUPPORT}

        if toxic:
            print(f"Dropping {len(toxic)} toxic classes")
            # This filtering logic needs to be careful with list indices
            train_indices_to_keep = [i for i, y in enumerate(y_train) if y not in toxic]
            X_train = [X_train[i] for i in train_indices_to_keep]
            y_train = [y_train[i] for i in train_indices_to_keep]

            test_indices_to_keep = [i for i, y in enumerate(y_test) if y not in toxic]
            X_test = [X_test[i] for i in test_indices_to_keep]
            y_test = [y_test[i] for i in test_indices_to_keep]

            valid_tax_indices = [i for i, s in enumerate(sub_names_tax) if s not in toxic]
            sub_names_tax = [sub_names_tax[i] for i in valid_tax_indices]
            cat_names_tax = [cat_names_tax[i] for i in valid_tax_indices]
            desc_vecs = desc_vecs[valid_tax_indices]

        # Build mappings
        sub2id = {name: i for i, name in enumerate(sub_names_tax)}
        id2sub = {i: name for name, i in sub2id.items()}
        cat2id, id2cat = {}, {}
        for name in cat_names_tax:
            if name not in cat2id:
                cid = len(cat2id)
                cat2id[name], id2cat[cid] = cid, name

        y_sub_train = np.array([sub2id[s] for s in y_train], dtype=np.int32)
        y_cat_train = np.array([cat2id[sub2cat[s]] for s in y_train], dtype=np.int32)
        y_sub_test = np.array([sub2id[s] for s in y_test], dtype=np.int32)
        
        parent_map = np.array([cat2id[sub2cat[s]] for s in sub_names_tax], dtype=np.int32)
        N_SUB, N_CAT = len(sub2id), len(cat2id)
        print(f"Taxonomy: {N_SUB} subcategories, {N_CAT} parents")
        
        # ========== EMBEDDINGS ==========
        X_train_emb = np.array(X_train) # Already L2 normalized
        X_test_emb = np.array(X_test)   # Already L2 normalized
        print(f"Embeddings: train={X_train_emb.shape}, test={X_test_emb.shape}")
        
        # ... (Your "CONTRASTIVE PROTOTYPES" and "PARENT CLASSIFIER" sections)
        # ========== CONTRASTIVE PROTOTYPES ==========
        print("\n" + "="*80 + "\nBUILDING PROTOTYPES\n" + "="*80)
        def build_protos(X_arr, y_arr):
            protos = np.zeros((N_SUB, desc_vecs.shape[1]), dtype=np.float32)
            for i in range(N_SUB):
                idx = np.where(y_arr == i)[0]
                positive = X_arr[idx].mean(0) if len(idx) > 0 else desc_vecs[i]
                parent_id = parent_map[i]
                siblings = [s for s in range(N_SUB) if parent_map[s] == parent_id and s != i]
                if siblings and len(idx) > 0:
                    sib_samples = [X_arr[np.where(y_arr == sib)[0]] for sib in siblings if len(np.where(y_arr == sib)[0]) > 0]
                    if sib_samples:
                        negative = np.mean([s.mean(0) for s in sib_samples], axis=0)
                        protos[i] = l2((1 + BETA) * positive - BETA * negative + ALPHA * desc_vecs[i])
                    else:
                        protos[i] = l2((1 - ALPHA) * positive + ALPHA * desc_vecs[i])
                else:
                    protos[i] = l2((1 - ALPHA) * positive + ALPHA * desc_vecs[i])
            return protos
        protos = build_protos(X_train_emb, y_sub_train)

        # ========== PARENT CLASSIFIER ==========
        print("\n" + "="*80 + "\nTRAINING PARENT CLASSIFIER\n" + "="*80)
        def extract_parent_features(x_emb):
            features = []
            for stat_fn in [np.max, np.mean]:
                for p in range(N_CAT):
                    mask = parent_map == p
                    if mask.any(): features.append(stat_fn(protos[mask] @ x_emb))
                    else: features.append(0.0)
            parent_max_sims = np.array([(protos[parent_map == p] @ x_emb).max() if (parent_map == p).any() else 0 for p in range(N_CAT)])
            if len(parent_max_sims) >= 2:
                sorted_sims = np.sort(parent_max_sims)
                features.append(sorted_sims[-1] - sorted_sims[-2])
            else: features.append(0.0)
            return np.array(features, dtype=np.float32)

        X_parent_train = np.stack([extract_parent_features(x) for x in X_train_emb])
        X_parent_test = np.stack([extract_parent_features(x) for x in X_test_emb])
        scaler = StandardScaler().fit(X_parent_train)
        X_parent_train_scaled = scaler.transform(X_parent_train)
        parent_clf = LogisticRegression(C=5.0, max_iter=3000, class_weight='balanced', random_state=42)
        parent_clf.fit(X_parent_train_scaled, y_cat_train)
        print("Parent classifier trained.")

        # ... (Your "EVALUATION" section can be here for checking performance)
        print(f"Parent classifier train acc: {accuracy_score(y_cat_train, parent_clf.predict(X_parent_train_scaled)):.3f}")

        # ========== SAVE MODEL ==========
        print("\n" + "="*80 + "\nSAVING MODEL\n" + "="*80)
        model_data = {
            'protos': protos, 'parent_clf': parent_clf, 'scaler': scaler,
            'sub2id': sub2id, 'id2sub': id2sub, 'cat2id': cat2id, 'id2cat': id2cat,
            'parent_map': parent_map, 'N_CAT': N_CAT
        }
        with open('final_model.pkl', 'wb') as f:
            pickle.dump(model_data, f)
        with open('mappings.json', 'w', encoding='utf-8') as f:
            json.dump({'sub2id': sub2id, 'id2sub': id2sub, 'cat2id': cat2id, 'id2cat': id2cat}, f, ensure_ascii=False, indent=2)
        print("Model 'final_model.pkl' and 'mappings.json' saved successfully.")

if __name__ == "__main__":
    main()