# based_basic_demo.py  — frozen bge-m3 + shrinked prototypes + tiny LR head (string labels safe)
import polars as pl
import numpy as np
import json
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
import openai, os
from dotenv import load_dotenv

load_dotenv()

ALPHA, TAU, DELTA = 0.30, 0.35, 0.05  # shrinkage & abstain gates
client = openai.OpenAI(
    api_key=os.getenv("SCIBOX_API_KEY"), base_url="https://llm.t1v.scibox.tech/v1"
)


def get_embeddings_batch(texts: list[str], model: str = "bge-m3") -> np.ndarray:
    resp = client.embeddings.create(model=model, input=texts)
    return np.asarray([d.embedding for d in resp.data], dtype=np.float32)


def l2(a: np.ndarray) -> np.ndarray:
    a = np.asarray(a)
    if a.ndim == 1:
        n = np.linalg.norm(a) + 1e-9
        return a / n
    if a.ndim == 2:
        n = np.linalg.norm(a, axis=1, keepdims=True) + 1e-9
        return a / n
    raise ValueError("l2() expects 1-D or 2-D array")


# ---------- Load data (strings allowed) ----------
# data.csv must have columns: text; subcategory; category  (all strings for cat/subcat are OK)
data = pl.read_csv("data.csv", separator=";")
queries = data["text"].to_list()
sub_names_data = data["subcategory"].to_list()
cat_names_data = data["category"].to_list()

# label_desc.csv must have: subcategory; category; generated_label (all strings)
label_desc = pl.read_csv("label_desc.csv")
sub_names_tax = label_desc["subcategory"].to_list()
cat_names_tax = label_desc["category"].to_list()
label_texts = label_desc["generated_label"].to_list()

# ---------- Build ID mappings from taxonomy (authoritative) ----------
# Subcategory mapping (order follows label_desc rows)
sub2id = {name: i for i, name in enumerate(sub_names_tax)}
id2sub = {i: name for name, i in sub2id.items()}

# Category mapping (contiguous 0..C-1 in order of first occurrence in label_desc)
cat2id, id2cat = {}, {}
for name in cat_names_tax:
    if name not in cat2id:
        cid = len(cat2id)
        cat2id[name], id2cat[cid] = cid, name

# Save mappings (optional but useful)
with open("subcategory_id_map.json", "w", encoding="utf-8") as f:
    json.dump({"sub2id": sub2id, "id2sub": id2sub}, f, ensure_ascii=False, indent=2)
with open("category_id_map.json", "w", encoding="utf-8") as f:
    json.dump({"cat2id": cat2id, "id2cat": id2cat}, f, ensure_ascii=False, indent=2)

# ---------- Map dataset labels to ints (this fixes your error) ----------
# Ensure every data subcategory is in taxonomy
missing_subs = sorted(set(sub_names_data) - set(sub2id.keys()))
if missing_subs:
    raise ValueError(
        f"data.csv contains subcategories not in label_desc.csv: {missing_subs}"
    )

y_sub = np.asarray([sub2id[s] for s in sub_names_data], dtype=np.int32)  # 0..N_SUB-1
y_cat = np.asarray(
    [cat2id.get(c, -1) for c in cat_names_data], dtype=np.int32
)  # may be -1 if data has extra cats

# Parent map per subcategory (aligned to sub IDs order from label_desc)
parent_map_ids = np.asarray([cat2id[c] for c in cat_names_tax], dtype=np.int32)

N_SUB = len(sub2id)
N_CAT = len(cat2id)

# ---------- Embed everything (frozen) ----------
X_emb = l2(get_embeddings_batch(queries))
desc_vecs = l2(
    get_embeddings_batch(label_texts)
)  # one vector per subcategory (same order as sub2id)


# ---------- Prototype + features ----------
def build_protos(
    X_train: np.ndarray, y_train: np.ndarray, desc_vecs: np.ndarray
) -> np.ndarray:
    """Shrinked prototype per sub (train-only stats)."""
    d = desc_vecs.shape[1]
    protos = np.zeros((N_SUB, d), dtype=np.float32)
    for i in range(N_SUB):
        idx = np.where(y_train == i)[0]
        base = X_train[idx].mean(0) if len(idx) else desc_vecs[i]
        protos[i] = l2((1 - ALPHA) * base + ALPHA * desc_vecs[i])
    return protos


def make_features(
    x_vec: np.ndarray, protos: np.ndarray, parent_map: np.ndarray
) -> np.ndarray:
    """[35 sub-cos] + [N_CAT parent max-cos] + [margin]  → compact, stable features."""
    sim_sub = protos @ x_vec
    sim_par = np.zeros(N_CAT, dtype=np.float32)
    for c in range(N_CAT):
        mask = parent_map == c
        sim_par[c] = sim_sub[mask].max() if mask.any() else 0.0
    if sim_sub.size >= 2:
        # faster/safer than full sort
        s = np.partition(sim_sub, -2)
        margin = s[-1] - s[-2]
    else:
        margin = 0.0
    return np.concatenate([sim_sub, sim_par, [margin]]).astype(np.float32)


# ---------- Honest CV (2 folds due to tiny classes) ----------
skf = StratifiedKFold(n_splits=2, shuffle=True, random_state=42)
f1s = []
for tr, te in skf.split(X_emb, y_sub):
    P_tr = build_protos(X_emb[tr], y_sub[tr], desc_vecs)

    X_tr = np.stack([make_features(X_emb[i], P_tr, parent_map_ids) for i in tr])
    X_te = np.stack([make_features(X_emb[i], P_tr, parent_map_ids) for i in te])

    scaler = StandardScaler().fit(X_tr)
    clf = LogisticRegression(
        C=0.5, max_iter=2000, class_weight="balanced"
    )  # multi_class default (multinomial) is fine
    clf.fit(scaler.transform(X_tr), y_sub[tr])

    # probs and hierarchy masking
    raw = clf.predict_proba(scaler.transform(X_te))  # (n_te, N_SUB)
    parent_scores = np.zeros((len(te), N_CAT), dtype=np.float32)
    for c in range(N_CAT):
        mask = parent_map_ids == c
        parent_scores[:, c] = raw[:, mask].max(1) if mask.any() else 0.0
    parents_hat = parent_scores.argmax(1)

    masked = raw.copy()
    for i, p in enumerate(parents_hat):
        masked[i, parent_map_ids != p] = 0.0

    top1 = masked.argmax(1)
    # robust top2 (handle branches with 1 child)
    part = np.partition(masked, -2, axis=1)
    sec = part[:, -2]
    top2 = np.where(sec > 0, np.argsort(masked, axis=1)[:, -2], top1)

    score1 = masked[np.arange(len(te)), top1]
    margin = score1 - masked[np.arange(len(te)), top2]
    good = (score1 >= TAU) & (margin >= DELTA)

    f1s.append(f1_score(y_sub[te][good], top1[good], average="macro", zero_division=0))

print(f"macro-F1 (confident subset): {np.mean(f1s):.3f} ± {np.std(f1s):.3f}")


# ---------- Inference ----------
def predict_ticket(text: str):
    x = l2(get_embeddings_batch([text]))[0]
    P = build_protos(X_emb, y_sub, desc_vecs)  # use all data for final protos
    fv = make_features(x, P, parent_map_ids)
    z = clf.predict_proba(scaler.transform([fv]))[0]

    # parent choose + mask
    p_scores = np.array(
        [
            z[parent_map_ids == c].max() if (parent_map_ids == c).any() else 0.0
            for c in range(N_CAT)
        ],
        dtype=np.float32,
    )
    parent = int(p_scores.argmax())
    z[parent_map_ids != parent] = 0.0

    top3 = z.argsort()[-3:][::-1]
    score1 = float(z[top3[0]])
    score2 = float(z[top3[1]]) if len(top3) > 1 else 0.0
    payload = {
        "parent_name": id2cat[parent],
        "top3_ids": [int(i) for i in top3.tolist()],
        "top3_names": [id2sub[int(i)] for i in top3.tolist()],
        "scores": [float(z[i]) for i in top3.tolist()],
    }
    if score1 < TAU or (score1 - score2) < DELTA:
        payload["handoff_to_llm"] = True
        return payload
    payload["handoff_to_llm"] = False
    payload["subcat_id"] = int(top3[0])
    payload["subcat_name"] = id2sub[int(top3[0])]
    return payload


if __name__ == "__main__":
    print(predict_ticket("Не могу поменять пароль, пишет ошибка E1234"))
