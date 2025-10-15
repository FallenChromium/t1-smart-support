# based_basic_demo.py — frozen bge-m3 + shrinked prototypes + tiny LR head
import argparse
import json
import os
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from dotenv import load_dotenv
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler

import openai

load_dotenv()

ALPHA, TAU, DELTA = 0.30, 0.35, 0.05  # shrinkage & abstain gates
RANDOM_STATE = 42
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


# ---------- Load data ----------
data = pl.read_csv("data.csv", separator=";")
queries = data["text"].to_list()
sub_names_data = data["subcategory"].to_list()
cat_names_data = data["category"].to_list()

label_desc = pl.read_csv("label_desc.csv")
sub_names_tax = label_desc["subcategory"].to_list()
cat_names_tax = label_desc["category"].to_list()
label_texts = label_desc["generated_label"].to_list()

# ---------- Build ID mappings ----------
sub2id = {name: i for i, name in enumerate(sub_names_tax)}
id2sub = {i: name for name, i in sub2id.items()}

cat2id, id2cat = {}, {}
for name in cat_names_tax:
    if name not in cat2id:
        cid = len(cat2id)
        cat2id[name], id2cat[cid] = cid, name

# Persist mappings (helpful for downstream agents)
with open("subcategory_id_map.json", "w", encoding="utf-8") as f:
    json.dump({"sub2id": sub2id, "id2sub": id2sub}, f, ensure_ascii=False, indent=2)
with open("category_id_map.json", "w", encoding="utf-8") as f:
    json.dump({"cat2id": cat2id, "id2cat": id2cat}, f, ensure_ascii=False, indent=2)

# ---------- Map labels ----------
missing_subs = sorted(set(sub_names_data) - set(sub2id.keys()))
if missing_subs:
    raise ValueError(
        f"data.csv contains subcategories not in label_desc.csv: {missing_subs}"
    )

y_sub = np.asarray([sub2id[s] for s in sub_names_data], dtype=np.int32)
y_cat = np.asarray([cat2id.get(c, -1) for c in cat_names_data], dtype=np.int32)
parent_map_ids = np.asarray([cat2id[c] for c in cat_names_tax], dtype=np.int32)

N_SUB = len(sub2id)
N_CAT = len(cat2id)

# ---------- Embed strings with frozen encoder ----------
X_emb = l2(get_embeddings_batch(queries))
desc_vecs = l2(get_embeddings_batch(label_texts))


def build_protos(
    X_train: np.ndarray, y_train: np.ndarray, desc_vectors: np.ndarray
) -> np.ndarray:
    """Train-only prototypes shrunk toward label descriptions."""
    d = desc_vectors.shape[1]
    protos = np.zeros((N_SUB, d), dtype=np.float32)
    for i in range(N_SUB):
        idx = np.where(y_train == i)[0]
        base = X_train[idx].mean(0) if len(idx) else desc_vectors[i]
        protos[i] = l2((1 - ALPHA) * base + ALPHA * desc_vectors[i])
    return protos


def make_features(x_vec: np.ndarray, protos: np.ndarray) -> np.ndarray:
    """Return sub-cosines + parent max-cos + margin."""
    sim_sub = protos @ x_vec
    sim_par = np.zeros(N_CAT, dtype=np.float32)
    for c in range(N_CAT):
        mask = parent_map_ids == c
        sim_par[c] = sim_sub[mask].max() if mask.any() else 0.0
    if sim_sub.size >= 2:
        s = np.partition(sim_sub, -2)
        margin = float(s[-1] - s[-2])
    else:
        margin = 0.0
    return np.concatenate([sim_sub, sim_par, [margin]]).astype(np.float32)


def feature_matrix(indices: np.ndarray, protos: np.ndarray) -> np.ndarray:
    return np.stack([make_features(X_emb[i], protos) for i in indices], axis=0)


def apply_parent_mask(probs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    parent_scores = np.zeros((probs.shape[0], N_CAT), dtype=np.float32)
    for c in range(N_CAT):
        mask = parent_map_ids == c
        if mask.any():
            parent_scores[:, c] = probs[:, mask].max(axis=1)
    parents = parent_scores.argmax(axis=1)
    masked = probs.copy()
    for i, p in enumerate(parents):
        masked[i, parent_map_ids != p] = 0.0
    return masked, parents


def train_model(train_indices: np.ndarray):
    train_indices = np.asarray(train_indices, dtype=np.int32)
    protos = build_protos(X_emb[train_indices], y_sub[train_indices], desc_vecs)
    X_train = feature_matrix(train_indices, protos)
    scaler = StandardScaler().fit(X_train)
    clf = LogisticRegression(
        C=0.5, max_iter=2000, class_weight="balanced", multi_class="auto"
    )
    clf.fit(scaler.transform(X_train), y_sub[train_indices])
    return clf, scaler, protos


def predict_with_model(model, indices: np.ndarray):
    clf, scaler, protos = model
    indices = np.asarray(indices, dtype=np.int32)
    feats = feature_matrix(indices, protos)
    probs = clf.predict_proba(scaler.transform(feats))
    masked, parents = apply_parent_mask(probs)
    order = np.argsort(masked, axis=1)
    top1 = order[:, -1]
    top2 = order[:, -2]
    score1 = masked[np.arange(len(indices)), top1]
    score2 = masked[np.arange(len(indices)), top2]
    margin = score1 - score2
    handoff = (score1 < TAU) | (margin < DELTA)
    return {
        "top1": top1,
        "top2": top2,
        "score1": score1,
        "score2": score2,
        "margin": margin,
        "handoff": handoff,
        "parents": parents,
        "masked_probs": masked,
        "raw_probs": probs,
    }


def plot_confusion(cm: np.ndarray, labels: list[str], title: str, path: Path) -> None:
    cm = cm.astype(np.float32)
    row_sums = cm.sum(axis=1, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        cm_norm = np.divide(cm, row_sums, where=row_sums != 0)
    size = max(6.0, 0.35 * len(labels))
    fig, ax = plt.subplots(figsize=(size, size))
    im = ax.imshow(cm_norm, interpolation="nearest", cmap="Blues")
    ax.set_title(title)
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=90, ha="center", fontsize=8)
    ax.set_yticklabels(labels, fontsize=8)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def plot_category_bars(rows: list[dict], path: Path) -> None:
    labels = [row["label"] for row in rows]
    f1_vals = [row["f1"] for row in rows]
    order = np.argsort(f1_vals)
    labels_sorted = [labels[i] for i in order]
    f1_sorted = [f1_vals[i] for i in order]
    height = max(6.0, 0.3 * len(labels_sorted))
    fig, ax = plt.subplots(figsize=(10, height))
    ax.barh(labels_sorted, f1_sorted, color="#377eb8")
    ax.set_xlim(0.0, 1.0)
    ax.set_xlabel("F1 score")
    ax.set_title("Validation F1 by Subcategory")
    plt.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def compute_per_class_metrics(cm: np.ndarray, label_lookup: dict[int, str]) -> list[dict]:
    total = cm.sum()
    metrics = []
    for idx in range(cm.shape[0]):
        tp = cm[idx, idx]
        fp = cm[:, idx].sum() - tp
        fn = cm[idx, :].sum() - tp
        tn = total - tp - fp - fn
        prec_denom = tp + fp
        rec_denom = tp + fn
        precision = tp / prec_denom if prec_denom else 0.0
        recall = tp / rec_denom if rec_denom else 0.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall)
            else 0.0
        )
        accuracy = (tp + tn) / total if total else 0.0
        metrics.append(
            {
                "id": int(idx),
                "label": label_lookup[idx],
                "tp": int(tp),
                "fp": int(fp),
                "fn": int(fn),
                "tn": int(tn),
                "precision": float(precision),
                "recall": float(recall),
                "f1": float(f1),
                "accuracy": float(accuracy),
                "support": int(tp + fn),
            }
        )
    return metrics


def run_cross_validation(train_indices: np.ndarray) -> dict:
    skf = StratifiedKFold(n_splits=2, shuffle=True, random_state=RANDOM_STATE)
    macro_all, macro_conf, coverage = [], [], []
    train_indices = np.asarray(train_indices, dtype=np.int32)
    for fold_tr, fold_te in skf.split(train_indices, y_sub[train_indices]):
        tr_idx = train_indices[fold_tr]
        te_idx = train_indices[fold_te]
        model = train_model(tr_idx)
        preds = predict_with_model(model, te_idx)
        y_true = y_sub[te_idx]
        y_pred = preds["top1"]
        macro_all.append(
            f1_score(y_true, y_pred, average="macro", zero_division=0.0)
        )
        mask = ~preds["handoff"]
        if mask.any():
            macro_conf.append(
                f1_score(
                    y_true[mask], y_pred[mask], average="macro", zero_division=0.0
                )
            )
        else:
            macro_conf.append(0.0)
        coverage.append(float(mask.mean()))
    return {
        "macro_all_mean": float(np.mean(macro_all)),
        "macro_all_std": float(np.std(macro_all)),
        "macro_conf_mean": float(np.mean(macro_conf)),
        "macro_conf_std": float(np.std(macro_conf)),
        "coverage_mean": float(np.mean(coverage)),
        "coverage_std": float(np.std(coverage)),
    }


def evaluate_on_validation(
    train_indices: np.ndarray, val_indices: np.ndarray, report_dir: Path
) -> dict:
    report_dir.mkdir(parents=True, exist_ok=True)
    model = train_model(train_indices)
    preds = predict_with_model(model, val_indices)

    y_true = y_sub[val_indices]
    y_pred = preds["top1"]
    parent_true = parent_map_ids[y_true]
    parent_pred = preds["parents"]

    overall_accuracy = float(accuracy_score(y_true, y_pred))
    macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0.0
    )
    coverage = float((~preds["handoff"]).mean())

    # Predictions CSV
    pred_rows = []
    for idx_local, idx_global in enumerate(val_indices):
        sub_true = int(y_true[idx_local])
        sub_pred = int(y_pred[idx_local])
        pred_rows.append(
            {
                "row_id": int(idx_global),
                "text": queries[idx_global],
                "true_subcategory": id2sub[sub_true],
                "pred_subcategory": id2sub[sub_pred],
                "true_category": id2cat[int(parent_true[idx_local])],
                "pred_category": id2cat[int(parent_pred[idx_local])],
                "score": float(preds["score1"][idx_local]),
                "margin": float(preds["margin"][idx_local]),
                "handoff": bool(preds["handoff"][idx_local]),
            }
        )
    pl.DataFrame(pred_rows).write_csv(report_dir / "predictions_vs_ground_truth.csv")

    # Per-subcategory metrics
    cm_sub = confusion_matrix(y_true, y_pred, labels=np.arange(N_SUB))
    sub_metrics = compute_per_class_metrics(cm_sub, id2sub)
    pl.DataFrame(sub_metrics).write_csv(report_dir / "per_subcategory_metrics.csv")

    # Per-parent metrics
    cm_parent = confusion_matrix(parent_true, parent_pred, labels=np.arange(N_CAT))
    parent_metrics_raw = compute_per_class_metrics(cm_parent, id2cat)
    pl.DataFrame(parent_metrics_raw).write_csv(report_dir / "per_category_metrics.csv")

    # Plots
    plot_confusion(
        cm_parent,
        [id2cat[i] for i in range(N_CAT)],
        "Validation Confusion — Parent Categories",
        report_dir / "confusion_parent.png",
    )
    plot_category_bars(sub_metrics, report_dir / "subcategory_f1.png")

    summary = {
        "train_size": int(len(train_indices)),
        "validation_size": int(len(val_indices)),
        "overall_accuracy": overall_accuracy,
        "macro_precision": float(macro_precision),
        "macro_recall": float(macro_recall),
        "macro_f1": float(macro_f1),
        "coverage": coverage,
        "timestamp": datetime.utcnow().isoformat(timespec="seconds"),
    }
    with open(report_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    return summary


FULL_MODEL: tuple | None = None


def ensure_full_model() -> None:
    global FULL_MODEL
    if FULL_MODEL is None:
        FULL_MODEL = train_model(np.arange(len(queries)))


def predict_ticket(text: str):
    ensure_full_model()
    clf, scaler, protos = FULL_MODEL  # type: ignore[arg-type]
    x = l2(get_embeddings_batch([text]))[0]
    fv = make_features(x, protos)
    probs = clf.predict_proba(scaler.transform([fv]))[0]
    masked, parents = apply_parent_mask(probs[np.newaxis, :])
    masked = masked[0]
    parent = int(parents[0])

    order = np.argsort(masked)
    top3 = order[-3:][::-1]
    score1 = float(masked[top3[0]])
    score2 = float(masked[top3[1]]) if len(top3) > 1 else 0.0

    payload = {
        "parent_name": id2cat[parent],
        "top3_ids": [int(i) for i in top3.tolist()],
        "top3_names": [id2sub[int(i)] for i in top3.tolist()],
        "scores": [float(masked[i]) for i in top3.tolist()],
    }
    if score1 < TAU or (score1 - score2) < DELTA:
        payload["handoff_to_llm"] = True
        return payload
    payload["handoff_to_llm"] = False
    payload["subcat_id"] = int(top3[0])
    payload["subcat_name"] = id2sub[int(top3[0])]
    return payload


def run_demo() -> None:
    cv_results = run_cross_validation(np.arange(len(queries)))
    ensure_full_model()
    print(
        f"macro-F1 (all preds): {cv_results['macro_all_mean']:.3f} ± {cv_results['macro_all_std']:.3f}"
    )
    print(
        f"macro-F1 (confident subset): {cv_results['macro_conf_mean']:.3f} ± {cv_results['macro_conf_std']:.3f}"
    )
    print(
        f"coverage of confident predictions: {cv_results['coverage_mean']:.3f} ± {cv_results['coverage_std']:.3f}"
    )


def run_report(val_size: float = 0.2, random_state: int = RANDOM_STATE) -> Path:
    indices = np.arange(len(queries))
    train_idx, val_idx = train_test_split(
        indices,
        test_size=val_size,
        stratify=y_sub,
        random_state=random_state,
    )
    cv_results = run_cross_validation(train_idx)
    report_dir = (
        Path("runs")
        / "reports"
        / datetime.utcnow().strftime("%Y%m%d-%H%M%S")
    )
    summary = evaluate_on_validation(train_idx, val_idx, report_dir)

    print(
        f"[train] macro-F1 all {cv_results['macro_all_mean']:.3f} ± {cv_results['macro_all_std']:.3f}"
    )
    print(
        f"[train] macro-F1 confident {cv_results['macro_conf_mean']:.3f} ± {cv_results['macro_conf_std']:.3f}"
    )
    print(
        f"[train] coverage {cv_results['coverage_mean']:.3f} ± {cv_results['coverage_std']:.3f}"
    )
    print(
        f"[validation] accuracy {summary['overall_accuracy']:.3f} | macro-F1 {summary['macro_f1']:.3f}"
    )
    print(f"[validation] coverage {summary['coverage']:.3f}")
    print(f"Artifacts saved in {report_dir}")

    ensure_full_model()
    return report_dir


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Frozen bge-m3 baseline with reporting utilities."
    )
    parser.add_argument(
        "--mode",
        choices=("demo", "report"),
        default="demo",
        help="demo: quick CV stats; report: hold-out validation with artifacts",
    )
    parser.add_argument(
        "--val-size",
        type=float,
        default=0.2,
        help="Validation size fraction for report mode (default: 0.2).",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=RANDOM_STATE,
        help="Random seed for splitting (default: 42).",
    )
    args = parser.parse_args()

    if args.mode == "demo":
        run_demo()
    else:
        run_report(val_size=args.val_size, random_state=args.random_state)


if __name__ == "__main__":
    main()
