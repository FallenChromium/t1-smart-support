# AGENTS.md — Frozen-Embedding Classification Lab (bge-m3, ≤150 dims)

### Objective
Predict 35 sub-categories (and 6 parents) from 201 FAQ tickets with **frozen bge-m3**. Keep the model honest, data-efficient, and deployable.

---

## Agents & Responsibilities
| Agent | Key Ops | Guardrails |
|-------|---------|------------|
| **DataAgent** | load csv; preprocess: lowercase, keep mixed-script, strip emoji; add regex flags (`has_latin_caps_token`, `has_errorcode`, `starts_with_how`, …); compute `len_tokens_clipped`. | No stats from valid fold. |
| **EmbedAgent** | produce & cache L2 bge-m3 embeddings; embed label-descs & frozen synonyms. | Encoder **never** tuned. |
| **ProtoAgent** | centroids/medoids; shrink to label-desc (α); rival-push (λ); optional Ledoit-Wolf whitener. | Stats fold-local. |
| **FeatureAgent** | build ≤150-dim stack: proto-cos (35), label-cos (35), parent (6), margin (1), length (1), regex flags (≤10), TF-IDF⊗label top-5 (5), optional LLM tags (≤10), synonym-cos top-5 (5). | No raw 768-dim embeddings. |
| **ModelAgent** | LogisticRegression / Ridge / LinearSVC+calib. | Hyper-params via inner CV. |
| **HierarchyAgent** | parent-first pick, mask sub-cats, β bonus. | No cross-branch picks. |
| **AbstainAgent** | margin/τ + conformal calibration; output coverage-F1. | Thresholds fit on train. |
| **LLMAgent** | Gated Qwen calls: L1 rerank, L2 tags, L3 synonyms (offline), L4 parent vote, L5 rewrite (short/noisy), L6 tiny synth tails. | All calls logged; synonyms frozen ex-ante. |
| **Metrics&VizAgent** | macro-F1, cat-F1, F1@coverage, ECE, confusion, LR coeffs, NN inspection, UMAP. | Variance across folds reported. |

---

### Feature Toggles (default)
- `use_priority` = **False** (one-hot optional).
- `use_audience` = **False** (diagnostic only).
- `use_projection_head` = **False** (enable only if inner CV ΔF1 > 1.5).
- `use_Mahalanobis` = **False** by default (enable `H5_safe` only if Σ well-conditioned).

---

### Primary Hypotheses to run
1. **B0 / B1 / B2** — sanity floor
2. **H1 + H6** — shrinkage + hierarchy
3. **H2** — similarity-stacked LogReg (+regex, length)
4. **H3** — whitened version
5. **L1 + L3** — gated rerank & synonym-cos
6. **H5_safe** — Mahalanobis (optional)
7. **L6** — tiny synth tails (if tail F1 < 10 %)

Everything else is off unless CV justifies.

Visualize and plot main metrics for easy comparison of hypotheses. top@1 and top-3 acc are both interesting, but top-1 acc is more important.
