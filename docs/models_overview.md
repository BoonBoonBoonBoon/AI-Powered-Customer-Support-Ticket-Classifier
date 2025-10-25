# Models and Tools Overview (Beginner-Friendly)

This guide explains the modeling options in this project, what each does, how they differ, typical scores we observed, and when to use them.

## Quick glossary
- Macro‑F1: Average F1 across classes (treats all classes equally). Good when classes are imbalanced.
- Accuracy: Percent of correct predictions overall. Can be misleading with class imbalance.
- Embedding: Turning text into numeric vectors that capture meaning.
- Fine‑tuning: Training a large language model (like BERT/Roberta) on your task.

---

## 1) Sklearn Baseline (TF‑IDF + Linear Models)
- What it is: Classic bag‑of‑words features (TF‑IDF) feeding simple linear classifiers for each task:
  - Priority (Urgent/High/Medium/Low)
  - Department (Billing/Sales/Tech Support)
- Why use it: 
  - Trains in seconds on CPU
  - Very small model and fast to serve
  - Stable under class imbalance (no collapse)
- Where: `models/v1.x/` (e.g., `models/v1.0.10/`)
- Typical validation scores (example: v1.0.10):
  - Priority: macro‑F1 ≈ 0.257, accuracy ≈ 0.257
  - Department: macro‑F1 ≈ 0.318, accuracy ≈ 0.393
- When to choose: As a strong, reliable starting point and production fallback when GPU is unavailable.
- Potential: Can be nudged up a bit with smarter n‑grams, features, and regularization—but has a lower ceiling than transformer approaches.

---

## 2) SBERT Embeddings + Sklearn Heads (CPU‑friendly middle path)
- What it is: Use a pre‑trained SentenceTransformer (MiniLM) to embed Title+Description into semantic vectors; then train two lightweight LogisticRegression classifiers.
- Why use it:
  - Still fast on CPU (minutes)
  - Usually stronger than TF‑IDF for semantic tasks
  - Smaller and faster than full transformer fine‑tuning
- Where: `scripts/train_sbert_sklearn.py` produces `models/sbert/<version>/` (e.g., `models/sbert/s1.0.0/`).
- Example scores (s1.0.0 with all‑MiniLM‑L6‑v2):
  - Priority: macro‑F1 ≈ 0.234, accuracy ≈ 0.234
  - Department: macro‑F1 ≈ 0.837, accuracy ≈ 0.860
- Interpretation:
  - Department strongly improves vs baseline (great semantic separation)
  - Priority stays tough (class semantics likely subtle or data requires richer training)
- When to choose: If you need a quick CPU upgrade for department and want a tiny model with good latency.
- Potential:
  - Try a larger encoder (e.g., `all‑MiniLM‑L12‑v2`) for a small boost (slower)
  - Swap LogisticRegression for LinearSVC or tune C to combat class overlap
  - Add engineered features (length, keywords) alongside embeddings

---

## 3) Transformer Fine‑Tuning (Dual‑Head, HuggingFace)
- What it is: End‑to‑end fine‑tuning of a base encoder (DistilBERT/Roberta) with two heads: one for Priority, one for Department.
- Why use it:
  - Highest ceiling with enough data/training time
  - Learns task‑specific patterns directly from text
- Where: `scripts/train_transformer.py` writes to `models/transformers/t1.x/`.
- Training recipe highlights:
  - Label smoothing (stable)
  - Class weights (to help imbalance)
  - Department loss warmup (avoid early collapse)
  - Selection by priority macro‑F1
- Example scores (local CPU/GPU‑limited trials):
  - t1.0.3 (DistilBERT, 3 epochs): Priority macro‑F1 ≈ 0.235; Department macro‑F1 ≈ 0.251 (accuracy appears high due to collapse)
  - t1.0.4 (tuned): Priority macro‑F1 ≈ 0.167; Department macro‑F1 ≈ 0.293 (partial collapse)
  - t1.0.5r (roberta attempt): Priority macro‑F1 ≈ 0.111; Department macro‑F1 ≈ 0.112 (strong collapse)
- Interpretation:
  - On CPU or short runs, transformers can underperform and sometimes collapse to predicting one class
  - With a stable recipe and a GPU (e.g., in Colab), they can approach/beat baseline, but need careful tuning
- When to choose: When you have GPU or can tolerate longer training; best long‑term upside.
- Potential:
  - Use GPU (Colab/AWS/Local) and train 5–7+ epochs with DistilRoBERTa/Roberta
  - Keep simplified loss (no focal/weighted sampler) + class weights + label smoothing
  - Hyper‑parameter sweep: LR (2e‑5..3e‑5), batch size, warmup, max length

---

## Tools you can run
- Training
  - `scripts/train_transformer.py` — full fine‑tune; outputs `models/transformers/t*`
  - `scripts/train_sbert_sklearn.py` — quick CPU baseline+ semantic boost; outputs `models/sbert/*`
- Evaluation & comparison
  - `scripts/compare_models.py` — collect/compare metrics across versions
  - `scripts/error_analysis.py` — dig into mistakes per class
- Registry & serving
  - `scripts/emit_manifest.py` — creates a manifest pointing to any local model folder
  - FastAPI app (`app/`) — loads model via manifest and serves predictions
- Docs
  - `docs/colab_quickstart.md` — GPU training in Colab
  - `docs/local_cpu_quickstart.md` — SBERT + sklearn CPU quickstart

---

## Which should I use?
- Need something reliable today on CPU: start with the Sklearn baseline or SBERT+sklearn.
- Department is the priority task: SBERT+sklearn shows a big jump on department.
- Highest long‑term ceiling (both tasks): go Transformer fine‑tuning, but plan for GPU and tuning time.

---

## Reading the metrics files
Every trained run writes a `metrics.json` with, for each task:
- `macro_f1`: main quality metric (higher is better)
- `accuracy`: overall correctness
- `report`: per‑class precision/recall/F1

Look under `models/<family>/<version>/metrics.json` to compare.

---

## Practical tips
- Class imbalance: favor macro‑F1 for selection; use class weights for transformers and balanced class_weight for sklearn.
- Text prep: concatenating Title + Description generally works well. Consider removing obvious department hints if it leaks labels.
- Reproducibility: keep versioned folders (`v1.x`, `t1.x`, `s1.x`) and update the manifest only when a new model beats your gate.

If you want, I can run a quick SBERT experiment with a larger encoder or swap the classifier and add the best one to staging.
