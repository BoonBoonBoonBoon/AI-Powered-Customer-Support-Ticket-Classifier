# AI-Powered Customer Support Ticket Classifier — Case Study

Last updated: 2025-10-24

## Executive summary
We built an end-to-end system that classifies customer tickets by priority (Urgent/High/Medium/Low) and department. We started with classical ML (scikit-learn), hardened the pipeline for reproducibility and leakage, and then pivoted to a transformer model (DistilBERT) with a dual-head classifier. The project includes a FastAPI service, versioned model registry, training scripts, and comparison tooling.

Highlights:
- Production baseline (sklearn v1.0.10) sets a strong reference on validation.
- Early transformer attempts showed leakage (unrealistically perfect department scores) → we diagnosed and fixed it with exclusion guards.
- Post-fix transformer (t1.0.3) improved significantly vs the canary but still trails the sklearn baseline on priority. We’re iterating with better loss weighting, input formatting, and training schedule (t1.0.4).

## Problem and goals
- Problem: Given ticket title/description, predict Priority and Department.
- Goals:
  - High-quality priority classification (macro-F1) without leakage.
  - Reproducible training and fair comparisons across versions.
  - Safe serving via a registry (staging vs production) and readiness checks.

## Data and leakage handling
- Dataset: ~8.5k enriched tickets with title, description, priority, department.
- Leakage discovered: Enrichment added tokens like “__type_*” that revealed department.
- Mitigation: Regex-based exclusion during training and inference for parity.

## Architecture overview
- API: FastAPI service exposing /classify and health endpoints.
- Routing: Environment flag to route either sklearn or transformer by default.
- Model registry: JSON pointers for staging/production with local file:// URIs.
- Manifests: Per-model manifest.json capturing artifacts, training params, and metrics.
- Training: Dedicated scripts for sklearn and transformers with logs and metrics.
- Tooling: Scripts to compare models and emit manifests; docs for reproducibility.

Key files:
- `app/main.py` — FastAPI endpoints and routing
- `app/models/runtime_loader.py` — Loads artifacts via registry/manifests (Windows-safe file://)
- `app/models/inference_transformer.py` — Transformer inference with leak guards
- `scripts/train_transformer.py` — Dual-head trainer (class weights, label smoothing, selection metric, loss weights)
- `scripts/run_t103.ps1`, `scripts/run_t104.ps1` — End-to-end training pipelines with logs
- `scripts/compare_models.py` — Dynamic comparison across versions
- `models/registry/*.json` — staging/production pointers

## Baseline to transformer: metrics evolution (validation)
- Sklearn v1.0.10
  - Priority macro-F1 0.2570 | Accuracy 0.2574
  - Department macro-F1 0.3179 | Accuracy 0.3926
- Transformer t1.0.1 (pre-fix; leaked)
  - Priority macro-F1 0.2203 | Accuracy 0.2355
  - Department macro-F1 1.0000 | Accuracy 1.0000 (invalid due to leakage)
- Transformer t1.0.2 (canary; leak guard on)
  - Priority macro-F1 0.0996 | Accuracy 0.2420
  - Department macro-F1 0.2506 | Accuracy 0.6021
- Transformer t1.0.3 (improved training; leak guard on)
  - Priority macro-F1 0.2348 | Accuracy 0.2462
  - Department macro-F1 0.2506 | Accuracy 0.6021
- Transformer t1.0.4 (in progress; new loss weights + [SEP] input)
  - Smoke 1-epoch check: Priority macro-F1 0.1392 | Department macro-F1 0.3047

Interpretation:
- After removing leakage, the transformer’s department head initially collapsed to the majority class (macro-F1 ~0.25, high accuracy). Equal loss weighting caused this weaker task to hurt the shared encoder. We’re reweighting losses and improving inputs; early smoke verifies the new pipeline.

## What we built (skills and decisions)
- Machine learning:
  - scikit-learn: TF-IDF + LogisticRegression baselines with tuning and weighted classes.
  - Transformers: PyTorch + Hugging Face (DistilBERT) dual-head classifier; label smoothing; class weights; validation-based selection.
- MLOps and software engineering:
  - FastAPI microservice with health checks and routing by environment flag.
  - Model registry (staging/production) and per-model manifests with artifact URIs and metrics.
  - Reproducibility: fixed train/val splits (seeded), versioned artifacts and metrics.
  - Anti-leakage: regex exclusions in both training and inference for parity.
  - Automation: PowerShell training pipelines with logs; comparison tooling.
  - Git hygiene: removed large binaries, .gitignore rules for weights/tokenizers.
  - Windows-friendly engineering: file:// path handling, tokenizer directory copying, quoting and process control in PowerShell.
- Testing and quality gates:
  - Unit tests for app and utilities (26 tests passing).
  - Planned CI gates to block leakage or metric regressions.

## Impact and learnings
- We detected and eliminated a subtle source of data leakage that would have overestimated model performance.
- The baseline sklearn model remains competitive on macro-F1; transformers require careful training and task weighting to surpass it.
- Clean separation between training-time behavior and serving through manifests/registry makes iteration safe and auditable.

## What’s next (roadmap)
- Finish t1.0.4 run (5 epochs, priority-focused loss) and re-compare to baseline.
- If needed, try a stronger encoder (roberta-base), slightly longer training, or early stopping on priority macro-F1.
- CI improvements: automatic comparisons and promotion gates; smoke-load staging manifest on PR.
- Optional: ONNX export for inference speed; calibration/thresholding for priority tiers.

## How to explore this repo (optional)
- Models and metrics:
  - `models/transformers/<version>/metrics.json` and `manifest.json`
  - `models/registry/staging.json` and `production.json`
- Compare models:
  - `scripts/compare_models.py` prints a metrics table across versions.
- Train transformers:
  - `scripts/run_t104.ps1` runs the tuned pipeline with logs in the model folder.

## Conclusion
This project showcases practical ML engineering: from classical baselines and leakage-proofing to transformer fine-tuning with a safe serving path. It balances research-style experimentation with production discipline (tests, manifests, registries, metrics), and it’s structured to keep improving while protecting quality.
