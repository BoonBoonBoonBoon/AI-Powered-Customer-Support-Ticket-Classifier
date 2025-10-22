# Project Progress and Key Decisions

Date: 2025-10-21
Branch: dev

## Overview
We stabilized a classical sklearn pipeline, ran focused A/B experiments, introduced governance (metrics gates, leakage checks), and then pivoted to a transformer path to pursue higher accuracy. We also added a simple model registry with manifests and wired the API to support transformer inference via a new endpoint.

## What changed and why

### 1) Classical model hardening (sklearn TF‑IDF + LogisticRegression)
- Added optional char n‑gram features and solver adjustments (saga when needed).
- Exposed per‑target hyperparameters (C, penalty, l1_ratio) and word n‑gram maxima.
- Fixed an evaluation bug by ensuring char features are stacked at eval time.
- Reason: extract marginal gains safely and ensure evaluation matches training features.

Results (high level):
- Best classical config (v1.0.10: priority char‑only) slightly improved priority macro‑F1 (~+0.0076) while keeping department stable.
- Other experiments (cost weights, lower C, tri‑gram + elasticnet, deduplication) had negligible or negative impact.

### 2) Data quality and leakage guardrails
- Ran deduplication via Jaccard shingles; impact was neutral to slightly negative, so kept as optional.
- Ran token mutual information checks on enrichment patterns; no leakage‑like features found.
- Reason: reduce risk of overfitting/leakage and improve data hygiene.

### 3) Governance: metrics, gates, and metadata
- Persisted metrics, calibration stats, and split indices; added recall thresholds and macro‑F1 gating.
- Introduced `configs/gates.yaml` to centralize CI thresholds.
- Reason: enforce minimum quality bars and make runs reproducible.

### 4) Model registry and manifests
- Added `models/registry/{production,staging}.json` pointers and per‑version `manifest.json` files with URIs.
- Added a JSON Schema (`models/specs/model_manifest.schema.json`) for validation.
- Reason: separate code from model artifacts, enable canarying and promotion via pointers.

### 5) Transformer pivot and baseline
- Added a DistilBERT dual‑head training script (`scripts/train_transformer.py`).
- Installed transformer stack in venv and trained a 1‑epoch smoke (`t1.0.0`).
- Wrote a transformer manifest and set `staging` to point to it; `production` remains sklearn.
- Reason: classical tweaks hit diminishing returns; transformers are the best ROI path toward our target accuracy.

### 6) API readiness and transformer inference
- Readiness endpoint now validates registry + manifest accessibility.
- Implemented transformer runtime (`app/models/inference_transformer.py`) and a lazy registry loader.
- Added new endpoint `/classify/transformer` that uses the model referenced by staging by default.
- Reason: allow canary testing of transformer without disrupting the main sklearn path.

## Current state
- Serving: sklearn remains the default for `/classify`; transformer available at `/classify/transformer`. You can route `/classify` to transformer via `SERVE_MODEL_TYPE=transformer`.
- Registry pointers:
  - production → sklearn v1.0.10
  - staging → transformer t1.0.2 (canary)
- Infra: manifests, schema, gates config in place; loader supports directory artifacts (tokenizer). Inference path can apply exclusion patterns for leakage parity.

### Transformer canary results (validation)

| Model | Priority Macro-F1 | Priority Acc | Dept Macro-F1 | Dept Acc | Notes |
|-------|-------------------:|-------------:|--------------:|---------:|-------|
| sklearn v1.0.10 | 0.2570 | 0.2574 | 0.3179 | 0.3926 | Best classical baseline (char-only priority) |
| transformer t1.0.1 | 0.2203 | 0.2355 | 1.0000 | 1.0000 | Invalid (department leakage detected) |
| transformer t1.0.2 | 0.0996 | 0.2420 | 0.2506 | 0.6021 | 1 epoch @ 128; leak guard applied; majority bias on dept |

Interpretation:
- t1.0.2 confirms the leakage fix (no more perfect department scores). As a fast, 1‑epoch canary it underperforms sklearn on priority and trades macro‑F1 for higher department accuracy (favoring the majority class). This is expected at low epochs and shorter max_len.
- Next we should train a stronger transformer baseline (3 epochs @ 256 tokens) and reassess.

## Next steps (short list)
1) Train a stronger transformer baseline (t1.0.3: 3 epochs, max_len 256) and update staging if it beats sklearn.
2) Auto-emit manifest from the trainer (persist `exclude_patterns`, base_model, metrics, URIs) to reduce manual steps.
3) Add a CI compare step using `scripts/compare_models.py` and enforce transformer gates (avoid regressions and leakage).
4) Optional: ONNX export for CPU latency, publish/promote scripts for cloud storage.

## Acceptance targets (suggested)
- Priority: accuracy ≥ 0.50, macro‑F1 ≥ 0.45
- Department: accuracy ≥ 0.65, macro‑F1 ≥ 0.58

## Notes
- Keep large binaries out of git; track only manifests/metadata.
- Default transformer canary via `SERVE_MODEL_ENV=staging`; promote when gates pass.
