# Phase 1 Hardening Updates

This document summarizes the initial production hardening (Phase 1) applied to the project and the rationale behind each change.

## Summary of Changes

1. Configuration System (`app/config.py`)
   - Introduced a `Settings` class (Pydantic `BaseSettings`) to centralize runtime configuration.
   - Added support for environment-driven overrides (`.env` file) and provided `.env.example`.
   - Rationale: Centralizes tunable parameters (model version, seed, validation split) for reproducibility and deploy-time flexibility.

2. Environment Example (`.env.example`)
   - Provides a template for configuring the application in different environments (local, staging, production).
   - Rationale: Reduces onboarding friction and encourages explicit configuration management.

3. Training Pipeline Enhancements (`train.py`)
   - Added deterministic seeding (`numpy` + `PYTHONHASHSEED`).
   - Added validation split (configurable) and optional evaluation toggle.
   - Generated `metrics.json` (classification reports + confusion matrices for priority & department).
   - Generated `model_metadata.json` (version, timestamp, data hash, distributions, sample counts, metrics reference).
   - Introduced versioned model directory structure: `models/v<MODEL_VERSION>/`.
   - Rationale: Enables traceability, reproducibility, and supports future model iteration without overwriting artifacts.

4. Data Integrity Hashing
   - SHA-256 hash of the training CSV included in metadata.
   - Rationale: Detects silent data drift or mismatches between reported and actual training data.

5. Artifact Separation
   - Metrics and metadata stored alongside serialized models inside the versioned directory.
   - Rationale: Keeps a cohesive, portable artifact bundle for deployment or auditing.

## Files Added / Modified

| File | Type | Purpose |
|------|------|---------|
| `app/config.py` | New | Central configuration via Pydantic settings |
| `.env.example` | New | Example environment configuration |
| `train.py` | Modified | Added eval, metrics, metadata, seeding, versioned output |
| `README_PHASE1_UPDATES.md` | New | Documentation of Phase 1 changes |

## Resulting Artifacts (After Training)
```
models/
  v1.0.0/
    priority_model.pkl
    department_model.pkl
    metrics.json          # (if ENABLE_EVAL=true)
    model_metadata.json
```

## Next Recommended Steps (Phase 2+)
- Lifespan refactor for FastAPI startup (replace deprecated `@app.on_event`).
- Structured logging & request ID middleware.
- Pre-commit + linting + mypy integration.
- Dockerfile & CI pipeline for automated builds and test gating.
- `/version` endpoint surfacing model + API + metrics summary.

## Repro Instructions
1. Copy `.env.example` to `.env` and adjust if needed.
2. Run training:
   ```bash
   python train.py --data data/sample_tickets.csv --models-dir models
   ```
3. Inspect artifacts:
   ```bash
   tree models/v1.0.0
   ```
4. Examine `model_metadata.json` & `metrics.json` for provenance and performance.

## Rationale Recap
These changes focus on *traceability*, *reproducibility*, and *controlled iteration*. By locking in versioned artifacts with explicit metadata and metrics, future model improvements can be benchmarked objectively and rolled back safely.

---
Prepared as part of Phase 1 hardening for V1 readiness.
