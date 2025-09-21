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

## Phase 2 Modernization (Initial Changes)

The second phase begins introducing runtime and operational modernization without altering the core ML logic. The initial Phase 2 changes now included in this document are focused on application lifecycle management and version introspection.

### 1. Lifespan Refactor (`app/main.py`)
- Replaced deprecated `@app.on_event("startup")` usage with FastAPI's recommended `lifespan` async context manager.
- Ensures models and metadata are loaded exactly once at startup and provides a clean place for future teardown (e.g., DB pools, async clients).
- Rationale: Future‑proofing against deprecation, clearer initialization flow, easier extensibility.

### 2. Automatic Latest Model Discovery
- Added helper `_find_latest_version_dir` to scan `models/` for semantic version directories (`vX.Y.Z`) and select the highest version.
- Rationale: Allows deploying new model versions side‑by‑side; the API automatically serves the latest without code changes.

### 3. Centralized Model & Metadata Loading
- During lifespan startup: loads the serialized classifier and parses `model_metadata.json` (if present) from the latest version directory.
- Exposes loaded metadata for endpoints (e.g., `/version`).
- Handles absence of artifacts gracefully (API still starts, reports `model_loaded=false`).
- Rationale: Robust startup that degrades gracefully in fresh environments or during blue/green rollout sequencing.

### 4. Version Introspection Endpoint: `GET /version`
- Returns:
   - `api_version`: Mirrors `settings.MODEL_VERSION` (current deployment / contract version).
   - `model_loaded`: Boolean indicating successful artifact load.
   - `model_metadata`: Subset passthrough of training metadata (if available) enabling traceability.
- Rationale: Operational transparency (supports monitoring dashboards, release validation scripts, and automated smoke tests).

### 5. Root Endpoint Behavior
- Retains explicit CORS header injection used earlier to satisfy existing tests; slated for consolidation under a dedicated CORS middleware configuration in a later phase.

### 6. Testing Impact
- All existing tests (23) pass unchanged after the refactor, confirming no functional regressions to classification or schema behavior.

### 7. Additional Modernization Implemented (Post-Initial Phase 2)
- Structured JSON logging with request correlation IDs (middleware injected `X-Request-Id`).
- Request body size limiting middleware (64KB) to mitigate abuse and accidental large payloads.
- Liveness (`/health/live`), readiness (`/health/ready`), and backward-compatible legacy health (`/health`) endpoints.
- Endpoint tagging (`inference`, `health`) preparing for richer OpenAPI grouping.

### 8. Future Phase 2 / Phase 3 Targets (Still Pending)
- Centralized exception handling with standardized error envelope.
- Docker & multi-stage build for reproducible deployments.
- CI pipeline (lint, tests, coverage, artifact packaging, security scanning).
- Additional security hardening: dependency auditing, potential basic rate limiting, stricter field length constraints (beyond current body limit).
- Performance baseline scripts (Locust/k6) for latency regression tracking.

## Updated File Inventory (Phase 2 Additions)
| File | Change Type | Purpose |
|------|-------------|---------|
| `app/main.py` | Modified | Introduced lifespan context, automatic model version discovery, `/version` endpoint |

## Operational Notes
- The API `version` attribute now mirrors the configured `MODEL_VERSION`, aligning OpenAPI docs with deployed artifact expectations.
- The `/version` endpoint enables simple readiness verification beyond mere liveness (e.g., confirm model + metadata presence in orchestration probes or smoke tests).

## Next Recommended Steps
1. Introduce CI workflow (build, test, security scan, artifact publish).
2. Containerize (multi-stage Dockerfile + pinned runtime deps).
3. Add centralized exception handling + error response models.
4. Implement security pipeline steps (pip-audit, optional bandit) & rate limiting notes.
5. Establish performance baseline (load testing script & initial metrics).

---
This document will continue to evolve as subsequent hardening phases are completed.
