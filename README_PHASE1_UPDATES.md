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

## Phase 3 Operations Enablement

This phase focused on establishing reproducible deployment, automated quality gates, and release management scaffolding.

### 1. Containerization
- Added multi-stage `Dockerfile` (builder + slim runtime) for smaller production images.
- Added `.dockerignore` to exclude development and cache artifacts.
- Non-root `appuser` runtime user for improved security posture.
- Rationale: Deterministic build artifacts, parity between local and CI builds, easier orchestration deployment.

### 2. Continuous Integration Workflow (`.github/workflows/ci.yml`)
- Triggers on pushes and PRs to `main` and `dev`.
- Steps: dependency install, ruff lint, mypy type check, pip-audit security scan, pytest execution, Docker image build.
- Tag-based publish job scaffolds pushing image to GHCR (`ghcr.io`).
- Rationale: Prevents regressions, enforces a baseline security and style bar before merging.

### 3. Developer Tooling
- `ruff` + `ruff-format` + `black` + `mypy` integrated via `.pre-commit-config.yaml`.
- `mypy.ini` with warnings tuned for progressive typing hardening.
- `ruff.toml` (import sorting + line length 120) for consistent style.
- Rationale: Reduces stylistic churn in PRs and surfaces structural issues early.

### 4. Release Management Assets
- `CHANGELOG.md` introduced with Unreleased section and initial tag baseline.
- `RELEASE_CHECKLIST.md` enumerates pre/post release tasks (tests, audit, metadata validation, tagging, smoke tests).
- Rationale: Formalizes release cadence and reduces human error in deployment steps.

### 5. Version Bump Automation
- Script `scripts/bump_version.py` to increment semantic version (major/minor/patch) in `app/config.py`.
- Rationale: Minimizes manual edits & keeps version change atomic for tagging + image build.

### 6. Artifact Versioning Alignment
- CI + version bump support align with existing `models/v<MODEL_VERSION>/` artifact layout.
- Rationale: Ensures traceable linkage between code version, Docker image tag, and model artifacts.

### 7. Security Baseline (Initial)
- Added pip-audit into CI (non-blocking for now) to surface vulnerable dependencies early.
- Request size limits + body validation already enforced at runtime.
- Rationale: Establish foundation before adding rate limiting & advanced threat mitigations in later phases.

### 8. Remaining Gaps Before Phase 4
- Evaluation pipeline enhancements (per-label F1 tracking persisted).
- Expanded OpenAPI docs + standardized error schema.
- Performance/load baseline scripts (Locust or k6) & initial benchmark record.
- Optional: Make security scans blocking after triage cycle.

## Updated File Inventory (Phase 3 Additions)
| File | Type | Purpose |
|------|------|---------|
| `Dockerfile` | New | Multi-stage container build definition |
| `.dockerignore` | New | Excludes non-essential files from image context |
| `.github/workflows/ci.yml` | New | CI automation pipeline |
| `.pre-commit-config.yaml` | New | Local lint/type hooks |
| `mypy.ini` | New | Type checking configuration |
| `ruff.toml` | New | Ruff lint/format configuration |
| `CHANGELOG.md` | New | Change history log |
| `RELEASE_CHECKLIST.md` | New | Release process guide |
| `scripts/bump_version.py` | New | Semantic version bump automation |

## Phase 3 Additions (Post Initial Documentation Update)

After initial Phase 3 drafting, the following items were subsequently completed:

1. Evaluation metrics summary fields added (macro/weighted F1 & accuracy) into `metrics.json` plus surfaced macro F1 in `model_metadata.json`.
2. Standardized error response model (`ErrorResponse`) and global exception handlers for HTTP & unhandled exceptions.
3. Basic IP-based in-memory rate limiting middleware (configurable via `RATE_LIMIT_REQUESTS` / `RATE_LIMIT_WINDOW_SEC`).
4. Performance baseline scaffolding: `load_test.md` and `locustfile.py` for initial latency & throughput benchmarks.
5. Security posture strengthened with non-root runtime user, request size limit, rate limiting, and CI `pip-audit` (advisory mode).

## Updated File Inventory (Additional Additions)
| File | Type | Purpose |
|------|------|---------|
| `app/models/schemas.py` | Modified | Added `ErrorResponse` model |
| `app/main.py` | Modified | Global exception handlers + rate limiting middleware |
| `train.py` | Modified | Metrics summary (macro/weighted F1, accuracy) persisted |
| `load_test.md` | New | Performance baseline documentation |
| `locustfile.py` | New | Load generation script for benchmarking |

## Ready for Phase 4
Foundation for advanced observability and metrics export is now in place. Phase 4 will focus on:
- Prometheus-compatible metrics endpoint (request counts, latency histograms, model inference timing).
- Enhanced structured error taxonomy (stable error codes & documentation).
- Optional distributed rate limiting strategy (back-end store) if scaling beyond single instance.
- Instrumentation of classifier inference path (timers + success/failure counters).

## Evaluation Tooling & Continuous Quality Gate (Added Post Phase 3)

Two scripts were introduced to enable fast human and automated feedback loops:

| Script | Purpose |
|--------|---------|
| `scripts/smoke_test.py` | Verifies liveness, readiness, version metadata, and a single classification (colored output) |
| `scripts/bulk_eval.py` | Runs labeled dataset to compute accuracy & macro-F1, produces optional Markdown report, supports failure thresholds |

These are now integrated into the CI `evaluation` job which:
1. Starts the API (`uvicorn app.main:app`).
2. Executes the smoke test (health + one prediction).
3. Runs bulk evaluation against `data/mock_eval.csv`.
4. Enforces accuracy thresholds (currently priority ≥ 0.80, department ≥ 0.90).
5. Uploads `reports/eval_report.md` as an artifact for inspection.

Planned enhancements:
- Add macro-F1 gating flags.
- Persist historical evaluation artifacts for trend analysis.
- Introduce JSONL raw predictions output for future drift detection.
