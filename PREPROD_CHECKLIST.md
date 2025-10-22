# Pre-Production Checklist (Portfolio)

Use this checklist to get the project to a clean, portfolio-ready pre-production state. Keep it light, reproducible, and standards-aligned.

## 1) Transformer Baselines (t1.0.1 • t1.0.2)
- [x] Train initial DistilBERT baseline (t1.0.1)
  - [x] 3 epochs, max_len=256 (baseline); observed suspicious department metrics → leakage
  - [x] Document anomaly and add leak guard (`--exclude-pattern "__type_[a-z0-9_]+"`)
- [x] Train fast canary with leak guard (t1.0.2)
  - [x] 1 epoch, max_len=128 for speed; artifacts present (`pytorch_model.bin`, `tokenizer/`, `label_mappings.json`, `metrics.json`)
  - [x] `models/transformers/t1.0.2/manifest.json` created (URIs, base_model, max_length, exclude_patterns)
  - [x] Compare metrics vs sklearn v1.0.10 using `scripts/compare_models.py`

## 2) Registry Pointers
- [x] Update `models/registry/staging.json` → `transformer:t1.0.2`
- [x] Keep `models/registry/production.json` → `sklearn:v1.0.10` (for now)
- [x] `/health/ready` passes (registry + manifest accessible)

## 3) Model Selection Flag
- [x] Add `SERVE_MODEL_TYPE` env (values: `sklearn`|`transformer`)
- [x] Route `/classify` to selected backend
- [x] Keep `/classify/transformer` for explicit canary testing

## 4) Tests (API + Contracts)
- [x] Unit tests for `/classify` (sklearn) and `/classify/transformer` (transformer)
  - [x] Happy path returns `TicketResponse`
  - [x] 400 on bad input, 503 when model not ready
- [x] Snapshot minimal JSON shape (keys + types) for responses (basic)

## 5) Manifest Validation
- [ ] Script validates `manifest.json` against `models/specs/model_manifest.schema.json`
- [x] Verifies referenced files exist locally (runtime loader used as a check)
- [ ] (Optional) Computes SHA256 and writes checksum into manifest

## 6) Publish/Promote Scripts (local-first)
- [ ] `scripts/publish_model.py` copies artifacts to versioned folder and writes manifest
- [ ] `scripts/promote_model.py` flips `models/registry/{staging|production}.json`
- [ ] Document usage in README

## 7) Docker (CPU-only)
- [ ] Build image and run API locally
- [ ] Set Hugging Face cache env for container
- [ ] Document run command and port mapping

## 8) CI Pipeline + Gates
- [ ] Lint (ruff/black), typecheck (mypy), tests (pytest)
- [ ] Gates check using `configs/gates.yaml` (accuracy, macro-F1, per-class recall)
- [ ] Fail CI on regressions

## 9) Docs Refresh
- [x] README quickstart (sklearn + transformer)
- [x] Training guide and registry overview
- [x] Endpoints table (`/classify`, `/classify/transformer`, health endpoints)
- [ ] Update `MODEL_CARD.md` with latest metrics
- [ ] Update `CHANGELOG.md`

## 10) Examples + Postman
- [ ] Python client snippet and curl examples
- [ ] 5–10 curated sample tickets
- [ ] Postman collection export

## 11) Optional: ONNX + Latency
- [ ] Export ONNX and add `onnxruntime` path
- [ ] Measure latency on sample inputs (CPU) and record in README

## 12) Security Hygiene
- [ ] Pin critical dependencies; run `pip-audit` and `bandit`
- [ ] `.env.example` is sane; no secrets committed

## 13) Observability
- [ ] Ensure logs include `request_id` and `model_id`
- [ ] Add simple drift checks (length distribution + class priors) and log warnings

## 14) Portfolio Polish
- [ ] Short demo GIF of classify flows
- [ ] Small architecture diagram (training ↔ registry ↔ API)
- [ ] Performance table (date, dataset, metrics for sklearn vs transformer)

---

### Results Snapshot (Validation)

| Model | Priority Macro-F1 | Priority Acc | Dept Macro-F1 | Dept Acc |
|-------|-------------------:|-------------:|--------------:|---------:|
| sklearn v1.0.10 | 0.2570 | 0.2574 | 0.3179 | 0.3926 |
| transformer t1.0.1 | 0.2203 | 0.2355 | 1.0000 | 1.0000 |
| transformer t1.0.2 | 0.0996 | 0.2420 | 0.2506 | 0.6021 |

How to re-run the comparison:
```powershell
$env:PYTHONPATH=(Get-Location)
.\.venv\Scripts\python.exe scripts\compare_models.py
```

Notes:
- t1.0.1 shows leakage (perfect department) and should not be promoted.
- t1.0.2 confirms leak guard; it’s a fast canary (1 epoch @ 128) for wiring validation, not a production candidate.

---

Acceptance bar (suggested):
- Priority: accuracy ≥ 0.50, macro-F1 ≥ 0.45
- Department: accuracy ≥ 0.65, macro-F1 ≥ 0.58
- No leakage warnings in training logs; readiness and liveness endpoints pass.

Notes:
- Keep large binaries out of git; commit manifests/metadata only.
- Default transformer canary via `SERVE_MODEL_ENV=staging`; promote when gates pass.
