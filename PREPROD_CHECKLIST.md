# Pre-Production Checklist (Portfolio)

Use this checklist to get the project to a clean, portfolio-ready pre-production state. Keep it light, reproducible, and standards-aligned.

## 1) Transformer Baseline (t1.0.1)
- [ ] Train DistilBERT baseline (3–5 epochs, max_len=256)
  - [ ] Run: `scripts/train_transformer.py --data data/enriched_customer_tickets.csv --epochs 3 --batch-size 16 --max-len 256 --output-version t1.0.1`
  - [ ] Confirm artifacts in `models/transformers/t1.0.1/` (`pytorch_model.bin`, `tokenizer/`, `label_mappings.json`, `metrics.json`)
  - [ ] Write/update `models/transformers/t1.0.1/manifest.json` (URIs, base_model, max_length)
  - [ ] Compare metrics vs sklearn v1.0.10

## 2) Registry Pointers
- [ ] Update `models/registry/staging.json` → `transformer:t1.0.1`
- [ ] Keep `models/registry/production.json` → `sklearn:v1.0.10` (for now)
- [ ] `/health/ready` passes (registry + manifest accessible)

## 3) Model Selection Flag
- [ ] Add `SERVE_MODEL_TYPE` env (values: `sklearn`|`transformer`)
- [ ] Route `/classify` to selected backend
- [ ] Keep `/classify/transformer` for explicit canary testing

## 4) Tests (API + Contracts)
- [ ] Unit tests for `/classify` (sklearn) and `/classify/transformer` (transformer)
  - [ ] Happy path returns `TicketResponse`
  - [ ] 400 on bad input, 503 when model not ready
- [ ] Snapshot minimal JSON shape (keys + types) for responses

## 5) Manifest Validation
- [ ] Script validates `manifest.json` against `models/specs/model_manifest.schema.json`
- [ ] Verifies referenced files exist locally
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
- [ ] README quickstart (sklearn + transformer)
- [ ] Training guide and registry overview
- [ ] Endpoints table (`/classify`, `/classify/transformer`, health endpoints)
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

Acceptance bar (suggested):
- Priority: accuracy ≥ 0.50, macro-F1 ≥ 0.45
- Department: accuracy ≥ 0.65, macro-F1 ≥ 0.58
- No leakage warnings in training logs; readiness and liveness endpoints pass.

Notes:
- Keep large binaries out of git; commit manifests/metadata only.
- Default transformer canary via `SERVE_MODEL_ENV=staging`; promote when gates pass.
