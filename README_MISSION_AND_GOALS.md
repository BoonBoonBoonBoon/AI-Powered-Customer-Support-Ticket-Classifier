# Project Mission & Strategic Goals

## Mission Statement
Deliver a transparent, reliable, and production-ready machine learning service that automatically classifies customer support tickets by priority and department, reducing triage time, improving response quality, and enabling measurable operational insights.

The core principle: **Operational trust first, model sophistication second.** Every enhancement should strengthen correctness, observability, safety, or maintainability while laying the groundwork for future intelligent assistance.

## Problem Context
Modern support teams are overwhelmed by inbound volume. Manual triage introduces:
- Delays in routing urgent incidents
- Inconsistent prioritization across agents
- Fragmented visibility into workload distribution
- Limited feedback loops for quality improvement

This project addresses these pain points by combining deterministic classical ML, structured operational practices, and extensible evaluation tooling.

## Core Value Propositions
1. **Speed to Triage:** Immediate, automated routing & urgency scoring.
2. **Predictable Behavior:** Deterministic, versioned models and reproducible training pipeline.
3. **Operational Transparency:** Health endpoints, structured logs, metrics (planned), and evaluation gates.
4. **Safety & Governance:** Standardized error schemas, size limits, rate limiting, dependency auditing.
5. **Continuous Confidence:** Built-in smoke + bulk evaluation with CI gating to prevent silent regressions.
6. **Extensibility:** Clean separation of concerns—config, model, API, evaluation—supports future feature layers (LLM assist, active learning, drift detection).

## Strategic Pillars
| Pillar | Description | Success Signals |
|--------|-------------|-----------------|
| Reliability | Service is stable & predictable under normal & degraded conditions | 100% green smoke tests, <1% failed classify ops after deployment |
| Observability | Fast diagnosis of model or system issues | Metrics + logs used to isolate issue in <10 min |
| Quality | Classification accuracy and consistency improve over time | Priority macro-F1 ↑ release-over-release |
| Security & Safety | Baseline risk mitigations in place early | CI security scan + body/rate guards enforced |
| Reproducibility | Anyone can retrain & reproduce metrics | Hash + version metadata match deployed state |
| Evolvability | Easy to layer new features without rewrites | Phase 4+ additions require minimal refactors |

## Goal Hierarchy
### Tier 1 (Foundational: COMPLETE / MAINTAIN)
- Deterministic training with versioned artifacts (`models/vX.Y.Z/`).
- Structured JSON logging + request correlation.
- Health & readiness endpoints (K8s friendly).
- Basic rate limiting + request size enforcement.
- Evaluation scripts: `smoke_test.py`, `bulk_eval.py` with accuracy gating in CI.
- Multi-stage Docker + reproducible dependency locking.

### Tier 2 (Near-Term Execution: IN PROGRESS)
- Holdout split + enhanced metrics (priority macro-F1 gating).
- Dataset profiling & normalization (`profile_dataset.py`).
- Class weighting & simple feature augmentation for priority.
- Mission & strategy documentation (this file).

### Tier 3 (Planned Next Phase)
- Prometheus `/metrics` endpoint: request count, latency buckets, inference timing.
- Enhanced error taxonomy: stable codes + documentation table.
- SBOM + SAST (Bandit) integrated into CI with severity thresholds.
- Macro-F1 + calibration curves persisted per model release.
- Configurable model selection (serve N-1 for rollback safety).

### Tier 4 (Expansion / Intelligent Assistance)
- Active learning loop (uncertainty sampling → labeling → retrain automation).
- Drift monitoring (input distribution + confidence shift alerts).
- LLM-assisted summarization or reply suggestion pipeline.
- Multi-lingual classification support.
- Pluggable vector-based semantic routing (hybrid retrieval + classifier).

### Tier 5 (Operational Excellence)
- Canary deploy strategy (diff in evaluation metrics gating production promotion).
- Real-time feature store integration (customer tenure, SLA tier).
- SLA breach early-warning classifier (predict resolution risk).
- Automated rollback trigger on sustained regression.

## Decision Principles
When making changes:
1. **Traceability over novelty** – Provenance (what model, what data) must be recoverable.
2. **Simplicity first** – Prefer classical ML until complexity is justified by measurable lift.
3. **Metric-driven evolution** – Introduce complexity only with pre-defined success criteria.
4. **Failure containment** – Guardrails (rate limit, body size, validation) never optional.
5. **Incremental rollouts** – Small, reversible steps > large, risky leaps.
6. **Human feedback loops** – Evaluation artifacts, misclassification tables, and reports are core deliverables.

## Success Metrics (Initial Set)
| Category | Metric | Target / Rationale |
|----------|--------|--------------------|
| Quality | Priority macro-F1 | ≥ 0.75 after enrichment (baseline improvement) |
| Quality | Department accuracy | ≥ 0.90 maintained |
| Reliability | 100% green smoke tests per deploy | No degraded endpoint states |
| Observability | Time-to-diagnose (log + metrics) | < 10 min for common failures |
| Governance | Reproducible training hash match | 100% of releases |
| Security | Dependency scan critical vulns | 0 unaddressed |

## Evaluation Lifecycle
1. Train → produce `metrics.json` + `model_metadata.json`.
2. Run bulk eval on curated & holdout CSVs → capture misclassification heatmap (future).
3. CI evaluation job enforces accuracy thresholds.
4. Manual review of misclassifications informs next data enrichment batch.
5. Version bump only with demonstrable metric lift or operational improvement.

## Data Enrichment Strategy (Snapshot)
- Profile data on arrival (distribution + duplicates + missing).
- Normalize labels & map noisy variants early.
- Add minority class samples intentionally before tuning models.
- Maintain a static challenge set (immutable) for longitudinal tracking.
- Introduce uncertainty sampling after baseline stabilization.

## Risk Register (Condensed)
| Risk | Impact | Mitigation |
|------|--------|------------|
| Label Imbalance (priority) | Skewed predictions | Class weighting + targeted sampling |
| Silent Regression | Quality drop post-release | CI gating + holdout metrics |
| Data Drift | Gradual performance decay | Planned drift monitors (Phase 4) |
| Over-fitting Synthetic Data | Inflated metrics | Limit synthetic proportion <25% |
| Operational Blind Spots | Slow incident response | Metrics endpoint + structured logs |
| Supply Chain Vulnerability | Exploit risk | pip-audit + SBOM + patch cadence |

## Release Criteria (Definition of Ready for Promotion)
- All Tier 1 features green.
- No critical/high unresolved vulnerabilities.
- Holdout priority macro-F1 meets or exceeds previous release.
- Evaluation artifact attached to PR (report + metrics diff).
- CHANGELOG updated, `MODEL_VERSION` bumped, artifacts committed.

## Near-Term Action Plan (Next 5 Steps)
1. Add class_weight & feature augmentation to priority model training.
2. Implement holdout metrics file + macro-F1 gating logic in CI.
3. Expose preliminary `/metrics` endpoint stub (shape only).
4. Add confusion matrix visualization (script/notebook export).
5. Draft enhanced error code catalog and integrate into docs.

## How to Contribute Toward the Mission
When opening a PR, include:
- Objective (what pillar it advances)
- Affected metrics & expected movement
- Rollback plan if regression detected
- Screenshot or artifact (if evaluation-related)

## Long-Term Vision
A measurable, auditable ML support assistant platform: from deterministic classification → active learning loop → intelligent assist (summaries, next-action suggestions) → adaptive routing informed by real-time context and performance telemetry.

---
This document is a living artifact. Propose edits in PRs titled `docs(mission): ...` to evolve strategy with evidence.
