# Project TODO / Next Steps Roadmap

This file captures the prioritized improvement backlog for the Ticket Classifier. Tasks are grouped by urgency and domain. Treat this as a living document—update it as versions advance.

Legend:  Priority: (🔥 High, 🚀 Medium, 🧪 Low)  Effort: (S / M / L)

---
## Tier 0 – Immediate (Stabilize & Understand)
- [x] Error Analysis (🔥, S)
   - Completed for v1.0.3. Artifacts: `reports/error_analysis_v1.0.3.{json,md}`.
   - NOTE: Initial run used regenerated split (indices not yet persisted); subsequent enhancement adds index persistence for future versions.
- [x] Leakage Guard (🔥, S)
   - Added macro F1 perfection warning in `train.py` (prints WARNING if macro F1 == 1.0 with support ≥ 50). Commit: d9557a3.
- [x] Baseline Snapshot (🚀, S)
   - Added `reports/baseline_model_v1.0.3.md` summarizing metrics, discrepancies, and recommendations. Commit: abbc5b1.

## Tier 1 – Model Performance
- [x] Priority Feature Engineering (🔥, M)
   - Added engineered keyword + structural tokens behind `--priority-extra` (v1.0.4). Artifacts: `models/v1.0.4/*`.
   - Result: Priority macro F1 (val) 0.2436 -> 0.2402 (Δ -0.0034); holdout 0.2662 -> 0.2648 (Δ -0.0014). No uplift; retain for now pending interaction pruning/algo search.
   - Next: evaluate during Algorithm Comparison; consider removing low-signal structural tokens if no gain.
- [x] Department Noise Audit (🚀, S)
   - Added `scripts/department_noise_audit.py` to compare exclusion regex sets; artifacts to be generated on demand (`department_noise_audit.{json,md}`).
   - Pending execution with real exclusion patterns (baseline implemented).
- [x] Algorithm Comparison (🔥, M)
   - Added `scripts/algorithm_comparison.py`; artifacts: `reports/algo_comparison_v1.0.4.{json,md}`.
   - Explored: LogReg C∈{0.5,1,2,5}, LinearSVC C∈{0.5,1,2}, MultinomialNB α∈{0.5,1,2}; with/without engineered tokens.
   - Best priority macro F1: 0.2654 (LogReg C=5.0 balanced, no priority-extra) vs previous 0.2436 (+0.0218 abs; meets +0.02 target).
   - Best department macro F1: 0.3335 (LogReg C=2.0 balanced) vs 0.3267 (+0.0068).
   - Engineered priority tokens underperformed; candidate for pruning/refinement after calibration.
- [x] Probability Calibration (🚀, M)
   - Added `--calibrate` flag; wraps LogisticRegression in `CalibratedClassifierCV (sigmoid, cv=3)`.
   - Config persisted in `classifier_config.json` (`calibrate_probabilities=true/false`). Brier score storage still TODO.

## Tier 2 – Data Quality & Enrichment
- [ ] Hard Negative Mining (🚀, M)
   - Collect false positives/negatives ≥ confidence 0.60; store in `data/hard_negatives.csv`.
- [ ] De-duplication Pass (🚀, S)
   - MinHash / Jaccard to remove near duplicates; log % removed.
- [ ] Interaction Tokens (🧪, M)
   - Add composite tokens (e.g., `__csat_low__+urgent`). Include only if macro F1 improves ≥ 0.01.

## Tier 3 – Evaluation & Monitoring
- [x] Macro-F1 CI Gating (🚀, S)
   - Added `scripts/macro_f1_gate.py` + CI step enforcing thresholds (priority ≥0.22, department ≥0.30) on synthetic quick-train.
   - Future enhancement: run gate against real persisted validation metrics rather than synthetic retrain.
- [ ] Drift Metadata Stub (🧪, S)
   - Add average description length + vector norm to `model_metadata.json`.
- [ ] Confidence Histogram (🧪, S)
   - Persist bin counts; inform triage threshold (e.g., <0.55 human review).

## Tier 4 – Infrastructure & Reliability
- [ ] Prometheus Metrics Endpoint (🚀, M)
   - `/metrics`: request count, latency histogram, model inference duration.
- [ ] Orchestrated Make/Task Pipeline (🧪, S)
   - `make train-eval VERSION=...` runs train → error analysis → baseline report.

## Tier 5 – Codebase & Architecture
- [ ] Per-Target Config Refactor (🚀, M)
   - Partial: per-target `C` values (`--priority-C`, `--department-C`) integrated; full modular pipeline abstraction still pending.
- [ ] Model Registry Abstraction (🧪, M)
   - Symlink or JSON pointer to `latest` vs `canary` model.
- [ ] Stricter Typing (🧪, S)
   - Enable stricter mypy config; eliminate residual `Any`.

## Tier 6 – Security & Supply Chain
- [ ] Bandit SAST Stage (🚀, S)
   - Add to CI; fail on HIGH severity.
- [ ] SBOM Generation (🧪, M)
   - CycloneDX or Syft; attach artifact to build.

## Tier 7 – Active Learning (Later Wave)
- [ ] Feedback Endpoint (🧪, M)
   - `/feedback` captures corrected labels + model_version.
- [ ] Low-Confidence Queue (🧪, M)
   - Persist < threshold predictions for labeling pool.

## Tier 8 – R&D / Stretch
- [ ] Transformer Prototype (🧪, L)
   - DistilBERT/MiniLM dual-head fine-tune; compare macro F1 & latency.
- [ ] Ensemble Voting (🧪, M)
   - Blend logistic + calibrated SVM; evaluate uplift.
- [ ] Cost-Sensitive Strategy (🧪, M)
   - Weight misclassification of Urgent more; calibrate recall.

## Documentation
- [ ] MODEL_CARD.md (🚀, S)
   - Intended use, limitations, metrics, ethical notes.
- [ ] Operational Playbook (🧪, S)
   - Roll, rollback, evaluate, promote.
- [ ] Data Lineage Section (🚀, S)
   - Document enrichment token origins & transformations.

---
## Suggested Next Sprint Bundle
1. (1) Error Analysis
2. (2) Leakage Guard
3. (4) Priority Feature Engineering
4. (6) Algorithm Comparison
5. (11) Macro-F1 CI Gating
6. (26) MODEL_CARD.md

Success Criteria: measurable macro F1 improvement (priority +0.02, department stable/improved), automated guardrails to prevent silent regressions, added transparency artifacts.

---
## Tracking Table Snapshot
| ID | Task | Priority | Effort | Impact | Depends |
|----|------|----------|--------|--------|---------|
| 1 | Error analysis | 🔥 | S | High | - |
| 2 | Leakage guard | 🔥 | S | High | 1 (optional) |
| 4 | Priority features | 🔥 | M | High | 1 |
| 6 | Algo comparison | 🔥 | M | Medium | 1 |
| 11 | Macro-F1 gating | 🚀 | S | Medium | 1 |
| 26 | Model card | 🚀 | S | Medium | 1 |
| 14 | Metrics endpoint | 🚀 | M | Medium | - |
| 23 | Transformer prototype | 🧪 | L | High (future) | 4,6 |

---
## Update Process
- When completing a task: mark it done with a checkbox (e.g., `- [x]`), add result summary (metric delta, artifact path).
- Re-evaluate priorities whenever new data or performance changes.

---
_Last updated: 2025-10-07_
