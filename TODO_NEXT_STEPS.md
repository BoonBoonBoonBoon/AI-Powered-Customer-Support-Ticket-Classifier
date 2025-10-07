# Project TODO / Next Steps Roadmap

This file captures the prioritized improvement backlog for the Ticket Classifier. Tasks are grouped by urgency and domain. Treat this as a living document—update it as versions advance.

Legend:  Priority: (🔥 High, 🚀 Medium, 🧪 Low)  Effort: (S / M / L)

---
## Tier 0 – Immediate (Stabilize & Understand)
1. Error Analysis (🔥, S)
   - Run `scripts/error_analysis.py` on validation + holdout for v1.0.3.
   - Output: confusion matrices, per-class PRF1, top high-confidence misclassifications.
   - Artifact: `reports/error_analysis_v1.0.3.{json,md}`.
2. Leakage Guard (🔥, S)
   - Add warning in `train.py` if any target macro F1 == 1.0 with support > 30.
   - Unit test with synthetic perfectly separable dataset.
3. Baseline Snapshot (🚀, S)
   - Commit `reports/baseline_model_v1.0.3.md` summarizing metrics & distributions.

## Tier 1 – Model Performance
4. Priority Feature Engineering (🔥, M)
   - Add escalation keyword lexicon tokens; textual ratio features (caps %, digit count, punctuation density).
   - Goal: +0.02 absolute macro F1 on priority (validation).
5. Department Noise Audit (🚀, S)
   - Evaluate impact of removing/keeping `__product_` tokens; prune if not helping.
6. Algorithm Comparison (🔥, M)
   - Grid: LogisticRegression (C values), Linear SVM (calibrated), Multinomial NB.
   - Script outputs comparison table; select per-target best.
7. Probability Calibration (🚀, M)
   - Use `CalibratedClassifierCV` for both targets; store Brier score & reliability plot.

## Tier 2 – Data Quality & Enrichment
8. Hard Negative Mining (🚀, M)
   - Collect false positives/negatives ≥ confidence 0.60; store in `data/hard_negatives.csv`.
9. De-duplication Pass (🚀, S)
   - MinHash / Jaccard to remove near duplicates; log % removed.
10. Interaction Tokens (🧪, M)
   - Add composite tokens (e.g., `__csat_low__+urgent`). Include only if macro F1 improves ≥ 0.01.

## Tier 3 – Evaluation & Monitoring
11. Macro-F1 CI Gating (🚀, S)
   - Fail pipeline if priority macro F1 < 0.22 or department < 0.30 (tunable).
12. Drift Metadata Stub (🧪, S)
   - Add average description length + vector norm to `model_metadata.json`.
13. Confidence Histogram (🧪, S)
   - Persist bin counts; inform triage threshold (e.g., <0.55 human review).

## Tier 4 – Infrastructure & Reliability
14. Prometheus Metrics Endpoint (🚀, M)
   - `/metrics`: request count, latency histogram, model inference duration.
15. Orchestrated Make/Task Pipeline (🧪, S)
   - `make train-eval VERSION=...` runs train → error analysis → baseline report.

## Tier 5 – Codebase & Architecture
16. Per-Target Config Refactor (🚀, M)
   - Refactor `TicketClassifier` to modularize target pipelines.
17. Model Registry Abstraction (🧪, M)
   - Symlink or JSON pointer to `latest` vs `canary` model.
18. Stricter Typing (🧪, S)
   - Enable stricter mypy config; eliminate residual `Any`.

## Tier 6 – Security & Supply Chain
19. Bandit SAST Stage (🚀, S)
   - Add to CI; fail on HIGH severity.
20. SBOM Generation (🧪, M)
   - CycloneDX or Syft; attach artifact to build.

## Tier 7 – Active Learning (Later Wave)
21. Feedback Endpoint (🧪, M)
   - `/feedback` captures corrected labels + model_version.
22. Low-Confidence Queue (🧪, M)
   - Persist < threshold predictions for labeling pool.

## Tier 8 – R&D / Stretch
23. Transformer Prototype (🧪, L)
   - DistilBERT/MiniLM dual-head fine-tune; compare macro F1 & latency.
24. Ensemble Voting (🧪, M)
   - Blend logistic + calibrated SVM; evaluate uplift.
25. Cost-Sensitive Strategy (🧪, M)
   - Weight misclassification of Urgent more; calibrate recall.

## Documentation
26. MODEL_CARD.md (🚀, S)
   - Intended use, limitations, metrics, ethical notes.
27. Operational Playbook (🧪, S)
   - Roll, rollback, evaluate, promote.
28. Data Lineage Section (🚀, S)
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
_Last updated: <INSERT DATE WHEN EDITING NEXT>_
