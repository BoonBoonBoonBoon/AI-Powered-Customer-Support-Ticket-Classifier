# Model Card: Customer Support Ticket Classifier (v1.0.5)

## Overview
Dual-task text classifier predicting ticket Priority (Urgent, High, Medium, Low) and Department (Billing, Sales, Tech Support). Uses TF‑IDF + Logistic Regression with per-target hyperparameters and optional probability calibration.

## Intended Use
- Automate initial triage and routing of incoming support tickets.
- Provide confidence scores to drive human review thresholds.
- Supply operational metrics (macro F1, Brier score, reliability bins) for governance.

Not for: final SLA decisions without human oversight; abusive content moderation; legal or compliance classification.

## Version & Artifacts
- Version: v1.0.5
- Artifacts directory: `models/v1.0.5/`
- Files: `priority_model.joblib`, `department_model.joblib`, vectorizers, encoders, `metrics.json`, `holdout_metrics.json`, `calibration_metrics.json`, `model_metadata.json`.

## Data
Source: Enriched internal support ticket dataset (8,469 rows) with injected structured tokens (product, channel, ticket type, CSAT). Department leakage mitigated by excluding `__type_*` tokens for department model.

Class distributions:
- Priority: ~balanced (Low 24%, High 25%, Medium 26%, Urgent 25%).
- Department: Billing 60%, Tech Support 21%, Sales 19% (moderate imbalance).

## Training Configuration
| Aspect | Value |
|--------|-------|
| Vectorizer | TF-IDF (1–2 grams, min_df=2) |
| Priority Model | LogisticRegression (C=5.0, class_weight=balanced) |
| Department Model | LogisticRegression (C=2.0, class_weight=balanced) |
| Calibration | Sigmoid (Platt) via CalibratedClassifierCV(cv=3) |
| Length Tokens | Enabled (`--augment-length`) |
| Priority Extra Tokens | Disabled (no uplift) |
| Exclusions (Department) | `__type_[a-z0-9_]+` |
| Split | Train/Val/Holdout with persisted indices (0.15 holdout, 0.2 val of remainder) |
| Seed | 42 |

## Evaluation Metrics (Validation)
| Target | Macro F1 | Accuracy | Brier | Notes |
|--------|----------|----------|-------|-------|
| Priority | 0.1585 | 0.2535 | 0.7497 | Collapsed to majority-like Medium; calibration reveals low confidence concentration |
| Department | 0.2545 | 0.6174 | 0.5463 | Model predicts only Billing; strong imbalance indicates regression |

Holdout results (from `holdout_metrics.json`): Priority macro F1 0.1557 (acc 0.2472), Department macro F1 0.2481 (acc 0.5929).

WARNING: v1.0.5 configuration (C tuning + calibration) produced a severe collapse (no predictions for minority classes). Recommend rollback to v1.0.4 configuration or re-tuning with stronger regularization / class weighting strategy before promotion.

## Calibration & Reliability
`calibration_metrics.json` includes:
- Multiclass Brier score per target.
- Reliability bins (count, avg_conf, accuracy) for max class probability deciles.
Usage: align human review threshold where accuracy >= desired precision corridor; monitor drift if avg_conf - accuracy gap widens.

## Ethical & Operational Considerations
| Risk | Mitigation |
|------|-----------|
| Misprioritizing urgent incidents | Confidence threshold + manual review for low / borderline scores |
| Hidden leakage via new enrichment tokens | Regex exclusion list & periodic noise audit (`department_noise_audit.py`) |
| Data drift degrading performance | Plan: periodic re-run of error analysis + drift metadata logging |
| Over-reliance on low confidence predictions | Provide confidence + encourage threshold gating (<0.55 review) |

## Limitations
- Pure bag-of-words features; limited semantic understanding.
- Calibration only sigmoid; isotonic not yet evaluated.
- No cost-sensitive weighting beyond class_weight=balanced.
- Engineered priority tokens currently disabled—future refinement possible.

## Future Improvements
- Record Brier & reliability trends over time and expose via metrics endpoint.
- Add isotonic calibration comparison.
- Cost-sensitive loss / class-specific recall targets (especially Urgent).
- Transformer prototype for semantic lift (see roadmap Tier 8).

## Reproducibility
- Deterministic split indices persisted (`train_indices.txt`, `val_indices.txt`).
- Data file hash stored in metadata.
- Full training flags captured in `model_metadata.json` & `classifier_config.json`.

## Change Log Summary
- v1.0.3: Leakage fix for department tokens.
- v1.0.4: Priority engineered features (no gain).
- v1.0.5: Per-target C, probability calibration, calibration metrics, macro-F1 CI gate.

## Contact / Ownership
Maintainer: ML Platform / Triage Automation Team
Escalation Path: Create issue with label `model-regression` including offending samples & version.

---
This model card should be updated whenever a new version is trained or a material behavior change is introduced.
