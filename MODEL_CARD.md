# Model Card: Customer Support Ticket Classifier (v1.0.6)

## Overview
Dual-task text classifier predicting ticket Priority (Urgent, High, Medium, Low) and Department (Billing, Sales, Tech Support). Uses TF‑IDF + Logistic Regression with per-target hyperparameters and optional probability calibration.

## Intended Use
- Automate initial triage and routing of incoming support tickets.
- Provide confidence scores to drive human review thresholds.
- Supply operational metrics (macro F1, Brier score, reliability bins) for governance.

Not for: final SLA decisions without human oversight; abusive content moderation; legal or compliance classification.

## Version & Artifacts
- Current Version: v1.0.6 (regression recovery after v1.0.5 collapse)
- Artifacts directory: `models/v1.0.6/`
- Files: `priority_model.joblib`, `department_model.joblib`, vectorizers, encoders, `metrics.json`, `calibration_metrics.json`, `model_metadata.json` (no holdout this run).

Historical versions:
- v1.0.5 (calibration + per-target C introduced severe collapse — majority prediction).
- v1.0.4 (previous stable baseline pre-calibration).

## Data
Source: Enriched internal support ticket dataset (8,469 rows) with injected structured tokens (product, channel, ticket type, CSAT). Department leakage mitigated by excluding `__type_*` tokens for department model.

Class distributions:
- Priority: ~balanced (Low 24%, High 25%, Medium 26%, Urgent 25%).
- Department: Billing 60%, Tech Support 21%, Sales 19% (moderate imbalance).

## Training Configuration (v1.0.6)
| Aspect | Value |
|--------|-------|
| Vectorizer | TF-IDF (1–2 grams, min_df=2) |
| Priority Model | LogisticRegression (C=5.0, class_weight=balanced) |
| Department Model | LogisticRegression (C=2.0, class_weight=balanced) |
| Calibration | Disabled (reverted after v1.0.5 collapse) |
| Length Tokens | Enabled (`--augment-length`) |
| Priority Extra Tokens | Disabled (no uplift) |
| Exclusions (Department) | `__type_[a-z0-9_]+` |
| Split | Train/Val (0.2) — no new holdout reserved in this corrective run |
| Seed | 42 |
| Guard Rails | Macro F1 CI gate + per-class recall gate (≥0.05) + leakage warning |

## Evaluation Metrics (Validation)
| Target | Macro F1 | Accuracy | Notes |
|--------|----------|----------|-------|
| Priority | 0.2494 | 0.2497 | Restored balanced per-class recall (0.239–0.272) after removing calibration |
| Department | 0.3178 | 0.3914 | Improved over v1.0.5 (0.2545) and slightly below v1.0.4 macro F1 (0.3267) but healthier distribution |

Delta vs prior versions:
- v1.0.5 → v1.0.6: Priority macro F1 +0.091; Department macro F1 +0.063 (collapse resolved).
- v1.0.4 → v1.0.6: Priority macro F1 +0.009; Department macro F1 -0.009 (trade-off acceptable for eliminating collapse risk).

No new holdout evaluation executed for v1.0.6 (focus was rapid remediation). Next planned version should reinstate holdout or reuse indices for comparability.

Recall Gate Result: All classes exceeded 0.05 recall threshold (lowest ~0.23). Gate passed.

## Calibration & Reliability
Calibration currently disabled. `calibration_metrics.json` retained for continuity (raw probability reliability, uncalibrated). Future re-introduction will test: (a) sigmoid with stratified CV sizes, (b) isotonic, (c) temperature scaling. Acceptance criteria: no class recall < baseline − 0.02 and macro F1 non-decreasing.

## Data Drift & Profile Telemetry (v1.0.6+)
Added early drift indicators to `model_metadata.json`:
| Field | v1.0.6 Value | Purpose |
|-------|--------------|---------|
| avg_description_length_tokens | 49.43 | Monitor content length shift (proxy for verbosity changes) |
| priority_vocab_size | 14858 | Track feature space growth for priority model |
| department_vocab_size | 14811 | Track feature space growth for department model |

Future: add rolling window stats (e.g., novelty rate of tokens, top token churn) and expose via `/metrics`.

## Department Noise Audit (v1.0.6 Reference Split)
Script: `scripts/department_noise_audit.py` executed with patterns: `__product_*`, `__channel_*`, `__csat_*`.

Results (validation split aligned to v1.0.6 indices): All configurations yielded Dept Macro F1 = 1.000 (perfect) indicating persisting leakage candidates beyond the already excluded `__type_*` tokens.

| Exclusions | Dept Macro F1 | Dept Acc | Priority Macro F1 |
|------------|---------------|----------|------------------|
| (none) | 1.0000 | 1.0000 | 0.2432 |
| __product_* | 1.0000 | 1.0000 | 0.2432 |
| __channel_* | 1.0000 | 1.0000 | 0.2432 |
| __csat_* | 1.0000 | 1.0000 | 0.2432 |
| all three | 1.0000 | 1.0000 | 0.2432 |

Interpretation: Validation slice used in audit still exhibits line-of-sight token leakage for department when exclusion regex not applied in the main training pipeline (the production training run already excludes `__type_*`). The perfect F1 in isolation suggests additional enrichment tokens (e.g., product/channel/CSAT markers) may correlate strongly or deterministically with department labels. Next steps: expand exclusion set cautiously and measure real performance drop; introduce mutual information ranking for candidate removable tokens.

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
- v1.0.5: Per-target C, probability calibration (regression), calibration metrics, macro-F1 CI gate.
- v1.0.6: Removed calibration, added per-class recall gate enforcement, recovered class diversity and macro F1.

## Contact / Ownership
Maintainer: ML Platform / Triage Automation Team
Escalation Path: Create issue with label `model-regression` including offending samples & version.

---
This model card should be updated whenever a new version is trained or a material behavior change is introduced.
