# Baseline Snapshot – Model v1.0.3

Date: 2025-10-07

## Overview
Model version v1.0.3 trained on enriched real dataset (`data/enriched_customer_tickets.csv`) with department leakage mitigated using exclusion regex: `__type_[a-z0-9_]+`.

Training flags:
- holdout_frac: 0.15
- class_weight: balanced
- augment_length: True
- department_exclude_regex: ['__type_[a-z0-9_]+']

## Dataset Summary
- Total samples: 8,469
- Holdout samples: 1,270
- Validation samples: 1,440 (reconstructed in error analysis)
- Priority distribution: Medium 2192 | Urgent 2129 | High 2085 | Low 2063 (balanced)
- Department distribution: Billing 5081 | Tech Support 1747 | Sales 1641 (imbalanced)

## Validation Metrics (metrics.json)
### Priority
- Accuracy: 0.244
- Macro F1: 0.244
- Weighted F1: 0.244

| Class | Precision | Recall | F1 | Support |
|-------|-----------|--------|----|---------|
| High | 0.249 | 0.271 | 0.260 | 354 |
| Low | 0.237 | 0.248 | 0.242 | 351 |
| Medium | 0.266 | 0.241 | 0.253 | 373 |
| Urgent | 0.223 | 0.215 | 0.219 | 362 |

Confusion Matrix (rows=true, cols=pred):
```
[[96, 91, 87, 80],
 [109, 87, 75, 80],
 [87, 84, 90, 112],
 [93, 105, 86, 78]]
```

### Department
- Accuracy: 0.384
- Macro F1: 0.327
- Weighted F1: 0.407

| Class | Precision | Recall | F1 | Support |
|-------|-----------|--------|----|---------|
| Billing | 0.609 | 0.448 | 0.516 | 889 |
| Sales | 0.212 | 0.310 | 0.252 | 268 |
| Tech Support | 0.183 | 0.254 | 0.213 | 283 |

Confusion Matrix:
```
[[398, 241, 250],
 [113, 83, 72],
 [143, 68, 72]]
```

## Holdout Metrics (holdout_metrics.json)
| Target | Accuracy | Macro F1 |
|--------|----------|----------|
| Priority | 0.266 | 0.266 |
| Department | 0.382 | 0.323 |

## Error Analysis (validation split rebuilt)
See `reports/error_analysis_v1.0.3.md` & `.json`.

(Those error analysis values differ from training metrics, indicating the validation reconstruction produced different splits—investigate seed / split stratification mismatch. The unexpectedly high metrics in error analysis compared to training metrics suggests a bug: error analysis script may be using full dataset instead of leftover subset or model mismatch.)

### Observed Discrepancy
Error analysis Priority macro F1 reported ≈0.636 vs training 0.244.
Likely causes:
1. Re-splitting produced *new* validation set that the model effectively overfits due to enriched tokens correlating strongly with labels (or leak not fully excluded for priority?).
2. Error analysis script uses different text preprocessing vs training (exclusion patterns not applied? Should not affect priority drastically).
3. Misalignment: We trained with department token exclusion but error analysis script reconstruction didn't replicate TRAIN/VAL separation; instead it took a fresh sample.

### Next Action on Discrepancy
- Extract and persist original train/val indices during training (already done for earlier version? Add for v1.0.3 if missing) and use those in error analysis when available.
- Update error analysis script to check for `train_indices.txt` / `val_indices.txt` in model directory and use them if present; fall back to re-split otherwise with explicit warning.

## Key Findings
- Priority model under-performs: near random macro F1 (0.24–0.26) across validation/holdout.
- Department model modest performance (macro F1 ~0.32) after leakage removal.
- Large discrepancy between training metrics and reconstructed validation evaluation indicates need for deterministic split reuse.

## Recommendations (Aligned with Tier 1)
1. Persist and reuse split indices for reproducible validation metrics in analysis tools.
2. Engineer priority-specific lexical & semantic features (keyword tokens, structural ratios).
3. Hyperparameter exploration (LogReg C grid, linear SVM, NB baseline).
4. Calibrate probabilities for better triage thresholds.
5. Consider department-focused features (channel + product interactions) after ensuring no leakage.

## Risk Watch
- Hidden leakage in priority still possible if enrichment tokens encode priority indirectly.
- Overfitting risk if using regenerated splits inconsistently.

## Artifacts
- Model: `models/v1.0.3/`
- Training metrics: `models/v1.0.3/metrics.json`
- Holdout metrics: `models/v1.0.3/holdout_metrics.json`
- Error analysis: `reports/error_analysis_v1.0.3.{json,md}`

---
Generated automatically as part of Tier 0 baseline documentation.
