# Data Import & Training Workflow

This guide explains how to profile, normalize, split, train, evaluate, and integrate a new customer support tickets CSV dataset into the classifier system.

## 1. Source CSV Requirements
- Minimum columns (raw headers may differ, map via profile script):
  - Title / Subject (mapped to `title`)
  - Description / Body (mapped to `description`)
  - Priority label (mapped to `priority`) — values like: Urgent, High, Medium, Low
  - Department / Category (mapped to `department`) — e.g.: Tech Support, Billing, Sales
- Optional columns: customer email, created_at, ticket_id (ignored unless used for dedupe)

## 2. Profiling & Normalization
Run the profiling script to inspect distribution and produce a normalized CSV.

PowerShell:
```
python scripts/profile_dataset.py ^
  --input data/customer_support_tickets.csv ^
  --output-normalized data/normalized_tickets.csv ^
  --profile-json data/dataset_profile.json
```

What it does:
- Detects likely header names and maps them → title, description, priority, department
- Lowercases + strips whitespace
- Collapses duplicate whitespace
- Drops rows missing title or description
- Reports duplicate percentage, class distributions, row counts

Inspect `data/dataset_profile.json` for:
- `class_distribution.priority` and `class_distribution.department`
- Duplicate rate and null counts
- Use this to decide enrichment or rebalancing actions

## 3. Deduplication (Optional Enhanced Step)
If duplicates > 5%:
- Add a hash column: `hash = sha1(lower(title + description))`
- Keep first occurrence per hash
- Optionally log a `dedupe_report.json` with counts

## 4. Train / Validation / Holdout Split
Implemented advanced splitting:
- Optional holdout via `--holdout-frac` (stratified by priority)
- Example ratios with `--holdout-frac 0.15` & default validation 0.2 of remainder:
  - Holdout: 15%
  - Validation: 17% (0.2 of remaining 85%)
  - Train: 68%

Training script flags now available:
```
--holdout-frac 0.15          # Reserve 15% holdout
--min-class-samples 5        # Warn on under-represented classes
--class-weight balanced      # Enable class weighting
--augment-length             # Add length bucket tokens
```

## 5. Class Weighting Strategy
Why: Handle imbalance (e.g., few Urgent vs many Low).
Implementation plan:
- Use scikit-learn `class_weight='balanced'` for Logistic Regression priority model.
- Department model optionally also weighted if distribution skew > 2.5x between max and min.

## 6. Feature Engineering Roadmap
Implemented:
- Title + description fusion with markers
- Length bucket synthetic tokens via `--augment-length`
- Bigram TF-IDF (1,2) n-gram range

Future roadmap:
- Keyword flag feature tokens (`has_payment`, etc.)
- Calibration of probability outputs

## 7. Training Command (Baseline)
```
python train.py \
  --data data/normalized_tickets.csv \
  --priority-col priority \
  --department-col department \
  --title-col title \
  --description-col description \
  --output-dir models/v1.0.1
```

Artifacts produced:
- Vectorizers & model pickles (`*_vectorizer.joblib`, `*_model.joblib`, encoders)
- `metrics.json` (validation)
- `holdout_metrics.json` (if `--holdout-frac > 0`)
- `model_metadata.json` (includes flags used)
- Sample predictions printed to stdout

## 8. Evaluation
Two existing scripts:
- Smoke test: `python scripts/smoke_test.py`
- Bulk eval (expects labeled CSV):
  ```
  python scripts/bulk_eval.py \
    --data data/normalized_tickets.csv \
    --title-col title \
    --description-col description \
    --priority-col priority \
    --department-col department \
    --model-dir models/v1.0.1
  ```

Outputs:
- Accuracy, macro-F1 thresholds (CI enforces baseline)
- Misclassification samples (for error analysis)

## 9. Continuous Integration Hook
- CI workflow runs bulk evaluation on PRs
- Future: Add macro-F1 gating (e.g., require >= 0.70) and holdout delta guard (validation vs holdout gap < 5pp)

## 10. Updating the Service
After a new model version:
- Place artifacts in `models/vX.Y.Z/`
- Update environment variable or config referencing active model path (if applicable)
- Deploy (Railway / Docker) — health endpoint should report readiness

## 11. Data Quality Checklist (Run Before Training)
- Priority & department coverage > 98% non-null
- No single class < 1% (else consider synthetic augmentation or grouping)
- Duplicates < 5% after dedupe
- Train/validation/holdout each contain all classes
- Drift check (if updating): Compare previous vs new class distributions (KL divergence flagged if > 0.2)

## 12. Future Enhancements (Planned)
- Active learning pool: store low-confidence predictions for human labeling
- Drift monitoring: periodic distribution comparison of live predictions vs training
- Probability calibration (Platt or isotonic) for more reliable confidence scores
- SBOM + vulnerability scanning integrated into CI security job

## 13. Troubleshooting
| Issue | Cause | Action |
|-------|-------|--------|
| Class not found error | Column mapping wrong | Inspect `dataset_profile.json` mappings |
| Validation collapse | Imbalance after split | Increase `--min-class-samples` or reduce holdout fraction |
| Poor recall on rare class | Under-represented | Enable class weighting, gather more samples |
| High overfit (val >> holdout) | Data leakage or small holdout | Re-check split, increase holdout size |

## 14. Quick Start TL;DR
```
python scripts/profile_dataset.py --input data/customer_support_tickets.csv --output-normalized data/normalized_customer_tickets.csv --export-json-profile data/customer_profile.json

# (Optional) Enrich with structured feature tokens
python scripts/enrich_train_features.py --input-raw data/customer_support_tickets.csv \
  --include-product --include-ticket-type --include-channel --include-csat \
  --output data/enriched_customer_tickets.csv

# Train on enriched dataset (PowerShell syntax shown)
$env:MODEL_VERSION="1.0.2"
python train.py --data data/enriched_customer_tickets.csv --holdout-frac 0.15 \
  --class-weight balanced --augment-length --min-class-samples 10

# Error analysis (holdout subset)
python scripts/error_analysis.py --data data/enriched_customer_tickets.csv \
  --model-dir models/v1.0.2 --subset-indices models/v1.0.2/holdout_indices.txt \
  --report-md reports/error_analysis_v1.0.2.md
```

---
This document will be updated as advanced training enhancements are merged (holdout metrics, class weighting flags, feature augmentation, calibration).
