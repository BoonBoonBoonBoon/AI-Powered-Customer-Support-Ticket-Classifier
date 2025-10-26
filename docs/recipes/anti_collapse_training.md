# Anti-collapse training recipe (dual-head transformer)

This recipe stabilizes the priority head against majority-class collapse and adds options to improve department fairness, while keeping priority as the selection metric.

Applies to: `scripts/train_transformer.py` (DistilBERT/Roberta encoders with dual heads)

## Core settings (priority-focused)

- Weighted sampler: priority (`--weighted-sampler priority`)
- Focal loss (priority): gamma 1.5 (`--focal-priority-gamma 1.5`)
- Class weights: priority=auto, department=none
  - `--class-weight-priority auto`
  - `--class-weight-department none`
- Loss weights: priority=2.0, department=0.3
  - `--loss-weight-priority 2.0` `--loss-weight-department 0.3`
- Department-loss warmup: 2 epochs (`--dept-loss-warmup-epochs 2`)
- Label smoothing: 0.05 (`--label-smoothing 0.05`)
- Selection metric: priority (`--select-metric priority`)
- Device: CUDA if available (`--device auto|cuda`)

Recommended model: `distilroberta-base`, LR=3e-5, max_len=256, batch_size 12, grad_accum 2.

### Example (PowerShell)

```powershell
.\.venv312\Scripts\python.exe scripts\train_transformer.py ^
  --data data\enriched_customer_tickets.csv ^
  --model-name distilroberta-base ^
  --epochs 6 --batch-size 12 --grad-accum 2 ^
  --lr 3e-5 --warmup-ratio 0.1 --max-len 256 ^
  --output-version t1.gpu.v1 ^
  --exclude-pattern "__department_[a-z0-9_]+" ^
  --exclude-pattern "__dept_[a-z0-9_]+" ^
  --exclude-pattern "__type_[a-z0-9_]+" ^
  --class-weight-priority auto --class-weight-department none ^
  --label-smoothing 0.05 --select-metric priority ^
  --loss-weight-priority 2.0 --loss-weight-department 0.3 ^
  --dept-loss-warmup-epochs 2 --weighted-sampler priority ^
  --focal-priority-gamma 1.5 --device cuda
```

Diagnostics: the trainer writes `epoch_metrics.jsonl` logging per-epoch prediction/label distributions and macro-F1 per head. Watch for any single-class domination across epochs; if seen, raise gamma slightly (e.g., 1.7–2.0).

## Department fairness variant

Add light balancing for department without sacrificing priority:

- Class weights: `--class-weight-department auto`
- Loss weight (dept): 0.4 (`--loss-weight-department 0.4`)
- Focal loss (dept): gamma 0.7 (`--focal-department-gamma 0.7`)
- Early stopping: patience 2–3 (`--early-stopping-patience 2`)

Example:

```powershell
.\.venv312\Scripts\python.exe scripts\train_transformer.py ^
  --data data\enriched_customer_tickets.csv ^
  --model-name distilroberta-base ^
  --epochs 8 --batch-size 12 --grad-accum 2 ^
  --lr 3e-5 --warmup-ratio 0.1 --max-len 256 ^
  --output-version t1.gpu.dept2 ^
  --exclude-pattern "__department_[a-z0-9_]+" ^
  --exclude-pattern "__dept_[a-z0-9_]+" ^
  --exclude-pattern "__type_[a-z0-9_]+" ^
  --class-weight-priority auto --class-weight-department auto ^
  --label-smoothing 0.05 --select-metric priority ^
  --loss-weight-priority 2.0 --loss-weight-department 0.4 ^
  --dept-loss-warmup-epochs 2 --weighted-sampler priority ^
  --focal-priority-gamma 1.5 --focal-department-gamma 0.7 ^
  --early-stopping-patience 2 --device cuda
```

Notes:
- Expect realistic department macro-F1 to be lower than a collapsed majority predictor but significantly more fair across classes.
- If department remains under-predicting minority classes, consider raising dept loss weight to 0.45 or gamma to 0.9.

## Priority uplift tweaks

Once collapse is mitigated and department is reasonably fair, try:
- Slightly higher focal for priority: `--focal-priority-gamma 1.7`
- Or LR=4e-5 (keep warmup=0.1) with early stopping 2–3
- Keep epochs modest (6–10) and rely on early stopping; long runs rarely outperform early peaks in our data.

## Leakage guard (critical)

Always exclude enrichment tokens that correlate with department:
```
--exclude-pattern "__department_[a-z0-9_]+" \
--exclude-pattern "__dept_[a-z0-9_]+" \
--exclude-pattern "__type_[a-z0-9_]+"
```
These are applied to the description text (not the title) during tokenization.

## Results reference (from this repo)

- Priority-focused 20-epoch run (best epoch 5): macro-F1 ≈ 0.2588 (balanced predictions, no collapse).
- Dept-aware (softer): macro-F1 (priority) ≈ 0.2544, macro-F1 (department) ≈ 0.2427 with more equitable class predictions.

SBERT baseline (MiniLM + LR) for reference on the same split:
- Priority macro-F1 ≈ 0.234 (lower than transformer recipe)
- Department macro-F1 ≈ 0.837 (likely inflated by enrichment leakage if exclusions are not applied consistently)
