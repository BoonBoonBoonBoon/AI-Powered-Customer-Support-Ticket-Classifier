# Training Findings — 2025-10-24

Time window: 2025-10-22 → 2025-10-24
Author: Automation

## Context
- Baseline (sklearn v1.0.10) is strong relative to current transformers.
- Transformer canaries had leakage initially (perfect department scores), fixed by exclusion patterns.
- Latest completed transformer (t1.0.3) improved but still below baseline on priority macro-F1.
- A tuned run (t1.0.4) was started as a PowerShell background job and exited early (code 1) without logs.

## Evidence
- sklearn v1.0.10 (validation):
  - Priority macro-F1 0.2570 | Accuracy 0.2574
  - Department macro-F1 0.3179 | Accuracy 0.3926
- transformer t1.0.3 (validation):
  - Priority macro-F1 0.2348 | Accuracy 0.2462
  - Department macro-F1 0.2506 | Accuracy 0.6021 (majority collapse pattern)
- transformer t1.0.2 (validation):
  - Priority macro-F1 0.0996 | Accuracy 0.2420
  - Department macro-F1 0.2506 | Accuracy 0.6021
- t1.0.4 job status: Completed quickly with exit code 1; no `train.log` produced at `models/transformers/t1.0.4/`.

## Likely root causes
1. Department head collapse after leak removal
   - Macro-F1 ~0.25 with high accuracy indicates a majority-class predictor; gradients from this head are noisy and hurt shared encoder learning.
2. Equal loss weighting for both heads
   - Summing losses equally means the weak department task competes with priority and can drag optimization away from the main objective.
3. Selection/optimization mismatch
   - Earlier runs saved checkpoints by combined score; this may choose suboptimal checkpoints for priority. Selection should be based on priority macro-F1.
4. Under-training and schedule
   - 3 epochs with lr=5e-5 and short warmup may underfit. Slightly longer training and gentler schedule typically help.
5. Input formatting
   - Using literal markers "[TITLE]/[DESC]" creates unknown tokens; better to feed title/description as two sequences (uses [SEP]) for clearer structure.

## Action plan
1. Reweight multi-task losses (priority-focused)
   - New flags: `--loss-weight-priority 2.0`, `--loss-weight-department 0.5`.
   - Loss = w_p * loss_p + w_d * loss_d; continue class weights (auto) and label smoothing.
2. Improve input formatting
   - Tokenize with `tokenizer(text, text_pair)` so [SEP] is used; keep exclusion patterns on description.
3. Train longer with gentler schedule
   - 5 epochs, `lr=3e-5`, `warmup_ratio=0.1`, `batch_size=16`, `grad_accum=2` (effective 32).
4. Selection target
   - Save best by `--select-metric priority`.
5. Execution reliability
   - First run a 1-epoch smoke (`t1.0.4-smoke`) in foreground to capture any errors; then kick off full `t1.0.4` pipeline.

## Success criteria
- No leakage (department macro-F1 well below 1.0 with realistic accuracy).
- Priority macro-F1 ≥ sklearn baseline (≥ 0.2570) or a clear upward trend vs t1.0.3.
- All tests pass; artifacts and manifest written correctly.

## Next steps
- Implement code changes (loss weights, [SEP] input).
- Smoke-run 1 epoch with new flags and verify logs/metrics.
- Launch full t1.0.4 pipeline and monitor progress.
- If still below baseline, consider roberta-base and early stopping on priority macro-F1.
