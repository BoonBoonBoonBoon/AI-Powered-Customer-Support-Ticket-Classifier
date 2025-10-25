## Transformer runs summary (Oct 25, 2025)

This note summarizes the recent transformer training runs vs the established sklearn baseline. Metrics are from validation splits; macro-F1 emphasizes balanced performance across classes.

### Snapshot of results

- sklearn:v1.0.10
  - Priority: macro-F1 0.2570, Acc 0.2574
  - Department: macro-F1 0.3179, Acc 0.3926

- transformer:t1.0.3 (DistilBERT, 2-seq, leak-guard)
  - Priority: macro-F1 0.2348, Acc 0.2462
  - Department: macro-F1 0.2506, Acc 0.6021

- transformer:t1.0.4 (DistilBERT, tuned loss weights, sampler, warmup)
  - Priority: macro-F1 0.1673, Acc 0.2290
  - Department: macro-F1 0.2933, Acc 0.5531

- transformer:t1.0.4-smoke (1 epoch sanity)
  - Priority: macro-F1 0.1392, Acc 0.2397
  - Department: macro-F1 0.3047, Acc 0.3377

- transformer:t1.0.5r (RoBERTa-base, 3 epochs)
  - Priority: macro-F1 0.1109, Acc 0.2426
  - Department: macro-F1 0.1117, Acc 0.2013

Notes:
- t1.0.0 and t1.0.1 were leak-affected on department (macro-F1 ≈ 1.0); these are not comparable.
- t1.0.2 removed leak and established a clean baseline but underperformed.
- t1.0.3 was the strongest transformer so far on priority macro-F1 (0.2348), still slightly below sklearn.
- t1.0.4 introduced additional training tweaks; priority regressed while department improved.
- t1.0.5r (roberta-base) did not improve; it collapsed into predicting a dominant class (high recall for a single class, near-zero for others).

### What likely happened

- Class collapse on priority (and department for t1.0.5r):
  - Reports show very high recall for one class and near-zero for others. This indicates the model or the sampling/loss setup biased toward a majority class.
  - Weighted sampler by priority plus focal loss (gamma 1.5) and loss-weighting may have interacted poorly with label smoothing and warmup windows.

- Two-sequence tokenization is correct, but the training signal for department may still dominate without careful balancing; loss warmup helped, but not enough.

- CPU-only training is slow, which limited our ability to iterate on epochs and LR schedules rapidly.

### Recommendations (next iteration)

Short-term, low-risk changes:
- Turn off focal loss initially (gamma 0) and keep plain CE with modest label smoothing (0.05).
- Remove weighted sampler; rely on class weights only (auto) or try sampler OR weights, not both.
- Keep dept-loss warmup for 1–2 epochs but raise priority loss weight only slightly (e.g., 1.5) and set department to 1.0 after warmup to avoid starvation.
- Save best checkpoint by priority macro-F1 each epoch (already done), and add early stopping by patience.
- Try distilroberta-base with LR 3e-5, epochs 4–5, max_len 256.

Medium-term experiments:
- Increase training length with cosine or linear schedule and validate every half-epoch to detect progress earlier.
- Try RoBERTa-base again but with simpler loss (no focal, no sampler) and tune LR (2e-5 → 3e-5) and warmup ratio (0.06–0.1).
- Evaluate inference-time exclusion parity (already aligned) and consider mild text normalization on titles too (lowercasing)
  if not already handled by the tokenizer.

Quality gates and serving:
- Keep production on sklearn:v1.0.10.
- Do not promote any transformer to staging yet; t1.0.3 is closest but still under baseline on priority macro-F1.
- Continue logging and compare via `scripts/compare_models.py` after each run.

### Next concrete run (proposal)

- distilroberta-base, epochs=5, batch_size=16, grad_accum=2, LR=3e-5, warmup_ratio=0.1, max_len=256
- loss: CE (no focal), label_smoothing=0.05
- class weights: priority=auto, department=auto
- weighted sampler: none
- loss weights: priority=1.5, department=1.0; dept_loss_warmup_epochs=1
- select_metric=priority

Success criteria:
- Priority macro-F1 >= 0.26 (clear win over sklearn) or at least >= 0.24 with positive trend and stable confusion matrix distribution.

If still under baseline:
- Lower max_len to 192 to improve throughput and regularization; increase epochs to 6.
- Sweep LR (2e-5, 3e-5) and label smoothing (0.0, 0.05).
