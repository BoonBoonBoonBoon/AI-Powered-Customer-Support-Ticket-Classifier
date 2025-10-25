# Local CPU Quickstart (No Colab)

This guide trains a strong baseline using SBERT embeddings + scikit‑learn on CPU in minutes, and saves metrics/artifacts.

## Prereqs
- Python venv activated for this repo
- Packages installed (we added sentence-transformers to requirements)

Optional: install/update deps now:

```powershell
# From repo root (Windows PowerShell)
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

## Run SBERT + sklearn training

```powershell
# From repo root
& ".\.venv\Scripts\python.exe" "scripts\train_sbert_sklearn.py" --data "data\enriched_customer_tickets.csv" --encoder "sentence-transformers/all-MiniLM-L6-v2" --output-version s1.0.0 --batch-size 64
```

Artifacts will be written to `models/sbert/s1.0.0/`:
- `metrics.json` – macro‑F1 and accuracy for priority and department
- `label_mappings.json`
- `priority_clf.joblib`, `department_clf.joblib`
- `encoder.json` – SBERT model name used for embeddings

## Inspect metrics

```powershell
Get-Content -LiteralPath "models\sbert\s1.0.0\metrics.json"
```

## Notes
- This approach is fast on CPU and competitive. If/when a GPU is available again, you can return to transformer fine‑tuning.
- You can change `--encoder` to other SentenceTransformers like `all-MiniLM-L12-v2` for a small boost (slower/large), or `all-MiniLM-L6-v2` for speed.
