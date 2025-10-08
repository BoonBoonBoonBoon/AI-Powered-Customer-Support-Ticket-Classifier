#!/usr/bin/env python3
"""Hard Negative Mining Utility

Scans a labeled CSV and collects high-confidence misclassifications to
bootstrap targeted re-labeling or feature engineering.

Output: data/hard_negatives.csv (appended or created)

Usage:
  python scripts/hard_negative_mining.py \
    --data data/enriched_customer_tickets.csv \
    --models-dir models --version 1.0.6 \
    --confidence-threshold 0.60 \
    --output data/hard_negatives.csv

Columns written:
  title,description,true_priority,pred_priority,pred_priority_conf,
  true_department,pred_department,pred_department_conf,model_version
"""
from __future__ import annotations
import argparse
import csv
from pathlib import Path
import pandas as pd
from app.models.classifier import TicketClassifier
from app.config import settings


def load_model(models_dir: str, version: str) -> TicketClassifier:
    clf = TicketClassifier()
    clf.load_models(f"{models_dir}/v{version}")
    return clf


def mine(df: pd.DataFrame, clf: TicketClassifier, conf_thresh: float, version: str) -> list[dict]:
    rows = []
    for _, r in df.iterrows():
        p_pred, d_pred, p_conf, d_conf = clf.predict(r['title'], r['description'])
        p_true = r.get('priority')
        d_true = r.get('department')
        if p_true and p_pred != p_true and p_conf >= conf_thresh:
            rows.append({
                'title': r['title'], 'description': r['description'],
                'true_priority': p_true, 'pred_priority': p_pred, 'pred_priority_conf': round(p_conf,3),
                'true_department': d_true, 'pred_department': d_pred, 'pred_department_conf': round(d_conf,3),
                'model_version': version
            })
        if d_true and d_pred != d_true and d_conf >= conf_thresh:
            rows.append({
                'title': r['title'], 'description': r['description'],
                'true_priority': p_true, 'pred_priority': p_pred, 'pred_priority_conf': round(p_conf,3),
                'true_department': d_true, 'pred_department': d_pred, 'pred_department_conf': round(d_conf,3),
                'model_version': version
            })
    return rows


def main():
    ap = argparse.ArgumentParser(description='Collect high-confidence misclassifications for relabeling / feature ideas.')
    ap.add_argument('--data', required=True)
    ap.add_argument('--models-dir', default='models')
    ap.add_argument('--version', default=settings.MODEL_VERSION)
    ap.add_argument('--confidence-threshold', type=float, default=0.6)
    ap.add_argument('--output', default='data/hard_negatives.csv')
    args = ap.parse_args()

    df = pd.read_csv(args.data)
    clf = load_model(args.models_dir, args.version)
    findings = mine(df, clf, args.confidence_threshold, args.version)
    if not findings:
        print('No high-confidence misclassifications found at given threshold.')
        return
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    file_exists = out_path.exists()
    with out_path.open('a', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=[
            'title','description','true_priority','pred_priority','pred_priority_conf',
            'true_department','pred_department','pred_department_conf','model_version'
        ])
        if not file_exists:
            writer.writeheader()
        writer.writerows(findings)
    print(f"Appended {len(findings)} hard negatives -> {out_path}")

if __name__ == '__main__':
    main()
