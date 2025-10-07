#!/usr/bin/env python3
"""Department noise audit.

Evaluates the effect of excluding specified regex patterns (e.g., structured enrichment tokens)
from department training text while leaving priority untouched.

Produces a small comparison table for different exclusion sets versus a baseline.

Usage:
  python scripts/department_noise_audit.py \
    --data data/enriched_customer_tickets.csv \
    --reference-version 1.0.4 \
    --models-dir models \
    --patterns "__product_[a-z0-9_]+" "__channel_[a-z0-9_]+" --output reports
"""
from __future__ import annotations
import argparse
import json
import re
from pathlib import Path
from typing import List, Dict, Any, Tuple
import pandas as pd
import numpy as np
from sklearn.metrics import classification_report
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder

from app.config import settings


def _preprocess(t: str) -> str:
    t = (t or "").strip().lower()
    return re.sub(r"\s+", " ", t)


def load_split(df: pd.DataFrame, models_dir: Path, reference_version: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    vdir = models_dir / f"v{reference_version}"
    val_idx = vdir / "val_indices.txt"
    train_idx = vdir / "train_indices.txt"
    if val_idx.exists() and train_idx.exists():
        try:
            val_indices = [int(x) for x in val_idx.read_text().splitlines() if x.strip()]
            train_indices = [int(x) for x in train_idx.read_text().splitlines() if x.strip()]
            return df.loc[train_indices], df.loc[val_indices]
        except Exception:
            pass
    from sklearn.model_selection import train_test_split
    return train_test_split(df, test_size=settings.VALIDATION_SPLIT, random_state=settings.RANDOM_SEED, stratify=df['priority'])


def build_texts(train_df: pd.DataFrame, val_df: pd.DataFrame, exclude_regexes: List[str]) -> Tuple[List[str], List[str], List[str], List[str]]:
    compiled = [re.compile(p) for p in exclude_regexes]
    pr_train, pr_val, dep_train, dep_val = [], [], [], []
    for mode, frame in (("train", train_df), ("val", val_df)):
        for _, row in frame.iterrows():
            title = _preprocess(row['title'])
            desc = _preprocess(row['description'])
            pr_text = f"title: {title}\nbody: {desc}".strip()
            dep_desc = desc
            for cre in compiled:
                dep_desc = cre.sub(" ", dep_desc)
            dep_desc = re.sub(r"\s+", " ", dep_desc).strip()
            dep_text = f"title: {title}\nbody: {dep_desc}".strip()
            if mode == "train":
                pr_train.append(pr_text)
                dep_train.append(dep_text)
            else:
                pr_val.append(pr_text)
                dep_val.append(dep_text)
    return pr_train, pr_val, dep_train, dep_val


def evaluate(train_df: pd.DataFrame, val_df: pd.DataFrame, exclude: List[str]) -> Dict[str, Any]:
    pr_train, pr_val, dep_train, dep_val = build_texts(train_df, val_df, exclude)
    pr_enc = LabelEncoder(); dep_enc = LabelEncoder()
    y_pr_tr = pr_enc.fit_transform(train_df['priority']); y_pr_val = pr_enc.transform(val_df['priority'])
    y_dep_tr = dep_enc.fit_transform(train_df['department']); y_dep_val = dep_enc.transform(val_df['department'])
    pr_vec = TfidfVectorizer(ngram_range=(1,2), min_df=2); dep_vec = TfidfVectorizer(ngram_range=(1,2), min_df=2)
    X_pr_tr = pr_vec.fit_transform(pr_train); X_pr_val = pr_vec.transform(pr_val)
    X_dep_tr = dep_vec.fit_transform(dep_train); X_dep_val = dep_vec.transform(dep_val)
    model_pr = LogisticRegression(max_iter=200, class_weight='balanced')
    model_dep = LogisticRegression(max_iter=200, class_weight='balanced')
    model_pr.fit(X_pr_tr, y_pr_tr)
    model_dep.fit(X_dep_tr, y_dep_tr)
    pr_report = classification_report(y_pr_val, model_pr.predict(X_pr_val), output_dict=True)
    dep_report = classification_report(y_dep_val, model_dep.predict(X_dep_val), output_dict=True)
    return {
        'exclude': exclude,
        'priority_macro_f1': pr_report['macro avg']['f1-score'],
        'department_macro_f1': dep_report['macro avg']['f1-score'],
        'department_accuracy': dep_report['accuracy'],
    }


def main():
    ap = argparse.ArgumentParser(description='Audit impact of excluding regex patterns from department text.')
    ap.add_argument('--data', required=True)
    ap.add_argument('--models-dir', default='models')
    ap.add_argument('--reference-version', default=settings.MODEL_VERSION)
    ap.add_argument('--patterns', nargs='*', default=[], help='Candidate exclusion patterns to test (space separated)')
    ap.add_argument('--output', default='reports')
    args = ap.parse_args()

    df = pd.read_csv(args.data)
    train_df, val_df = load_split(df, Path(args.models_dir), args.reference_version)

    # Evaluate baseline (no exclusions) and cumulative + individual patterns
    results: List[Dict[str, Any]] = []
    results.append(evaluate(train_df, val_df, []))
    for p in args.patterns:
        results.append(evaluate(train_df, val_df, [p]))
    if len(args.patterns) > 1:
        results.append(evaluate(train_df, val_df, args.patterns))

    results_sorted = sorted(results, key=lambda r: r['department_macro_f1'], reverse=True)
    out_dir = Path(args.output); out_dir.mkdir(exist_ok=True, parents=True)
    (out_dir / 'department_noise_audit.json').write_text(json.dumps(results_sorted, indent=2), encoding='utf-8')

    lines = ["# Department Noise Audit", "", "| Exclusions | Dept Macro F1 | Dept Acc | Priority Macro F1 |", "|-----------|---------------|---------|------------------|"]
    for r in results_sorted:
        lines.append(f"| {','.join(r['exclude']) if r['exclude'] else '(none)'} | {r['department_macro_f1']:.4f} | {r['department_accuracy']:.4f} | {r['priority_macro_f1']:.4f} |")
    (out_dir / 'department_noise_audit.md').write_text("\n".join(lines), encoding='utf-8')
    print("Wrote department_noise_audit.{json,md}")

if __name__ == '__main__':
    main()
