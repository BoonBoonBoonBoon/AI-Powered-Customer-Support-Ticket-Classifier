#!/usr/bin/env python3
"""Mutual Information Ranking for Enrichment Tokens

Computes mutual information between candidate token patterns and department labels
(to detect potential leakage-like features) and optionally priority labels.

Usage:
  python scripts/token_mi_ranking.py \
    --data data/enriched_customer_tickets.csv \
    --patterns __product_[a-z0-9_]+ __channel_[a-z0-9_]+ __csat_[a-z0-9_]+ __type_[a-z0-9_]+ \
    --top-k 30 --output reports/token_mi_ranking.json

Outputs JSON list sorted by department_mi desc.

Note: naive regex presence -> binary feature; uses sklearn mutual_info_classif.
"""
from __future__ import annotations
import argparse
import json
import re
from pathlib import Path
from typing import List, Dict
import pandas as pd
import numpy as np
from sklearn.feature_selection import mutual_info_classif
from sklearn.preprocessing import LabelEncoder


def build_features(df: pd.DataFrame, patterns: List[str]) -> Dict[str, np.ndarray]:
    feats = {}
    for pat in patterns:
        rx = re.compile(pat)
        feats[pat] = df['description'].astype(str).apply(lambda t: 1 if rx.search(t) else 0).to_numpy()
    return feats


def compute_mi(y, X_cols: Dict[str, np.ndarray]) -> Dict[str, float]:
    out = {}
    for name, col in X_cols.items():
        out[name] = float(mutual_info_classif(col.reshape(-1,1), y, discrete_features=True, random_state=42)[0])
    return out


def main():
    ap = argparse.ArgumentParser(description='Rank candidate enrichment token regexes by mutual information with department label.')
    ap.add_argument('--data', required=True)
    ap.add_argument('--patterns', nargs='+', required=True)
    ap.add_argument('--output', default='reports/token_mi_ranking.json')
    ap.add_argument('--top-k', type=int, default=50)
    args = ap.parse_args()

    df = pd.read_csv(args.data)
    feats = build_features(df, args.patterns)
    dep_enc = LabelEncoder().fit(df['department'])
    y_dep = dep_enc.transform(df['department'])
    mi_dep = compute_mi(y_dep, feats)
    pri_enc = LabelEncoder().fit(df['priority'])
    y_pri = pri_enc.transform(df['priority'])
    mi_pri = compute_mi(y_pri, feats)
    rows = []
    for name in feats:
        presence_rate = float(feats[name].mean())
        rows.append({
            'pattern': name,
            'presence_rate': presence_rate,
            'department_mi': mi_dep[name],
            'priority_mi': mi_pri[name]
        })
    rows.sort(key=lambda r: r['department_mi'], reverse=True)
    if args.top_k:
        rows = rows[:args.top_k]
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(rows, f, indent=2)
    print(f"Wrote {len(rows)} rows -> {args.output}")

if __name__ == '__main__':
    main()
