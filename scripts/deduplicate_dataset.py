#!/usr/bin/env python3
"""Simple de-duplication pass using Jaccard similarity over token shingles.

Goal: Identify near-duplicate tickets and optionally output a filtered CSV.

Usage:
  python scripts/deduplicate_dataset.py \
    --data data/enriched_customer_tickets.csv \
    --output data/enriched_customer_tickets_dedup.csv \
    --report reports/dedup_report.json \
    --jaccard-threshold 0.85

Approach:
  * Lowercase + whitespace normalize title + description
  * Build word 3-gram shingles set per record
  * Greedy pass: keep first occurrence; mark later records duplicate if Jaccard >= threshold

Report fields:
  total, duplicates_removed, percent_removed, threshold
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import pandas as pd
import re
from typing import List, Set


def normalize(text: str) -> str:
    text = (text or '').lower().strip()
    text = re.sub(r"\s+", " ", text)
    return text

def shingles(text: str, n: int = 3) -> Set[str]:
    parts = text.split()
    if len(parts) < n:
        return set([" ".join(parts)]) if parts else set()
    return {" ".join(parts[i:i+n]) for i in range(len(parts)-n+1)}


def deduplicate(df: pd.DataFrame, threshold: float) -> tuple[pd.DataFrame, int]:
    seen_sets: List[Set[str]] = []
    keep_flags = []
    dup_count = 0
    for _, row in df.iterrows():
        combined = normalize(f"{row['title']} {row['description']}")
        sh = shingles(combined)
        is_dup = False
        for prev in seen_sets:
            if not prev or not sh:
                continue
            inter = len(prev & sh)
            union = len(prev | sh)
            if union == 0:
                continue
            j = inter / union
            if j >= threshold:
                is_dup = True
                break
        if is_dup:
            dup_count += 1
            keep_flags.append(False)
        else:
            keep_flags.append(True)
            seen_sets.append(sh)
    return df[keep_flags].reset_index(drop=True), dup_count


def main():
    ap = argparse.ArgumentParser(description='Near-duplicate removal for ticket dataset (Jaccard over word 3-grams).')
    ap.add_argument('--data', required=True)
    ap.add_argument('--output', required=True)
    ap.add_argument('--report', default='reports/dedup_report.json')
    ap.add_argument('--jaccard-threshold', type=float, default=0.85)
    args = ap.parse_args()

    df = pd.read_csv(args.data)
    filtered, removed = deduplicate(df, args.jaccard_threshold)
    filtered.to_csv(args.output, index=False)
    stats = {
        'total': int(len(df)),
        'duplicates_removed': int(removed),
        'percent_removed': round(removed / len(df) * 100, 3) if len(df) else 0.0,
        'threshold': args.jaccard_threshold,
        'output': args.output
    }
    Path(args.report).parent.mkdir(parents=True, exist_ok=True)
    with open(args.report, 'w', encoding='utf-8') as f:
        json.dump(stats, f, indent=2)
    print(json.dumps(stats, indent=2))

if __name__ == '__main__':
    main()
