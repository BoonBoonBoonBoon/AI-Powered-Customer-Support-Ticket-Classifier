#!/usr/bin/env python3
"""
Leakage Audit: detect tokens highly correlated with labels (department, priority).

Usage (PowerShell):
  .\.venv312\Scripts\python.exe scripts\leakage_audit.py \
    --data data\enriched_customer_tickets.csv \
    --target department --topk 30 \
    --exclude-pattern "__department_[a-z0-9_]+" \
    --exclude-pattern "__dept_[a-z0-9_]+" \
    --exclude-pattern "__type_[a-z0-9_]+"

This will print the top tokens by chi^2 for the chosen target and flag those
matching any exclude pattern. Run for both targets to spot risky markers.
"""
from __future__ import annotations
import argparse, re
from typing import List
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_selection import chi2


def _apply_exclusions(text: str, compiled_patterns: List[re.Pattern] | None) -> str:
    if not compiled_patterns:
        return text
    for cre in compiled_patterns:
        text = cre.sub(" ", text)
    return re.sub(r"\s+", " ", text).strip()


def main():
    ap = argparse.ArgumentParser(description="Audit potential leakage tokens by chi^2 association")
    ap.add_argument("--data", required=True)
    ap.add_argument("--target", choices=["department","priority"], default="department")
    ap.add_argument("--topk", type=int, default=30)
    ap.add_argument("--min_df", type=int, default=5)
    ap.add_argument("--exclude-pattern", action="append", default=[])
    args = ap.parse_args()

    df = pd.read_csv(args.data)
    texts = (df["title"].astype(str).str.strip() + "\n\n" + df["description"].astype(str).str.strip()).tolist()
    compiled = [re.compile(p, flags=re.IGNORECASE) for p in (args.exclude_pattern or [])]
    texts = [_apply_exclusions(t, compiled) for t in texts]
    y = df[args.target].astype(str).values

    vec = CountVectorizer(lowercase=True, token_pattern=r"(?u)\b\w+\b", min_df=args.min_df)
    X = vec.fit_transform(texts)
    scores, _ = chi2(X, y)
    vocab = vec.get_feature_names_out()
    pairs = sorted(zip(vocab, scores), key=lambda x: x[1], reverse=True)[: args.topk]

    def flagged(tok: str) -> bool:
        return any(cre.search(tok) for cre in compiled)

    print(f"Top {args.topk} tokens by chi^2 for target='{args.target}':")
    for tok, sc in pairs:
        mark = " [EXCLUDED-PATTERN-MATCH]" if flagged(tok) else ""
        print(f"  {tok:30s}  chi2={sc:9.2f}{mark}")

    print("\nNote: Tokens flagged as EXCLUDED-PATTERN-MATCH indicate potential enrichment leakage.")

if __name__ == "__main__":
    main()
