#!/usr/bin/env python
"""Bulk evaluation against a mock (or real) labeled CSV dataset.

Features:
- Reads a CSV (default: data/mock_eval.csv)
- Calls remote or local API /classify for each row
- Prints a colorized table with per-sample prediction, confidence & match flags
- Computes accuracy for priority & department, plus macro precision/recall/F1 if labels available
- Exports a markdown report (--md-report) summarizing metrics + table

Usage:
  python scripts/bulk_eval.py --base-url https://<domain> --csv data/mock_eval.csv --md-report reports/eval_report.md
"""
from __future__ import annotations
import argparse
import csv
import json
import math
import os
import statistics
import sys
from dataclasses import dataclass
from typing import List, Dict, Any, Optional
import time
import requests

RESET = "\x1b[0m"
COL = {
    "green": "\x1b[32m",
    "red": "\x1b[31m",
    "yellow": "\x1b[33m",
    "cyan": "\x1b[36m",
    "bold": "\x1b[1m",
}

def c(txt: str, color: str) -> str:
    return f"{COL.get(color,'')}{txt}{RESET}"

@dataclass
class SampleResult:
    title: str
    priority_true: str | None
    dept_true: str | None
    priority_pred: str | None
    dept_pred: str | None
    p_conf: float | None
    d_conf: float | None
    error: str | None = None

    def priority_match(self) -> Optional[bool]:
        if self.priority_true is None or self.priority_pred is None:
            return None
        return self.priority_true == self.priority_pred

    def dept_match(self) -> Optional[bool]:
        if self.dept_true is None or self.dept_pred is None:
            return None
        return self.dept_true == self.dept_pred


def load_rows(path: str) -> list[dict[str, str]]:
    with open(path, newline='', encoding='utf-8') as f:
        return list(csv.DictReader(f))


def classify(base: str, title: str, description: str, timeout: float = 15.0) -> dict[str, Any]:
    url = base.rstrip('/') + '/classify'
    r = requests.post(url, json={"title": title, "description": description}, timeout=timeout)
    if r.status_code != 200:
        raise RuntimeError(f"status {r.status_code}: {r.text[:300]}")
    return r.json()


def macro_scores(truths: list[str], preds: list[str]) -> dict[str, float]:
    labels = sorted(set(truths))
    precs = []
    recs = []
    f1s = []
    for label in labels:
        tp = sum(1 for t,p in zip(truths,preds) if t==label and p==label)
        fp = sum(1 for t,p in zip(truths,preds) if t!=label and p==label)
        fn = sum(1 for t,p in zip(truths,preds) if t==label and p!=label)
        prec = tp / (tp+fp) if (tp+fp)>0 else 0.0
        rec = tp / (tp+fn) if (tp+fn)>0 else 0.0
        f1 = 2*prec*rec/(prec+rec) if (prec+rec)>0 else 0.0
        precs.append(prec); recs.append(rec); f1s.append(f1)
    return {
        "macro_precision": sum(precs)/len(precs) if precs else 0.0,
        "macro_recall": sum(recs)/len(recs) if recs else 0.0,
        "macro_f1": sum(f1s)/len(f1s) if f1s else 0.0,
    }


def render_table(results: list[SampleResult]) -> str:
    # Determine column widths
    headers = ["Title","P_true","P_pred","P_conf","MatchP","D_true","D_pred","D_conf","MatchD"]
    rows = []
    for r in results:
        rows.append([
            r.title[:30],
            r.priority_true or "",
            r.priority_pred or (r.error and "ERROR"),
            f"{r.p_conf:.2f}" if r.p_conf is not None else "",
            {True:"Y", False:"N", None:""}[r.priority_match()],
            r.dept_true or "",
            r.dept_pred or (r.error and "ERROR"),
            f"{r.d_conf:.2f}" if r.d_conf is not None else "",
            {True:"Y", False:"N", None:""}[r.dept_match()],
        ])
    widths = [max(len(str(h)), max((len(str(row[i])) for row in rows), default=0)) for i,h in enumerate(headers)]
    def fmt_row(row):
        return " | ".join(str(val).ljust(widths[i]) for i,val in enumerate(row))
    out = [fmt_row(headers), "-+-".join('-'*w for w in widths)]
    for row,r in zip(rows, results):
        # Color matches
        pm = r.priority_match(); dm = r.dept_match()
        if pm is True: row[4] = c(row[4], 'green')
        elif pm is False: row[4] = c(row[4], 'red')
        if dm is True: row[8] = c(row[8], 'green')
        elif dm is False: row[8] = c(row[8], 'red')
        out.append(fmt_row(row))
    return "\n".join(out)


def write_markdown(path: str, results: list[SampleResult], p_acc: float|None, d_acc: float|None, p_macro: dict[str,float], d_macro: dict[str,float]):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        f.write("# Bulk Evaluation Report\n\n")
        f.write(f"Total samples: {len(results)}\n\n")
        if p_acc is not None:
            f.write(f"**Priority Accuracy:** {p_acc:.2%}\n\n")
            f.write(f"**Priority Macro F1:** {p_macro.get('macro_f1',0):.3f}\n\n")
        if d_acc is not None:
            f.write(f"**Department Accuracy:** {d_acc:.2%}\n\n")
            f.write(f"**Department Macro F1:** {d_macro.get('macro_f1',0):.3f}\n\n")
        f.write("## Samples\n\n")
        f.write("| Title | P_true | P_pred | P_conf | MatchP | D_true | D_pred | D_conf | MatchD |\n")
        f.write("|-------|--------|--------|--------|--------|--------|--------|--------|--------|\n")
        for r in results:
            f.write(f"| {r.title[:30]} | {r.priority_true or ''} | {r.priority_pred or ''} | {r.p_conf and f'{r.p_conf:.2f}' or ''} | {r.priority_match()} | {r.dept_true or ''} | {r.dept_pred or ''} | {r.d_conf and f'{r.d_conf:.2f}' or ''} | {r.dept_match()} |\n")
        f.write("\nGenerated at runtime.\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--base-url', default='http://localhost:8000')
    ap.add_argument('--csv', default='data/mock_eval.csv')
    ap.add_argument('--md-report', default=None, help='Optional markdown report output path')
    ap.add_argument('--fail-threshold-priority-acc', type=float, default=None, help='Fail if priority accuracy below this (0-1)')
    ap.add_argument('--fail-threshold-dept-acc', type=float, default=None, help='Fail if department accuracy below this (0-1)')
    ap.add_argument('--rate-limit-sleep', type=float, default=0.0, help='Seconds to sleep between requests')
    args = ap.parse_args()

    rows = load_rows(args.csv)
    results: list[SampleResult] = []

    for row in rows:
        title = row.get('title','')
        desc = row.get('description','')
        p_true = row.get('priority') or None
        d_true = row.get('department') or None
        try:
            payload = classify(args.base_url, title, desc)
            results.append(SampleResult(
                title=title,
                priority_true=p_true,
                dept_true=d_true,
                priority_pred=payload.get('predicted_priority'),
                dept_pred=payload.get('predicted_department'),
                p_conf=payload.get('priority_confidence'),
                d_conf=payload.get('department_confidence'),
            ))
        except Exception as e:  # noqa: BLE001
            results.append(SampleResult(title=title, priority_true=p_true, dept_true=d_true,
                                        priority_pred=None, dept_pred=None, p_conf=None, d_conf=None, error=str(e)))
        if args.rate_limit_sleep:
            time.sleep(args.rate_limit_sleep)

    # Compute metrics
    pri_truth = [r.priority_true for r in results if r.priority_true and r.priority_pred]
    pri_pred  = [r.priority_pred for r in results if r.priority_true and r.priority_pred]
    dept_truth = [r.dept_true for r in results if r.dept_true and r.dept_pred]
    dept_pred  = [r.dept_pred for r in results if r.dept_true and r.dept_pred]

    p_acc = (sum(1 for t,p in zip(pri_truth, pri_pred) if t==p) / len(pri_truth)) if pri_truth else None
    d_acc = (sum(1 for t,p in zip(dept_truth, dept_pred) if t==p) / len(dept_truth)) if dept_truth else None
    p_macro = macro_scores(pri_truth, pri_pred) if pri_truth else {}
    d_macro = macro_scores(dept_truth, dept_pred) if dept_truth else {}

    # Print metrics summary
    print(c("Bulk Evaluation Summary", 'cyan'))
    if p_acc is not None:
        print(f"Priority Accuracy: {p_acc:.2%}  MacroF1: {p_macro.get('macro_f1',0):.3f}")
    if d_acc is not None:
        print(f"Department Accuracy: {d_acc:.2%} MacroF1: {d_macro.get('macro_f1',0):.3f}")

    print()
    print(render_table(results))

    # Write markdown report if requested
    if args.md_report:
        write_markdown(args.md_report, results, p_acc, d_acc, p_macro, d_macro)
        print(f"\nMarkdown report written to {args.md_report}")

    # Threshold enforcement
    exit_code = 0
    if args.fail_threshold_priority_acc is not None and p_acc is not None and p_acc < args.fail_threshold_priority_acc:
        print(c(f"Priority accuracy {p_acc:.3f} below threshold {args.fail_threshold_priority_acc}", 'red'))
        exit_code = 2
    if args.fail_threshold_dept_acc is not None and d_acc is not None and d_acc < args.fail_threshold_dept_acc:
        print(c(f"Department accuracy {d_acc:.3f} below threshold {args.fail_threshold_dept_acc}", 'red'))
        exit_code = max(exit_code, 3)

    sys.exit(exit_code)


if __name__ == '__main__':  # pragma: no cover
    main()
