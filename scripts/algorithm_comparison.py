#!/usr/bin/env python3
"""Algorithm comparison script.

Compares several model families and hyperparameters for both targets (priority, department)
using the same preprocessing scheme as `TicketClassifier` (including optional --priority-extra
and length bucket augmentation). Reuses persisted validation indices from a reference model
version for reproducibility.

Outputs:
  - JSON summary of all evaluated configurations with macro F1 per target
  - Markdown report sorted by priority macro F1 (desc) then department macro F1

Example:
  python scripts/algorithm_comparison.py \
    --data data/enriched_customer_tickets.csv \
    --reference-version 1.0.4 \
    --models-dir models \
    --output-dir reports \
    --department-exclude-regex __type_[a-z0-9_]+ \
    --augment-length \
    --priority-extra

"""
from __future__ import annotations
import argparse
import json
import os
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report

# Utilities replicated (lightweight) from classifier

def _preprocess(text: Any) -> str:
    if not text:
        return ""
    if not isinstance(text, str):
        text = str(text)
    text = text.strip().lower()
    text = re.sub(r"\s+", " ", text)
    return text

def _length_bucket(n_tokens: int) -> str:
    if n_tokens < 8:
        return "__LEN_VSHORT__"
    if n_tokens < 20:
        return "__LEN_SHORT__"
    if n_tokens < 40:
        return "__LEN_MEDIUM__"
    if n_tokens < 70:
        return "__LEN_LONG__"
    return "__LEN_VLONG__"

# Priority extra tokens logic (mirrors TicketClassifier._priority_extra_tokens)
URGENT_KW = ["outage", "down", "critical", "severe", "emergency", "unresponsive"]
HIGH_KW = ["failure", "crash", "error", "corrupt", "broken"]
BILLING_KW = ["invoice", "charged", "billing", "payment", "refund"]
ACCESS_KW = ["login", "credential", "password", "access", "locked"]

def _priority_extra_tokens(title: str, desc: str) -> List[str]:
    text = f"{title} {desc}".lower()
    tokens: List[str] = []
    def any_kw(words):
        return any(w in text for w in words)
    if any_kw(URGENT_KW):
        tokens.append("__KWD_URGENT__")
    if any_kw(HIGH_KW):
        tokens.append("__KWD_HIGH__")
    if any_kw(BILLING_KW):
        tokens.append("__KWD_BILLING_CTX__")
    if any_kw(ACCESS_KW):
        tokens.append("__KWD_ACCESS__")
    total_chars = max(len(desc), 1)
    caps_chars = sum(1 for c in desc if c.isupper())
    if caps_chars / total_chars > 0.2:
        tokens.append("__STRUCT_CAPS_HEAVY__")
    punct_ratio = sum(1 for c in desc if c in "!?") / total_chars
    if punct_ratio > 0.02:
        tokens.append("__STRUCT_PUNCT_ATTENTION__")
    if desc.count("!") >= 2:
        tokens.append("__STRUCT_MULTI_EXCL__")
    if any(ch.isdigit() for ch in desc):
        tokens.append("__STRUCT_DIGITS__")
    return tokens

@dataclass
class TargetTexts:
    priority: List[str]
    department: List[str]


def build_texts(df: pd.DataFrame, augment_length: bool, department_exclude: List[str], priority_extra: bool) -> TargetTexts:
    compiled_exclusions = [re.compile(p) for p in department_exclude]
    pr_texts: List[str] = []
    dep_texts: List[str] = []
    for _, row in df.iterrows():
        title = _preprocess(row["title"])
        desc_orig = _preprocess(row["description"])
        combined_priority = f"title: {title}\nbody: {desc_orig}".strip()
        if priority_extra:
            extra = _priority_extra_tokens(title, desc_orig)
            if extra:
                combined_priority = f"{combined_priority} {' '.join(extra)}"
        if augment_length:
            combined_priority = f"{combined_priority} {_length_bucket(len(combined_priority.split()))}"
        # Department path with exclusions
        dep_desc = desc_orig
        for cre in compiled_exclusions:
            dep_desc = cre.sub(" ", dep_desc)
        dep_desc = re.sub(r"\s+", " ", dep_desc).strip()
        combined_dep = f"title: {title}\nbody: {dep_desc}".strip()
        if augment_length:
            combined_dep = f"{combined_dep} {_length_bucket(len(combined_dep.split()))}"
        pr_texts.append(combined_priority)
        dep_texts.append(combined_dep)
    return TargetTexts(pr_texts, dep_texts)

@dataclass
class EvalResult:
    algo: str
    params: Dict[str, Any]
    priority_macro_f1: float
    department_macro_f1: float
    priority_accuracy: float
    department_accuracy: float
    training_seconds: float
    priority_extra: bool


def evaluate_configuration(train_df: pd.DataFrame, val_df: pd.DataFrame, algo: str, params: Dict[str, Any], augment_length: bool, department_exclude: List[str], priority_extra: bool) -> EvalResult:
    t0 = time.perf_counter()
    texts_train = build_texts(train_df, augment_length, department_exclude, priority_extra)
    texts_val = build_texts(val_df, augment_length, department_exclude, priority_extra)

    # Priority label encoding
    pr_enc = LabelEncoder()
    y_pr_train = pr_enc.fit_transform(train_df["priority"].tolist())
    y_pr_val = pr_enc.transform(val_df["priority"].tolist())

    dep_enc = LabelEncoder()
    y_dep_train = dep_enc.fit_transform(train_df["department"].tolist())
    y_dep_val = dep_enc.transform(val_df["department"].tolist())

    # Vectorizers separate per target
    pr_vec = TfidfVectorizer(ngram_range=(1,2), min_df=2)
    dep_vec = TfidfVectorizer(ngram_range=(1,2), min_df=2)
    X_pr_train = pr_vec.fit_transform(texts_train.priority)
    X_dep_train = dep_vec.fit_transform(texts_train.department)
    X_pr_val = pr_vec.transform(texts_val.priority)
    X_dep_val = dep_vec.transform(texts_val.department)

    # Instantiate models
    if algo == "logreg":
        model_pr = LogisticRegression(max_iter=200, class_weight=params.get("class_weight"), C=params.get("C",1.0))
        model_dep = LogisticRegression(max_iter=200, class_weight=params.get("class_weight"), C=params.get("C",1.0))
    elif algo == "linearsvc":
        model_pr = LinearSVC(C=params.get("C",1.0))
        model_dep = LinearSVC(C=params.get("C",1.0))
    elif algo == "mnb":
        model_pr = MultinomialNB(alpha=params.get("alpha",1.0))
        model_dep = MultinomialNB(alpha=params.get("alpha",1.0))
    else:
        raise ValueError(f"Unknown algo: {algo}")

    model_pr.fit(X_pr_train, y_pr_train)
    model_dep.fit(X_dep_train, y_dep_train)

    # Predictions
    y_pr_pred = model_pr.predict(X_pr_val)
    y_dep_pred = model_dep.predict(X_dep_val)

    pr_report = classification_report(y_pr_val, y_pr_pred, output_dict=True, zero_division=0)
    dep_report = classification_report(y_dep_val, y_dep_pred, output_dict=True, zero_division=0)

    priority_macro = pr_report.get("macro avg", {}).get("f1-score", 0.0)
    department_macro = dep_report.get("macro avg", {}).get("f1-score", 0.0)
    priority_acc = pr_report.get("accuracy", 0.0)
    department_acc = dep_report.get("accuracy", 0.0)
    t1 = time.perf_counter()

    return EvalResult(
        algo=algo,
        params=params,
        priority_macro_f1=priority_macro,
        department_macro_f1=department_macro,
        priority_accuracy=priority_acc,
        department_accuracy=department_acc,
        training_seconds=t1 - t0,
        priority_extra=priority_extra,
    )


def load_split(df: pd.DataFrame, models_dir: Path, reference_version: str, val_frac: float, seed: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
    version_dir = models_dir / f"v{reference_version}"
    val_idx_file = version_dir / "val_indices.txt"
    train_idx_file = version_dir / "train_indices.txt"
    if val_idx_file.exists() and train_idx_file.exists():
        try:
            val_indices = [int(x.strip()) for x in val_idx_file.read_text().splitlines() if x.strip()]
            train_indices = [int(x.strip()) for x in train_idx_file.read_text().splitlines() if x.strip()]
            return df.loc[train_indices], df.loc[val_indices]
        except Exception as e:
            print(f"WARNING: Failed to load persisted indices: {e}. Regenerating split.")
    # fallback deterministic split
    from sklearn.model_selection import train_test_split
    train_df, val_df = train_test_split(
        df,
        test_size=val_frac,
        random_state=seed,
        shuffle=True,
        stratify=df['priority'] if 'priority' in df else None
    )
    return train_df, val_df


def write_markdown(path: Path, results: List[EvalResult]):
    lines = ["# Algorithm Comparison", "", "Sorted by priority macro F1 desc then department macro F1.", "", "| Algo | Params | Priority Macro F1 | Dept Macro F1 | Priority Acc | Dept Acc | Train (s) | Priority Extra |", "|------|--------|-------------------|---------------|-------------|----------|-----------|---------------|"]
    for r in results:
        lines.append(f"| {r.algo} | {json.dumps(r.params)} | {r.priority_macro_f1:.4f} | {r.department_macro_f1:.4f} | {r.priority_accuracy:.4f} | {r.department_accuracy:.4f} | {r.training_seconds:.2f} | {r.priority_extra} |")
    best_pr = max(results, key=lambda x: x.priority_macro_f1)
    best_dep = max(results, key=lambda x: x.department_macro_f1)
    lines.append("\n## Best Configurations")
    lines.append(f"- Priority: {best_pr.algo} {best_pr.params} priority_extra={best_pr.priority_extra} macro_f1={best_pr.priority_macro_f1:.4f}")
    lines.append(f"- Department: {best_dep.algo} {best_dep.params} priority_extra={best_dep.priority_extra} macro_f1={best_dep.department_macro_f1:.4f}")
    path.write_text("\n".join(lines), encoding="utf-8")


def main():
    ap = argparse.ArgumentParser(description="Compare model algorithms for both targets")
    ap.add_argument('--data', required=True, help='CSV with title,description,priority,department')
    ap.add_argument('--models-dir', default='models', help='Models directory (for persisted indices)')
    ap.add_argument('--reference-version', default='1.0.4', help='Model version whose split indices to reuse')
    ap.add_argument('--val-frac', type=float, default=0.2, help='Validation fraction if split must be regenerated')
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--output-dir', default='reports')
    ap.add_argument('--department-exclude-regex', action='append', default=[], help='Regex pattern(s) to exclude from department text')
    ap.add_argument('--augment-length', action='store_true', help='Enable length bucket tokens')
    ap.add_argument('--priority-extra', action='store_true', help='Also evaluate variants WITH engineered priority tokens (both off & on)')
    args = ap.parse_args()

    np.random.seed(args.seed)
    data_path = Path(args.data)
    if not data_path.exists():
        raise SystemExit(f"Data not found: {data_path}")
    df = pd.read_csv(data_path)

    train_df, val_df = load_split(df, Path(args.models_dir), args.reference_version, args.val_frac, args.seed)
    print(f"Train size: {len(train_df)} | Val size: {len(val_df)}")

    department_exclude = args.department_exclude_regex or []

    configs: List[Tuple[str, Dict[str, Any]]] = []
    # Logistic Regression grid
    for C in [0.5, 1.0, 2.0, 5.0]:
        configs.append(("logreg", {"C": C, "class_weight": "balanced"}))
    # Linear SVC grid
    for C in [0.5, 1.0, 2.0]:
        configs.append(("linearsvc", {"C": C}))
    # Multinomial NB grid
    for alpha in [0.5, 1.0, 2.0]:
        configs.append(("mnb", {"alpha": alpha}))

    results: List[EvalResult] = []
    for algo, params in configs:
        for extra_flag in ([False, True] if args.priority_extra else [False]):
            r = evaluate_configuration(train_df, val_df, algo, params, args.augment_length, department_exclude, extra_flag)
            results.append(r)
            print(f"Evaluated {algo} params={params} priority_extra={extra_flag} -> pr_macro={r.priority_macro_f1:.4f} dep_macro={r.department_macro_f1:.4f}")

    # Sort
    results_sorted = sorted(results, key=lambda x: (-x.priority_macro_f1, -x.department_macro_f1))

    os.makedirs(args.output_dir, exist_ok=True)
    json_path = Path(args.output_dir) / f"algo_comparison_v{args.reference_version}.json"
    md_path = Path(args.output_dir) / f"algo_comparison_v{args.reference_version}.md"
    json_path.write_text(json.dumps([r.__dict__ for r in results_sorted], indent=2), encoding='utf-8')
    write_markdown(md_path, results_sorted)
    print(f"Wrote {json_path}")
    print(f"Wrote {md_path}")

if __name__ == '__main__':
    main()
