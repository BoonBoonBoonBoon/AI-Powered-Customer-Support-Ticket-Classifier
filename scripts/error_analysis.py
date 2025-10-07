import argparse
import json
import os
from pathlib import Path
from typing import List, Dict, Any
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix

from app.models.classifier import TicketClassifier
from app.config import settings


def load_model(version_dir: Path) -> TicketClassifier:
    clf = TicketClassifier()
    clf.load_models(str(version_dir))
    return clf

def evaluate_split(df: pd.DataFrame, clf: TicketClassifier) -> Dict[str, Any]:
    y_pr_true: List[str] = []
    y_dep_true: List[str] = []
    y_pr_pred: List[str] = []
    y_dep_pred: List[str] = []
    pr_conf: List[float] = []
    dep_conf: List[float] = []

    for _, row in df.iterrows():
        p, d, pc, dc = clf.predict(row['title'], row['description'])
        y_pr_true.append(row['priority'])
        y_dep_true.append(row['department'])
        y_pr_pred.append(p)
        y_dep_pred.append(d)
        pr_conf.append(pc)
        dep_conf.append(dc)

    pr_report = classification_report(y_pr_true, y_pr_pred, output_dict=True)
    dep_report = classification_report(y_dep_true, y_dep_pred, output_dict=True)
    pr_cm = confusion_matrix(y_pr_true, y_pr_pred).tolist()
    dep_cm = confusion_matrix(y_dep_true, y_dep_pred).tolist()

    # High-confidence misclassifications
    rows = []
    for yt_p, yp_p, yt_d, yp_d, pc, dc, (_, r) in zip(y_pr_true, y_pr_pred, y_dep_true, y_dep_pred, pr_conf, dep_conf, df.iterrows()):
        mis_p = yt_p != yp_p
        mis_d = yt_d != yp_d
        if (mis_p and pc >= 0.6) or (mis_d and dc >= 0.6):
            rows.append({
                'title': r['title'][:120],
                'priority_true': yt_p,
                'priority_pred': yp_p,
                'priority_conf': round(pc,3),
                'department_true': yt_d,
                'department_pred': yp_d,
                'department_conf': round(dc,3),
                'mis_priority': mis_p,
                'mis_department': mis_d,
            })

    return {
        'priority': {
            'report': pr_report,
            'confusion_matrix': pr_cm,
        },
        'department': {
            'report': dep_report,
            'confusion_matrix': dep_cm,
        },
        'high_conf_misclassifications': rows,
    }

def write_markdown(out_path: Path, data: Dict[str, Any], split_label: str):
    def fmt_report(rep: Dict[str, Any]) -> str:
        lines = ["| Class | Precision | Recall | F1 | Support |", "|-------|-----------|--------|----|---------|"]
        for cls, vals in rep.items():
            if cls in ('accuracy', 'macro avg', 'weighted avg'):
                continue
            lines.append(f"| {cls} | {vals.get('precision',0):.3f} | {vals.get('recall',0):.3f} | {vals.get('f1-score',0):.3f} | {int(vals.get('support',0))} |")
        macro = rep.get('macro avg', {})
        lines.append(f"| Macro Avg | {macro.get('precision',0):.3f} | {macro.get('recall',0):.3f} | {macro.get('f1-score',0):.3f} | - |")
        return '\n'.join(lines)

    md = [f"# Error Analysis: {split_label}"]
    pr = data['priority']['report']
    dep = data['department']['report']
    md.append("\n## Priority")
    md.append(fmt_report(pr))
    md.append("\n## Department")
    md.append(fmt_report(dep))
    md.append("\n## High-Confidence Misclassifications (confidence >= 0.6)")
    rows = data['high_conf_misclassifications']
    if not rows:
        md.append("None above confidence threshold.")
    else:
        md.append("| Title | P_true | P_pred | P_conf | D_true | D_pred | D_conf | MisP | MisD |")
        md.append("|-------|--------|--------|--------|--------|--------|--------|------|------|")
        for r in rows[:50]:
            md.append(f"| {r['title']} | {r['priority_true']} | {r['priority_pred']} | {r['priority_conf']:.2f} | {r['department_true']} | {r['department_pred']} | {r['department_conf']:.2f} | {'Y' if r['mis_priority'] else ''} | {'Y' if r['mis_department'] else ''} |")
        if len(rows) > 50:
            md.append(f"\n_Truncated to 50 of {len(rows)} rows._")

    out_path.write_text('\n'.join(md), encoding='utf-8')


def main():
    ap = argparse.ArgumentParser(description="Run error analysis for a trained model version")
    ap.add_argument('--version', default=settings.MODEL_VERSION, help='Model version (e.g., 1.0.3)')
    ap.add_argument('--models-dir', default=settings.MODELS_BASE_DIR, help='Models base directory')
    ap.add_argument('--data', required=True, help='CSV used for training (must include title,description,priority,department)')
    ap.add_argument('--val-frac', type=float, default=settings.VALIDATION_SPLIT, help='Validation fraction (to reproduce split)')
    ap.add_argument('--seed', type=int, default=settings.RANDOM_SEED)
    ap.add_argument('--output-dir', default='reports', help='Directory to write analysis outputs')
    args = ap.parse_args()

    import numpy as np
    np.random.seed(args.seed)

    data_path = Path(args.data)
    if not data_path.exists():
        raise SystemExit(f"Data not found: {data_path}")
    df = pd.read_csv(data_path)

    # Recreate split (mirrors train.py stratification by priority)
    from sklearn.model_selection import train_test_split
    _, val_df = train_test_split(
        df,
        test_size=args.val_frac,
        random_state=args.seed,
        shuffle=True,
        stratify=df['priority'] if 'priority' in df else None
    )

    version_dir = Path(args.models_dir) / f"v{args.version}"
    if not version_dir.exists():
        raise SystemExit(f"Model version directory not found: {version_dir}")

    clf = load_model(version_dir)

    analysis = evaluate_split(val_df, clf)

    os.makedirs(args.output_dir, exist_ok=True)
    json_path = Path(args.output_dir) / f"error_analysis_v{args.version}.json"
    md_path = Path(args.output_dir) / f"error_analysis_v{args.version}.md"
    json_path.write_text(json.dumps(analysis, indent=2), encoding='utf-8')
    write_markdown(md_path, analysis, f"v{args.version} (validation split)")

    print(f"Wrote: {json_path}")
    print(f"Wrote: {md_path}")

if __name__ == '__main__':
    main()
