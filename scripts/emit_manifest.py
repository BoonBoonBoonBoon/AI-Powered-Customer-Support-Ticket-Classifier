import argparse
import json
from pathlib import Path
from datetime import datetime

ROOT = Path(__file__).resolve().parents[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--version', required=True, help='Transformer version tag, e.g., t1.0.3')
    ap.add_argument('--base-model', default='distilbert-base-uncased')
    ap.add_argument('--max-len', type=int, default=256)
    ap.add_argument('--epochs', type=int, default=3)
    ap.add_argument('--batch-size', type=int, default=16)
    ap.add_argument('--exclude-pattern', action='append', default=[
        r"__department_[a-z0-9_]+",
        r"__dept_[a-z0-9_]+",
        r"__type_[a-z0-9_]+",
    ])
    args = ap.parse_args()

    model_dir = ROOT / 'models' / 'transformers' / args.version
    metrics_path = model_dir / 'metrics.json'
    if not metrics_path.exists():
        raise SystemExit(f"metrics.json not found at {metrics_path}")
    with metrics_path.open('r', encoding='utf-8') as f:
        metrics = json.load(f)

    manifest = {
        "model_id": f"transformer:{args.version}",
        "framework": "transformers",
        "format": "torch",
        "created_at": datetime.utcnow().replace(microsecond=0).isoformat() + 'Z',
        "training": {
            "base_model": args.base_model,
            "max_length": args.max_len,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "exclude_patterns": args.exclude_pattern,
        },
        "artifacts": {
            "weights_uri": f"file://models/transformers/{args.version}/pytorch_model.bin",
            "tokenizer_uri": f"file://models/transformers/{args.version}/tokenizer",
            "label_map_uri": f"file://models/transformers/{args.version}/label_mappings.json",
            "onnx_uri": ""
        },
        "metrics": {
            "validation": {
                "priority": {
                    "accuracy": metrics.get('priority', {}).get('accuracy') or metrics.get('priority', {}).get('report', {}).get('accuracy'),
                    "macro_f1": metrics.get('priority', {}).get('macro_f1') or metrics.get('priority', {}).get('report', {}).get('macro avg', {}).get('f1-score')
                },
                "department": {
                    "accuracy": metrics.get('department', {}).get('accuracy') or metrics.get('department', {}).get('report', {}).get('accuracy'),
                    "macro_f1": metrics.get('department', {}).get('macro_f1') or metrics.get('department', {}).get('report', {}).get('macro avg', {}).get('f1-score')
                }
            }
        }
    }

    out_path = model_dir / 'manifest.json'
    with out_path.open('w', encoding='utf-8') as f:
        json.dump(manifest, f, indent=2)
    print(f"Wrote manifest to {out_path}")


if __name__ == '__main__':
    main()
