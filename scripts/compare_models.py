import json
import os
from pathlib import Path
from typing import Dict

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent


def read_metrics(path: Path) -> Dict:
    with path.open('r', encoding='utf-8') as f:
        return json.load(f)


def main():
    # Start with the production sklearn baseline
    pairs = [
        ("sklearn:v1.0.10", ROOT / "models" / "v1.0.10" / "metrics.json"),
    ]

    # Dynamically include all transformer versions that have metrics.json
    t_dir = ROOT / "models" / "transformers"
    if t_dir.exists():
        for child in sorted(t_dir.iterdir()):
            if not child.is_dir():
                continue
            name = child.name
            if not name.startswith("t"):
                continue
            metrics_path = child / "metrics.json"
            if metrics_path.exists():
                pairs.append((f"transformer:{name}", metrics_path))

    rows = []
    for name, path in pairs:
        if not path.exists():
            rows.append((name, None))
            continue
        m = read_metrics(path)
        def get(d, *keys, default=None):
            cur = d
            for k in keys:
                if k not in cur:
                    return default
                cur = cur[k]
            return cur
        rows.append((name, {
            'priority_macro_f1': get(m, 'priority', 'macro_f1', default=get(m, 'priority', 'summary', 'macro_f1')),
            'priority_accuracy': get(m, 'priority', 'accuracy', default=get(m, 'priority', 'summary', 'accuracy')),
            'department_macro_f1': get(m, 'department', 'macro_f1', default=get(m, 'department', 'summary', 'macro_f1')),
            'department_accuracy': get(m, 'department', 'accuracy', default=get(m, 'department', 'summary', 'accuracy')),
        }))

    print("Model Comparison (Validation)")
    print("-" * 72)
    print(f"{'Model':28} | {'P MacroF1':>10} | {'P Acc':>7} | {'D MacroF1':>10} | {'D Acc':>7}")
    print("-" * 72)
    for name, metrics in rows:
        if metrics is None:
            print(f"{name:28} | {'(missing)':>10} | {'-':>7} | {'-':>10} | {'-':>7}")
            continue
        pm = metrics['priority_macro_f1']
        pa = metrics['priority_accuracy']
        dm = metrics['department_macro_f1']
        da = metrics['department_accuracy']
        def fmt(x):
            return f"{x:.4f}" if isinstance(x, (int, float)) else str(x)
        print(f"{name:28} | {fmt(pm):>10} | {fmt(pa):>7} | {fmt(dm):>10} | {fmt(da):>7}")


if __name__ == '__main__':
    main()
