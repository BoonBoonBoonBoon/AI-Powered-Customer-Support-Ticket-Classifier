#!/usr/bin/env python3
import argparse, glob, json, sys, os

def latest_metrics_file() -> str:
    candidates = sorted(glob.glob('models/v*/metrics.json'))
    if not candidates:
        print('No metrics.json files found', file=sys.stderr)
        sys.exit(1)
    return candidates[-1]

def main():
    ap = argparse.ArgumentParser(description='Macro-F1 gating utility')
    ap.add_argument('--priority-threshold', type=float, required=True)
    ap.add_argument('--department-threshold', type=float, required=True)
    args = ap.parse_args()
    mf = latest_metrics_file()
    with open(mf, 'r', encoding='utf-8') as f:
        data = json.load(f)
    pr = data['priority']['summary']['macro_f1']
    dep = data['department']['summary']['macro_f1']
    print(f"Gate check: priority={pr:.4f} (min {args.priority_threshold}) department={dep:.4f} (min {args.department_threshold})")
    if pr < args.priority_threshold or dep < args.department_threshold:
        print('Macro-F1 thresholds not met', file=sys.stderr)
        sys.exit(1)
    print('Macro-F1 gate passed')

if __name__ == '__main__':
    main()
