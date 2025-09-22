import argparse
import json
import sys
from pathlib import Path
import pandas as pd
import hashlib
from typing import Dict, Tuple


DEFAULT_COLUMN_MAP = {
    # Source Header            # Target required name
    'Ticket Subject': 'title',
    'Ticket Description': 'description',
    'Ticket Priority': 'priority',
    'Ticket Type': 'department',
}

PRIORITY_NORMALIZATION = {
    'p1': 'Urgent', 'sev1': 'Urgent', 'critical': 'Urgent', 'urgent': 'Urgent',
    'p2': 'High', 'sev2': 'High', 'high': 'High',
    'p3': 'Medium', 'sev3': 'Medium', 'medium': 'Medium', 'normal': 'Medium',
    'p4': 'Low', 'sev4': 'Low', 'low': 'Low', 'minor': 'Low'
}

DEPARTMENT_NORMALIZATION = {
    'technical support': 'Tech Support', 'tech support': 'Tech Support', 'support': 'Tech Support',
    'billing issue': 'Billing', 'billing': 'Billing', 'billing inquiry': 'Billing',
    'sales inquiry': 'Sales', 'sales': 'Sales', 'pricing question': 'Sales', 'pricing': 'Sales'
}

REQUIRED_TARGET_COLUMNS = ['title', 'description', 'priority', 'department']


def detect_separator(path: Path) -> str:
    with path.open('r', encoding='utf-8', errors='ignore') as f:
        first = f.readline()
    return '\t' if first.count('\t') > first.count(',') else ','


def stable_hash(text: str) -> str:
    return hashlib.sha256(text.encode('utf-8', errors='ignore')).hexdigest()[:12]


def load_dataset(path: Path) -> pd.DataFrame:
    sep = detect_separator(path)
    try:
        df = pd.read_csv(path, sep=sep)
    except Exception as e:
        print(f"Failed to read {path}: {e}")
        sys.exit(1)
    return df


def apply_column_mapping(df: pd.DataFrame, mapping: Dict[str, str]) -> pd.DataFrame:
    rename_dict = {src: tgt for src, tgt in mapping.items() if src in df.columns}
    df = df.rename(columns=rename_dict)
    return df


def normalize_values(df: pd.DataFrame) -> pd.DataFrame:
    if 'priority' in df.columns:
        df['priority'] = (
            df['priority']
            .astype(str)
            .str.strip()
            .str.lower()
            .map(PRIORITY_NORMALIZATION)
            .fillna(df['priority'].astype(str).str.strip())
        )
    if 'department' in df.columns:
        df['department'] = (
            df['department']
            .astype(str)
            .str.strip()
            .str.lower()
            .map(DEPARTMENT_NORMALIZATION)
            .fillna(df['department'].astype(str).str.strip())
        )
    return df


def compute_text_metrics(series: pd.Series) -> Tuple[float, float]:
    lengths_chars = series.fillna('').astype(str).str.len()
    lengths_tokens = series.fillna('').astype(str).str.split().apply(len)
    return lengths_chars.mean(), lengths_tokens.mean()


def profile(df: pd.DataFrame):
    print("=== Dataset Profile ===")
    print(f"Rows: {len(df)} | Columns: {len(df.columns)}")
    print("Columns:", ', '.join(df.columns))
    print()
    print("Missing Values (count):")
    print(df.isna().sum().sort_values(ascending=False).to_string())
    print()
    for col in ['priority', 'department']:
        if col in df.columns:
            print(f"Distribution for {col}:")
            print(df[col].value_counts(dropna=False).head(30).to_string())
            print()
    if 'title' in df.columns:
        c, t = compute_text_metrics(df['title'])
        print(f"Average title length: {c:.1f} chars | {t:.1f} tokens")
    if 'description' in df.columns:
        c, t = compute_text_metrics(df['description'])
        print(f"Average description length: {c:.1f} chars | {t:.1f} tokens")
    # Duplicate detection (title+description)
    if {'title', 'description'} <= set(df.columns):
        key_hash = (df['title'].fillna('') + '||' + df['description'].fillna('')).apply(stable_hash)
        dup_rate = 1 - key_hash.nunique() / len(df) if len(df) else 0
        print(f"Approx duplicate pair rate (title+description): {dup_rate*100:.2f}%")
    print("=======================\n")


def write_normalized(df: pd.DataFrame, output: Path):
    missing_cols = [c for c in REQUIRED_TARGET_COLUMNS if c not in df.columns]
    if missing_cols:
        print(f"Cannot write normalized dataset; missing columns: {missing_cols}")
        return False
    normalized = df[REQUIRED_TARGET_COLUMNS].copy()
    before = len(normalized)
    normalized = normalized.dropna(subset=['title', 'description'])
    after = len(normalized)
    if after < before:
        print(f"Dropped {before-after} rows with missing title/description")
    output.parent.mkdir(parents=True, exist_ok=True)
    normalized.to_csv(output, index=False)
    print(f"Normalized dataset written: {output} ({len(normalized)} rows)")
    return True


def parse_mapping(arg: str) -> Dict[str, str]:
    # Format: Source1:target1,Source Two:target2
    mapping = {}
    if not arg:
        return mapping
    parts = [p for p in arg.split(',') if p.strip()]
    for p in parts:
        if ':' not in p:
            print(f"Ignoring malformed mapping chunk: {p}")
            continue
        src, tgt = p.split(':', 1)
        mapping[src.strip()] = tgt.strip()
    return mapping


def main():
    ap = argparse.ArgumentParser(description="Profile and optionally normalize the support tickets dataset")
    ap.add_argument('--input', required=True, help='Path to raw dataset CSV')
    ap.add_argument('--output-normalized', default='data/normalized_tickets.csv', help='Where to write normalized 4-column dataset')
    ap.add_argument('--column-map', default='', help='Additional column mapping Source:target,Src2:target2 (applied after defaults)')
    ap.add_argument('--no-normalize', action='store_true', help='Only profile; do not write normalized CSV')
    ap.add_argument('--show-head', type=int, default=0, help='Show first N rows of (partially) mapped frame')
    ap.add_argument('--export-json-profile', help='Optional path to write JSON profile summary')
    args = ap.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Input file not found: {input_path}")
        sys.exit(1)

    df = load_dataset(input_path)

    # Apply mappings
    column_map = DEFAULT_COLUMN_MAP.copy()
    column_map.update(parse_mapping(args.column_map))
    df = apply_column_mapping(df, column_map)

    # Normalize categorical values
    df = normalize_values(df)

    profile(df)

    if args.show_head > 0:
        print(df.head(args.show_head).to_string())

    if args.export_json_profile:
        prof = {
            'n_rows': int(len(df)),
            'columns': list(df.columns),
            'priority_distribution': df['priority'].value_counts().to_dict() if 'priority' in df.columns else None,
            'department_distribution': df['department'].value_counts().to_dict() if 'department' in df.columns else None,
        }
        outp = Path(args.export_json_profile)
        outp.parent.mkdir(parents=True, exist_ok=True)
        outp.write_text(json.dumps(prof, indent=2), encoding='utf-8')
        print(f"JSON profile written to {outp}")

    if not args.no_normalize:
        write_normalized(df, Path(args.output_normalized))


if __name__ == '__main__':
    main()
