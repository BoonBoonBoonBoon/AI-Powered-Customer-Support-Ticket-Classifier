import argparse
import re
import sys
from pathlib import Path
import pandas as pd

PRIORITY_NORMALIZATION = {
    'critical': 'Urgent', 'urgent': 'Urgent', 'p1': 'Urgent', 'sev1': 'Urgent',
    'high': 'High', 'p2': 'High', 'sev2': 'High',
    'medium': 'Medium', 'normal': 'Medium', 'p3': 'Medium', 'sev3': 'Medium',
    'low': 'Low', 'minor': 'Low', 'p4': 'Low', 'sev4': 'Low'
}

DEPARTMENT_NORMALIZATION = {
    'technical issue': 'Tech Support', 'technical': 'Tech Support', 'tech support': 'Tech Support',
    'network problem': 'Tech Support', 'product setup': 'Tech Support', 'installation support': 'Tech Support',
    'data loss': 'Tech Support', 'peripheral compatibility': 'Tech Support',
    'billing inquiry': 'Billing', 'billing issue': 'Billing', 'billing': 'Billing', 'refund request': 'Billing',
    'cancellation request': 'Billing', 'account access': 'Billing', 'payment issue': 'Billing',
    'product inquiry': 'Sales', 'pricing question': 'Sales', 'sales inquiry': 'Sales', 'product pricing': 'Sales',
    'product roadmap': 'Sales', 'plan upgrade billing': 'Sales'
}

def slug(text: str) -> str:
    text = text.lower().strip()
    # Remove braces and punctuation
    text = re.sub(r"[{}\[\]()/'\"\\]+", ' ', text)
    text = re.sub(r'[^a-z0-9]+', '_', text)
    return re.sub(r'_+', '_', text).strip('_')[:40] or 'na'

def normalize_priority(val: str) -> str:
    v = str(val).strip().lower()
    return PRIORITY_NORMALIZATION.get(v, val.strip())

def normalize_department(val: str) -> str:
    v = str(val).strip().lower()
    return DEPARTMENT_NORMALIZATION.get(v, 'Tech Support' if 'tech' in v else val.strip())

def build_tokens(row, include_product, include_type, include_channel, include_csat):
    tokens = []
    if include_type and 'Ticket Type' in row and pd.notna(row['Ticket Type']):
        tokens.append(f"__type_{slug(str(row['Ticket Type']))}__")
    if include_channel and 'Ticket Channel' in row and pd.notna(row['Ticket Channel']):
        tokens.append(f"__channel_{slug(str(row['Ticket Channel']))}__")
    if include_product and 'Product Purchased' in row and pd.notna(row['Product Purchased']):
        tokens.append(f"__product_{slug(str(row['Product Purchased']))}__")
    if include_csat and 'Customer Satisfaction Rating' in row and pd.notna(row['Customer Satisfaction Rating']):
        try:
            rating = float(row['Customer Satisfaction Rating'])
            bucket = 'low' if rating < 2 else ('mid' if rating < 3.5 else 'high')
            tokens.append(f"__csat_{bucket}__")
        except Exception:
            pass
    return tokens

def clean_text(text: str) -> str:
    if not isinstance(text, str):
        return ''
    text = text.replace('{product_purchased}', ' ')
    text = re.sub(r'[{}]', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def main():
    ap = argparse.ArgumentParser(description="Enrich raw support tickets with feature tokens")
    ap.add_argument('--input-raw', required=True)
    ap.add_argument('--output', default='data/enriched_customer_tickets.csv')
    ap.add_argument('--include-product', action='store_true')
    ap.add_argument('--include-ticket-type', action='store_true')
    ap.add_argument('--include-channel', action='store_true')
    ap.add_argument('--include-csat', action='store_true')
    args = ap.parse_args()

    path = Path(args.input_raw)
    if not path.exists():
        print(f"Input not found: {path}")
        sys.exit(1)

    df = pd.read_csv(path)
    required = {'Ticket Subject', 'Ticket Description', 'Ticket Priority', 'Ticket Type'}
    miss = required - set(df.columns)
    if miss:
        print(f"Missing required columns: {miss}")
        sys.exit(1)

    out = pd.DataFrame()
    out['title'] = df['Ticket Subject'].astype(str).fillna('').str.strip()
    out['description'] = df['Ticket Description'].astype(str).fillna('')
    out['priority'] = df['Ticket Priority'].apply(normalize_priority)
    out['department'] = df['Ticket Type'].apply(normalize_department)

    enriched = []
    for idx, row in df.iterrows():
        base = clean_text(out.loc[idx, 'description'])
        toks = build_tokens(row, args.include_product, args.include_ticket_type, args.include_channel, args.include_csat)
        if toks:
            base = base + ' ' + ' '.join(toks)
        enriched.append(base)
    out['description'] = enriched

    out.to_csv(args.output, index=False)
    print(f"Enriched dataset written: {args.output} ({len(out)} rows)")
    print("Priority distribution:\n" + out['priority'].value_counts().to_string())
    print("Department distribution:\n" + out['department'].value_counts().to_string())

if __name__ == '__main__':
    main()
