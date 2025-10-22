#!/usr/bin/env python3
"""Dual-head DistilBERT fine-tuning for priority + department.

Outputs:
  models/transformers/<version>/
    config.json
    tokenizer/
    pytorch_model.bin
    label_mappings.json
    metrics.json

Usage (basic):
  python scripts/train_transformer.py \
    --data data/enriched_customer_tickets.csv \
    --model-name distilbert-base-uncased \
    --epochs 3 --batch-size 16 --lr 5e-5 \
    --output-version t1.0.0

Note: This is an initial prototype for experimentation; not yet wired into FastAPI app.
"""
from __future__ import annotations
import argparse, json, os, math, random, time, re
from dataclasses import dataclass
from typing import Dict, Any
import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModel, AutoTokenizer, get_linear_schedule_with_warmup
from tqdm import tqdm

SEED = 42
random.seed(SEED); torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)

PRIORITY_LABELS = ["Urgent","High","Medium","Low"]

def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    req = {"title","description","priority","department"}
    missing = req - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {missing}")
    return df

def _apply_exclusions(text: str, compiled_patterns: list[re.Pattern[str]] | None) -> str:
    if not compiled_patterns:
        return text
    for cre in compiled_patterns:
        text = cre.sub(" ", text)
    return re.sub(r"\s+", " ", text).strip()


def preprocess_text(title: str, description: str, max_len: int = 512, compiled_patterns: list[re.Pattern[str]] | None = None):
    # Remove leakage-like enrichment tokens if configured; truncation handled by tokenizer.
    title = title.strip()
    description = _apply_exclusions(description.strip(), compiled_patterns)
    return f"[TITLE] {title} [DESC] {description}"[:2000]

class TicketDataset(Dataset):
    def __init__(self, df: pd.DataFrame, tokenizer, max_len: int, pri2id: Dict[str,int], dep2id: Dict[str,int], compiled_patterns: list[re.Pattern[str]] | None = None):
        self.df = df.reset_index(drop=True)
        self.tok = tokenizer
        self.max_len = max_len
        self.pri2id = pri2id
        self.dep2id = dep2id
        self._compiled_patterns = compiled_patterns or []
    def __len__(self): return len(self.df)
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        text = preprocess_text(str(row.title), str(row.description), self.max_len, self._compiled_patterns)
        enc = self.tok(text, truncation=True, max_length=self.max_len, padding='max_length', return_tensors='pt')
        item = {k: v.squeeze(0) for k,v in enc.items()}
        item['priority_label'] = torch.tensor(self.pri2id[row.priority])
        item['department_label'] = torch.tensor(self.dep2id[row.department])
        return item

class DualHeadModel(nn.Module):
    def __init__(self, base_model_name: str, priority_classes: int, department_classes: int, dropout: float = 0.2):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(base_model_name)
        hidden = self.encoder.config.hidden_size
        self.dropout = nn.Dropout(dropout)
        self.priority_head = nn.Linear(hidden, priority_classes)
        self.department_head = nn.Linear(hidden, department_classes)
    def forward(self, input_ids, attention_mask):
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        pooled = out.last_hidden_state[:,0]
        pooled = self.dropout(pooled)
        return self.priority_head(pooled), self.department_head(pooled)

@dataclass
class TrainConfig:
    model_name: str
    epochs: int
    batch_size: int
    lr: float
    weight_decay: float
    warmup_ratio: float
    max_len: int
    grad_accum: int
    device: str

def compute_metrics(preds_p, labels_p, preds_d, labels_d):
    from sklearn.metrics import classification_report
    rep_p = classification_report(labels_p, preds_p, output_dict=True, zero_division=0)
    rep_d = classification_report(labels_d, preds_d, output_dict=True, zero_division=0)
    return {
        'priority': {
            'macro_f1': rep_p['macro avg']['f1-score'],
            'accuracy': rep_p['accuracy'],
            'report': rep_p
        },
        'department': {
            'macro_f1': rep_d['macro avg']['f1-score'],
            'accuracy': rep_d['accuracy'],
            'report': rep_d
        }
    }

def train_loop(cfg: TrainConfig, model, tokenizer, train_ds, val_ds, pri2id, dep2id, output_dir: str):
    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size)
    model.to(cfg.device)
    total_steps = cfg.epochs * math.ceil(len(train_loader)/cfg.grad_accum)
    warmup_steps = int(total_steps * cfg.warmup_ratio)
    no_decay = ["bias","LayerNorm.weight"]
    grouped = [
        {"params":[p for n,p in model.named_parameters() if not any(nd in n for nd in no_decay)],"weight_decay":cfg.weight_decay},
        {"params":[p for n,p in model.named_parameters() if any(nd in n for nd in no_decay)],"weight_decay":0.0},
    ]
    optimizer = torch.optim.AdamW(grouped, lr=cfg.lr)
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)
    loss_fn_p = nn.CrossEntropyLoss()
    loss_fn_d = nn.CrossEntropyLoss()
    best_macro = -1.0
    os.makedirs(output_dir, exist_ok=True)

    for epoch in range(1, cfg.epochs+1):
        model.train(); total_loss=0.0
        optimizer.zero_grad()
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}")
        for step,batch in enumerate(pbar, start=1):
            batch = {k:v.to(cfg.device) for k,v in batch.items()}
            logits_p, logits_d = model(batch['input_ids'], batch['attention_mask'])
            loss_p = loss_fn_p(logits_p, batch['priority_label'])
            loss_d = loss_fn_d(logits_d, batch['department_label'])
            loss = loss_p + loss_d
            loss.backward()
            if step % cfg.grad_accum == 0:
                optimizer.step(); scheduler.step(); optimizer.zero_grad()
            total_loss += loss.item()
            pbar.set_postfix(loss=f"{total_loss/step:.4f}")

        # Validation
        model.eval()
        all_p_pred=[]; all_p_true=[]; all_d_pred=[]; all_d_true=[]
        with torch.no_grad():
            for batch in val_loader:
                batch = {k:v.to(cfg.device) for k,v in batch.items()}
                lp, ld = model(batch['input_ids'], batch['attention_mask'])
                all_p_pred.extend(torch.argmax(lp, dim=1).cpu().tolist())
                all_p_true.extend(batch['priority_label'].cpu().tolist())
                all_d_pred.extend(torch.argmax(ld, dim=1).cpu().tolist())
                all_d_true.extend(batch['department_label'].cpu().tolist())
        metrics = compute_metrics(all_p_pred, all_p_true, all_d_pred, all_d_true)
        macro = metrics['priority']['macro_f1']
        # Save best
        if macro > best_macro:
            best_macro = macro
            torch.save(model.state_dict(), os.path.join(output_dir, 'pytorch_model.bin'))
            with open(os.path.join(output_dir,'metrics.json'),'w',encoding='utf-8') as f:
                json.dump(metrics, f, indent=2)
            print(f"Saved new best (priority macro_f1={macro:.4f})")

    # Persist label mappings & config
    mappings = {
        'priority_id2label': {v:k for k,v in pri2id.items()},
        'priority_label2id': pri2id,
        'department_id2label': {v:k for k,v in dep2id.items()},
        'department_label2id': dep2id,
    }
    with open(os.path.join(output_dir,'label_mappings.json'),'w',encoding='utf-8') as f:
        json.dump(mappings,f,indent=2)
    tokenizer.save_pretrained(os.path.join(output_dir,'tokenizer'))

def main():
    ap = argparse.ArgumentParser(description="Dual-head transformer trainer")
    ap.add_argument('--data', required=True)
    ap.add_argument('--model-name', default='distilbert-base-uncased')
    ap.add_argument('--epochs', type=int, default=3)
    ap.add_argument('--batch-size', type=int, default=16)
    ap.add_argument('--grad-accum', type=int, default=2)
    ap.add_argument('--lr', type=float, default=5e-5)
    ap.add_argument('--weight-decay', type=float, default=0.01)
    ap.add_argument('--warmup-ratio', type=float, default=0.06)
    ap.add_argument('--max-len', type=int, default=256)
    ap.add_argument('--val-split', type=float, default=0.2)
    ap.add_argument('--output-version', required=True, help='Version tag (e.g., t1.0.0)')
    ap.add_argument('--exclude-pattern', action='append', default=[], help='Regex patterns to exclude from text (applied to description)')
    args = ap.parse_args()

    df = load_data(args.data)
    # Stratified train/val on priority
    from sklearn.model_selection import train_test_split
    train_df, val_df = train_test_split(df, test_size=args.val_split, random_state=SEED, stratify=df['priority'])
    pri_labels = sorted(df['priority'].unique())
    dep_labels = sorted(df['department'].unique())
    pri2id = {l:i for i,l in enumerate(pri_labels)}
    dep2id = {l:i for i,l in enumerate(dep_labels)}

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    # Default exclusions to mitigate department leakage via enrichment tokens
    # Include common patterns observed in enriched datasets
    default_exclusions = [
        r"__department_[a-z0-9_]+",
        r"__dept_[a-z0-9_]+",
        r"__type_[a-z0-9_]+",
    ]
    patterns = args.exclude_pattern if args.exclude_pattern else default_exclusions
    compiled = [re.compile(p, flags=re.IGNORECASE) for p in patterns]
    train_ds = TicketDataset(train_df, tokenizer, args.max_len, pri2id, dep2id, compiled)
    val_ds = TicketDataset(val_df, tokenizer, args.max_len, pri2id, dep2id, compiled)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    cfg = TrainConfig(model_name=args.model_name, epochs=args.epochs, batch_size=args.batch_size, lr=args.lr, weight_decay=args.weight_decay, warmup_ratio=args.warmup_ratio, max_len=args.max_len, grad_accum=args.grad_accum, device=device)
    model = DualHeadModel(args.model_name, len(pri2id), len(dep2id))
    out_dir = os.path.join('models','transformers', args.output_version)
    train_loop(cfg, model, tokenizer, train_ds, val_ds, pri2id, dep2id, out_dir)
    print(f"Training complete. Artifacts in {out_dir}")

if __name__ == '__main__':
    main()