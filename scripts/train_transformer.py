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
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
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
    """Clean and return title+description as separate sequences.

    We apply exclusion patterns to the description only to mitigate leakage tokens.
    The tokenizer will handle special tokens (e.g., [CLS]/[SEP]) when passed a text_pair.
    """
    title = title.strip()
    description = _apply_exclusions(description.strip(), compiled_patterns)
    # Return as a pair; truncation occurs inside the tokenizer.
    return title, description

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
        t, d = preprocess_text(str(row.title), str(row.description), self.max_len, self._compiled_patterns)
        # Provide title as text and description as text_pair so tokenizer can insert [SEP]
        enc = self.tok(t, d, truncation=True, max_length=self.max_len, padding='max_length', return_tensors='pt')
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

def _compute_class_weights(labels: list[int], num_classes: int) -> torch.Tensor:
    import numpy as np
    counts = np.bincount(labels, minlength=num_classes).astype(float)
    # Avoid division by zero
    counts[counts == 0] = 1.0
    weights = (counts.sum() / (len(counts) * counts))
    return torch.tensor(weights, dtype=torch.float32)


class FocalLoss(nn.Module):
    def __init__(self, weight: torch.Tensor | None = None, gamma: float = 2.0, reduction: str = 'mean'):
        super().__init__()
        self.weight = weight
        self.gamma = gamma
        self.reduction = reduction
        self.ce = nn.CrossEntropyLoss(weight=weight, reduction='none')

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # Cross-entropy per-sample
        ce_loss = self.ce(logits, target)
        # Convert CE to pt = exp(-ce)
        pt = torch.exp(-ce_loss)
        focal = ((1 - pt) ** self.gamma) * ce_loss
        if self.reduction == 'mean':
            return focal.mean()
        elif self.reduction == 'sum':
            return focal.sum()
        return focal


def train_loop(cfg: TrainConfig, model, tokenizer, train_ds, val_ds, pri2id, dep2id, output_dir: str, class_weight_priority: str = 'none', class_weight_department: str = 'auto', label_smoothing: float = 0.0, select_metric: str = 'priority', loss_weight_priority: float = 1.0, loss_weight_department: float = 1.0, priority_labels_for_weights: list[int] | None = None, department_labels_for_weights: list[int] | None = None, weighted_sampler: str = 'none', dept_loss_warmup_epochs: int = 0, focal_priority_gamma: float = 0.0):
    # Build train loader (optionally with weighted sampler by priority)
    if weighted_sampler == 'priority' and priority_labels_for_weights is not None:
        import numpy as np
        counts = np.bincount(priority_labels_for_weights, minlength=len(pri2id)).astype(float)
        counts[counts == 0] = 1.0
        class_w = (counts.sum() / (len(counts) * counts))
        sample_weights = [class_w[y] for y in priority_labels_for_weights]
        sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)
        train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, sampler=sampler)
    else:
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
    # Prepare (optional) class weights
    cw_p = None
    cw_d = None
    if class_weight_priority == 'auto':
        if priority_labels_for_weights is None:
            # Fallback (slower): iterate dataset
            pri_labels = [train_ds[i]['priority_label'].item() for i in range(len(train_ds))]
        else:
            pri_labels = priority_labels_for_weights
        cw_p = _compute_class_weights(pri_labels, len(pri2id)).to(cfg.device)
    if class_weight_department == 'auto':
        if department_labels_for_weights is None:
            dep_labels = [train_ds[i]['department_label'].item() for i in range(len(train_ds))]
        else:
            dep_labels = department_labels_for_weights
        cw_d = _compute_class_weights(dep_labels, len(dep2id)).to(cfg.device)

    # Priority loss: optional focal; if focal used, ignore label smoothing
    loss_fn_p = FocalLoss(weight=cw_p, gamma=focal_priority_gamma) if focal_priority_gamma and focal_priority_gamma > 0 else nn.CrossEntropyLoss(weight=cw_p, label_smoothing=label_smoothing)
    loss_fn_d = nn.CrossEntropyLoss(weight=cw_d, label_smoothing=label_smoothing)
    best_score = -1.0
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
            # Warmup: optionally suppress department loss for initial epochs
            dept_w = 0.0 if epoch <= dept_loss_warmup_epochs else loss_weight_department
            loss = loss_weight_priority * loss_p + dept_w * loss_d
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
        # Model selection criterion
        if select_metric == 'priority':
            score = metrics['priority']['macro_f1']
        elif select_metric == 'department':
            score = metrics['department']['macro_f1']
        else:
            score = metrics['priority']['macro_f1'] + metrics['department']['macro_f1']
        # Save best
        if score > best_score:
            best_score = score
            torch.save(model.state_dict(), os.path.join(output_dir, 'pytorch_model.bin'))
            with open(os.path.join(output_dir,'metrics.json'),'w',encoding='utf-8') as f:
                json.dump(metrics, f, indent=2)
            print(f"Saved new best ({select_metric} score={best_score:.4f})")

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
    ap.add_argument('--class-weight-priority', choices=['none','auto'], default='none', help='Apply automatic inverse-frequency class weights for priority head')
    ap.add_argument('--class-weight-department', choices=['none','auto'], default='auto', help='Apply automatic inverse-frequency class weights for department head')
    ap.add_argument('--label-smoothing', type=float, default=0.0, help='Label smoothing for CrossEntropyLoss (e.g., 0.05)')
    ap.add_argument('--select-metric', choices=['priority','department','sum'], default='priority', help='Model selection criterion on validation')
    ap.add_argument('--loss-weight-priority', type=float, default=1.0, help='Weight for priority loss in multi-task sum')
    ap.add_argument('--loss-weight-department', type=float, default=1.0, help='Weight for department loss in multi-task sum')
    ap.add_argument('--weighted-sampler', choices=['none','priority'], default='none', help='Use WeightedRandomSampler by priority on the training set')
    ap.add_argument('--dept-loss-warmup-epochs', type=int, default=0, help='Number of initial epochs to suppress department loss (set weight=0)')
    ap.add_argument('--focal-priority-gamma', type=float, default=0.0, help='Enable focal loss for priority with given gamma (0 disables)')
    ap.add_argument('--device', choices=['auto','cuda','cpu','dml'], default='auto', help="Compute device: 'auto' (prefer CUDA, then DirectML, else CPU), or force 'cuda'/'cpu'/'dml'")
    ap.add_argument('--init-weights', type=str, default='', help='Optional path to a state_dict (.bin) to initialize model weights from a prior run')
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

    # Select device per user choice
    def select_device(choice: str):
        if choice == 'cuda':
            if torch.cuda.is_available():
                return 'cuda', "CUDA GPU"
            raise RuntimeError("--device cuda requested but CUDA is not available in this environment")
        if choice == 'cpu':
            return 'cpu', "CPU"
        if choice == 'dml':
            try:
                import torch_directml
                return torch_directml.device(), "DirectML"
            except Exception as e:
                raise RuntimeError(f"--device dml requested but torch-directml is not available: {e}")
        # auto: prefer CUDA, then DirectML, else CPU
        if torch.cuda.is_available():
            return 'cuda', "CUDA GPU"
        try:
            import torch_directml
            return torch_directml.device(), "DirectML"
        except Exception:
            return 'cpu', "CPU"

    device, device_name = select_device(args.device)
    print(f"Using device: {device_name}")
    cfg = TrainConfig(model_name=args.model_name, epochs=args.epochs, batch_size=args.batch_size, lr=args.lr, weight_decay=args.weight_decay, warmup_ratio=args.warmup_ratio, max_len=args.max_len, grad_accum=args.grad_accum, device=device)
    model = DualHeadModel(args.model_name, len(pri2id), len(dep2id))
    # Optionally initialize from a previous checkpoint's state_dict
    if args.init_weights:
        try:
            if os.path.isfile(args.init_weights):
                state = torch.load(args.init_weights, map_location='cpu')
                missing_unexpected = model.load_state_dict(state, strict=False)
                print(f"Initialized weights from {args.init_weights}. Load result: {missing_unexpected}")
            else:
                print(f"--init-weights path not found: {args.init_weights}")
        except Exception as e:
            print(f"Warning: failed to load init weights from {args.init_weights}: {e}")
    out_dir = os.path.join('models','transformers', args.output_version)
    # Precompute label ids for fast class weights
    pri_label_ids_train = [pri2id[l] for l in train_df['priority'].tolist()]
    dep_label_ids_train = [dep2id[l] for l in train_df['department'].tolist()]

    train_loop(
        cfg, model, tokenizer, train_ds, val_ds, pri2id, dep2id, out_dir,
        class_weight_priority=args.class_weight_priority,
        class_weight_department=args.class_weight_department,
        label_smoothing=args.label_smoothing,
        select_metric=args.select_metric,
        loss_weight_priority=args.loss_weight_priority,
        loss_weight_department=args.loss_weight_department,
        priority_labels_for_weights=pri_label_ids_train,
        department_labels_for_weights=dep_label_ids_train,
        weighted_sampler=args.weighted_sampler,
        dept_loss_warmup_epochs=args.dept_loss_warmup_epochs,
        focal_priority_gamma=args.focal_priority_gamma,
    )
    print(f"Training complete. Artifacts in {out_dir}")

if __name__ == '__main__':
    main()