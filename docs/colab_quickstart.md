# Google Colab Quickstart (GPU)

This guide lets you run the transformer trainer on a free GPU in Google Colab, with a stable configuration that avoids the class-collapse issues we observed on CPU.

Highlights of the recipe:
- Loss: Cross-Entropy with label smoothing 0.05 (no focal loss)
- Sampling: no weighted sampler (use class weights = auto)
- Loss weights: priority 1.5, department 1.0
- Department loss warmup: 1 epoch (department weight = 0 for epoch 1, then 1.0)
- Base encoder: distilroberta-base (recommended) or roberta-base
- LR: 3e-5 (distilroberta), 2e-5..3e-5 (roberta)
- Epochs: 5 to start; increase as needed on GPU

---

## 1) Start a Colab Notebook with GPU

- Open https://colab.research.google.com
- New Notebook → Runtime → Change runtime type → Hardware accelerator = GPU → Save

## 2) Install dependencies (run in the first cell)

```python
!pip -q install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
!pip -q install transformers==4.44.2 scikit-learn pandas tqdm
```

(Optional) If PyTorch prebuilt above fails on your GPU, fallback to the default:
```python
!pip -q install torch torchvision torchaudio
```

## 3) Upload your dataset CSV

Expected columns: title, description, priority, department

```python
from google.colab import files
import os
os.makedirs('data', exist_ok=True)
print('Choose your enriched CSV (e.g., enriched_customer_tickets.csv)')
uploaded = files.upload()  # pick your CSV
csv_name = next(iter(uploaded))
os.rename(csv_name, 'data/enriched_customer_tickets.csv')
print('Saved to data/enriched_customer_tickets.csv')
```

## 4) Trainer code (paste once, then reuse)

This cell brings in a slightly simplified version of the repo's trainer.

```python
import re, os, math, json, random
from dataclasses import dataclass
from typing import Dict, Any, List, Optional
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

class TicketDataset(Dataset):
  def __init__(self, df: pd.DataFrame, tokenizer, max_len: int, pri2id: Dict[str,int], dep2id: Dict[str,int], compiled_patterns: Optional[List[re.Pattern]] = None):
    self.df = df.reset_index(drop=True)
    self.tok = tokenizer
    self.max_len = max_len
    self.pri2id = pri2id
    self.dep2id = dep2id
    self._compiled = compiled_patterns or []
  def __len__(self): return len(self.df)
  def __getitem__(self, idx):
    row = self.df.iloc[idx]
    title = str(row.title).strip()
    desc = str(row.description).strip()
    for cre in self._compiled:
      desc = cre.sub(" ", desc)
    enc = self.tok(title, desc, truncation=True, max_length=self.max_len, padding='max_length', return_tensors='pt')
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

def classification_report_dict(y_true, y_pred):
  from sklearn.metrics import classification_report
  return classification_report(y_true, y_pred, output_dict=True, zero_division=0)

@torch.no_grad()
def evaluate(model, loader, device):
  model.eval()
  all_p_pred=[]; all_p_true=[]; all_d_pred=[]; all_d_true=[]
  for batch in loader:
    batch = {k:v.to(device) for k,v in batch.items()}
    lp, ld = model(batch['input_ids'], batch['attention_mask'])
    all_p_pred.extend(torch.argmax(lp, dim=1).cpu().tolist())
    all_p_true.extend(batch['priority_label'].cpu().tolist())
    all_d_pred.extend(torch.argmax(ld, dim=1).cpu().tolist())
    all_d_true.extend(batch['department_label'].cpu().tolist())
  rep_p = classification_report_dict(all_p_true, all_p_pred)
  rep_d = classification_report_dict(all_d_true, all_d_pred)
  return {
    'priority': {'macro_f1': rep_p['macro avg']['f1-score'], 'accuracy': rep_p['accuracy'], 'report': rep_p},
    'department': {'macro_f1': rep_d['macro avg']['f1-score'], 'accuracy': rep_d['accuracy'], 'report': rep_d},
  }


def compute_class_weights(labels: List[int], num_classes: int) -> torch.Tensor:
  import numpy as np
  counts = np.bincount(labels, minlength=num_classes).astype(float)
  counts[counts == 0] = 1.0
  weights = (counts.sum() / (len(counts) * counts))
  return torch.tensor(weights, dtype=torch.float32)


def train(
  data_csv: str = 'data/enriched_customer_tickets.csv',
  model_name: str = 'distilroberta-base',
  epochs: int = 5,
  batch_size: int = 16,
  grad_accum: int = 2,
  lr: float = 3e-5,
  weight_decay: float = 0.01,
  warmup_ratio: float = 0.1,
  max_len: int = 256,
  exclude_patterns = [r"__department_[a-z0-9_]+", r"__dept_[a-z0-9_]+", r"__type_[a-z0-9_]+"],
  loss_weight_priority: float = 1.5,
  loss_weight_department: float = 1.0,
  dept_loss_warmup_epochs: int = 1,
  label_smoothing: float = 0.05,
  output_dir: str = 'models/transformers/colab_run'
):
  device = 'cuda' if torch.cuda.is_available() else 'cpu'
  df = load_data(data_csv)
  from sklearn.model_selection import train_test_split
  train_df, val_df = train_test_split(df, test_size=0.2, random_state=SEED, stratify=df['priority'])
  pri_labels = sorted(df['priority'].unique()); dep_labels = sorted(df['department'].unique())
  pri2id = {l:i for i,l in enumerate(pri_labels)}; dep2id = {l:i for i,l in enumerate(dep_labels)}

  tok = AutoTokenizer.from_pretrained(model_name)
  compiled = [re.compile(p, flags=re.IGNORECASE) for p in exclude_patterns]
  train_ds = TicketDataset(train_df, tok, max_len, pri2id, dep2id, compiled)
  val_ds   = TicketDataset(val_df, tok, max_len, pri2id, dep2id, compiled)

  # class weights
  pri_ids = [pri2id[l] for l in train_df['priority'].tolist()]
  dep_ids = [dep2id[l] for l in train_df['department'].tolist()]
  cw_p = compute_class_weights(pri_ids, len(pri2id)).to(device)
  cw_d = compute_class_weights(dep_ids, len(dep2id)).to(device)

  train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
  val_loader = DataLoader(val_ds, batch_size=batch_size)

  cfg = TrainConfig(model_name=model_name, epochs=epochs, batch_size=batch_size, lr=lr, weight_decay=weight_decay, warmup_ratio=warmup_ratio, max_len=max_len, grad_accum=grad_accum, device=device)
  model = DualHeadModel(model_name, len(pri2id), len(dep2id)).to(device)

  no_decay = ["bias","LayerNorm.weight"]
  grouped = [
    {"params":[p for n,p in model.named_parameters() if not any(nd in n for nd in no_decay)],"weight_decay":weight_decay},
    {"params":[p for n,p in model.named_parameters() if any(nd in n for nd in no_decay)],"weight_decay":0.0},
  ]
  opt = torch.optim.AdamW(grouped, lr=lr)
  total_steps = epochs * math.ceil(len(train_loader)/grad_accum)
  warmup_steps = int(total_steps * warmup_ratio)
  sch = get_linear_schedule_with_warmup(opt, warmup_steps, total_steps)

  ce_p = nn.CrossEntropyLoss(weight=cw_p, label_smoothing=label_smoothing)
  ce_d = nn.CrossEntropyLoss(weight=cw_d, label_smoothing=label_smoothing)

  best = -1.0
  os.makedirs(output_dir, exist_ok=True)

  for epoch in range(1, epochs+1):
    model.train(); total=0.0
    pbar = tqdm(train_loader, desc=f"Epoch {epoch}")
    for step,batch in enumerate(pbar, start=1):
      batch = {k:v.to(device) for k,v in batch.items()}
      lp, ld = model(batch['input_ids'], batch['attention_mask'])
      loss_p = ce_p(lp, batch['priority_label'])
      loss_d = ce_d(ld, batch['department_label'])
      dept_w = 0.0 if epoch <= dept_loss_warmup_epochs else loss_weight_department
      loss = loss_weight_priority * loss_p + dept_w * loss_d
      loss.backward()
      if step % grad_accum == 0:
        opt.step(); sch.step(); opt.zero_grad()
      total += loss.item()
      pbar.set_postfix(loss=f"{total/step:.4f}")

    metrics = evaluate(model, val_loader, device)
    score = metrics['priority']['macro_f1']
    if score > best:
      best = score
      torch.save(model.state_dict(), os.path.join(output_dir, 'pytorch_model.bin'))
      with open(os.path.join(output_dir,'metrics.json'),'w') as f: json.dump(metrics, f, indent=2)
      print(f"Saved new best (priority macro-F1={best:.4f})")

  # Save label maps & tokenizer
  maps = {
    'priority_id2label': {v:k for k,v in pri2id.items()},
    'priority_label2id': pri2id,
    'department_id2label': {v:k for k,v in dep2id.items()},
    'department_label2id': dep2id,
  }
  with open(os.path.join(output_dir,'label_mappings.json'),'w') as f: json.dump(maps,f,indent=2)
  from transformers import PreTrainedTokenizerFast
  tok.save_pretrained(os.path.join(output_dir,'tokenizer'))
  print('Done. Artifacts at', output_dir)
```

## 5) Run the training (distilroberta-base recommended)

```python
train(
  data_csv='data/enriched_customer_tickets.csv',
  model_name='distilroberta-base',  # or 'roberta-base'
  epochs=5,
  batch_size=16,
  grad_accum=2,
  lr=3e-5,  # try 2e-5..3e-5 for roberta-base
  warmup_ratio=0.1,
  max_len=256,
  loss_weight_priority=1.5,
  loss_weight_department=1.0,
  dept_loss_warmup_epochs=1,
  label_smoothing=0.05,
  output_dir='models/transformers/colab_run'
)
```

You’ll see a live tqdm progress bar per epoch. After each epoch the script evaluates and saves the best checkpoint by priority macro‑F1.

## 6) Inspect metrics

```python
import json, os
with open('models/transformers/colab_run/metrics.json') as f:
  print(json.dumps(json.load(f), indent=2))
```

## 7) Download artifacts (optional)

```python
from google.colab import files
for fn in ['pytorch_model.bin','label_mappings.json','metrics.json']:
  path = os.path.join('models/transformers/colab_run', fn)
  if os.path.exists(path):
    files.download(path)
```

## Tips
- If metrics collapse to a single class, re-run with a slightly higher LR (e.g., 3e-5 → 2.5e-5 or 2e-5) and keep the simplified loss (no focal, no weighted sampler).
- Consider increasing epochs to 6–7 on GPU.
- To try `roberta-base`, change `model_name` and set `lr=2e-5..3e-5`; expect longer epochs vs distilroberta.
