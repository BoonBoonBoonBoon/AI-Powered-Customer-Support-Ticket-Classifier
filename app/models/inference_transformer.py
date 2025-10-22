from __future__ import annotations
from typing import Dict, Any
import json
import os
import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer
import re


def _apply_exclusions(text: str, compiled_patterns: list[re.Pattern[str]] | None) -> str:
    if not compiled_patterns:
        return text
    for cre in compiled_patterns:
        text = cre.sub(" ", text)
    return re.sub(r"\s+", " ", text).strip()


def preprocess_text(title: str, description: str, max_len: int = 512, compiled_patterns: list[re.Pattern[str]] | None = None) -> str:
    title = title.strip()
    description = _apply_exclusions(description.strip(), compiled_patterns)
    return f"[TITLE] {title} [DESC] {description}"[:2000]


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
        pooled = out.last_hidden_state[:, 0]
        pooled = self.dropout(pooled)
        return self.priority_head(pooled), self.department_head(pooled)


class TransformerInference:
    """Transformer runtime wrapper using the training architecture.

    Loads tokenizer, label mappings, and pytorch_model.bin from the artifact directory
    and serves predictions with softmax confidences.
    """

    def __init__(self, manifest: Dict[str, Any], artifacts: Dict[str, str | os.PathLike]):
        self.manifest = manifest
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # Load label mappings
        label_map_path = artifacts.get("label_map_uri")
        if not label_map_path:
            raise ValueError("label_map_uri missing in artifacts")
        with open(label_map_path, "r", encoding="utf-8") as f:
            lm = json.load(f)
        self.priority_id2label = {int(k): v for k, v in lm["priority_id2label"].items()}
        self.department_id2label = {int(k): v for k, v in lm["department_id2label"].items()}
        self.priority_label2id = {v: int(k) for k, v in self.priority_id2label.items()}
        self.department_label2id = {v: int(k) for k, v in self.department_id2label.items()}

        base_model = manifest.get("training", {}).get("base_model", "distilbert-base-uncased")
        self.max_len = int(manifest.get("training", {}).get("max_length", 256))
        # Optional exclusion patterns (to prevent leakage); if not present, no exclusions are applied
        exclude_patterns = manifest.get("training", {}).get("exclude_patterns", [])
        self._compiled_patterns = [re.compile(p, flags=re.IGNORECASE) for p in exclude_patterns]

        # Tokenizer
        tok_uri = artifacts.get("tokenizer_uri")
        if not tok_uri:
            raise ValueError("tokenizer_uri missing in artifacts")
        self.tok = AutoTokenizer.from_pretrained(tok_uri)

        # Model
        p_classes = len(self.priority_id2label)
        d_classes = len(self.department_id2label)
        self.model = DualHeadModel(base_model, p_classes, d_classes)
        weights_path = artifacts.get("weights_uri")
        if not weights_path:
            raise ValueError("weights_uri missing in artifacts")
        state = torch.load(weights_path, map_location="cpu")
        self.model.load_state_dict(state)
        self.model.to(self.device)
        self.model.eval()

        self.softmax = nn.Softmax(dim=-1)

    def predict(self, title: str, description: str) -> Dict[str, Any]:
        text = preprocess_text(title, description, self.max_len, self._compiled_patterns)
        enc = self.tok(text, truncation=True, max_length=self.max_len, padding="max_length", return_tensors="pt")
        enc = {k: v.to(self.device) for k, v in enc.items()}
        with torch.inference_mode():
            lp, ld = self.model(enc["input_ids"], enc["attention_mask"])
            p_probs = self.softmax(lp)[0].detach().cpu()
            d_probs = self.softmax(ld)[0].detach().cpu()
            p_idx = int(torch.argmax(p_probs).item())
            d_idx = int(torch.argmax(d_probs).item())
        return {
            "priority": self.priority_id2label[p_idx],
            "priority_conf": float(p_probs[p_idx].item()),
            "department": self.department_id2label[d_idx],
            "department_conf": float(d_probs[d_idx].item()),
        }


__all__ = ["TransformerInference"]
