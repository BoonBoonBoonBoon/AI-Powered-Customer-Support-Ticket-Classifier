"""Advanced TicketClassifier using scikit-learn.

Features:
 - Separate TF-IDF + LogisticRegression models for priority and department
 - Optional class weighting (balanced)
 - Optional length bucket feature augmentation (added as synthetic tokens)
 - Title + description fusion with markers
 - Deterministic training via external seeding

Artifacts saved (in version directory provided by training pipeline):
  priority_model.joblib
  department_model.joblib
  priority_vectorizer.joblib
  department_vectorizer.joblib
  classifier_config.json (simple metadata)

Interface intentionally mirrors previously referenced advanced classifier.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple, Optional, Dict, Any
import json
import os
import re

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
import joblib


PRIORITY_LEVELS = ["Urgent", "High", "Medium", "Low"]


def _length_bucket(n_tokens: int) -> str:
    if n_tokens < 8:
        return "__LEN_VSHORT__"
    if n_tokens < 20:
        return "__LEN_SHORT__"
    if n_tokens < 40:
        return "__LEN_MEDIUM__"
    if n_tokens < 70:
        return "__LEN_LONG__"
    return "__LEN_VLONG__"


@dataclass
class _ModelBundle:
    vectorizer: TfidfVectorizer
    model: LogisticRegression
    label_encoder: LabelEncoder


class TicketClassifier:
    def __init__(self):
        self.priority_bundle: Optional[_ModelBundle] = None
        self.department_bundle: Optional[_ModelBundle] = None
        self.is_trained: bool = False
        self.augment_length_buckets: bool = False
        self.class_weight: Optional[str] = None  # 'balanced' or None
        # Token exclusion patterns (regex) applied ONLY for department model to reduce leakage
        self.department_exclude_patterns: list[str] = []

    # ------------------------------ Public API ------------------------------
    def train(
        self,
        df: pd.DataFrame,
        class_weight: Optional[str] = None,
        augment_length_buckets: bool = False,
        department_exclude_regexes: Optional[list[str]] = None,
    ) -> None:
        self.class_weight = class_weight
        self.augment_length_buckets = augment_length_buckets
        if department_exclude_regexes:
            self.department_exclude_patterns = department_exclude_regexes

        required = {"title", "description", "priority", "department"}
        if not required.issubset(df.columns):
            missing = required - set(df.columns)
            raise ValueError(f"Missing required columns for training: {missing}")

        # Build combined text for priority model & potentially filtered for department
        priority_texts: list[str] = []
        department_texts: list[str] = []
        compiled_exclusions = [re.compile(p) for p in self.department_exclude_patterns]
        for _, row in df.iterrows():
            title = self._preprocess(row["title"])  # type: ignore[arg-type]
            desc_original = self._preprocess(row["description"])  # type: ignore[arg-type]
            combined_priority = f"title: {title}\nbody: {desc_original}".strip()
            if augment_length_buckets:
                bucket = _length_bucket(len(combined_priority.split()))
                combined_priority = f"{combined_priority} {bucket}"
            priority_texts.append(combined_priority)

            # Department version: remove excluded tokens
            dep_desc = desc_original
            if compiled_exclusions:
                for cre in compiled_exclusions:
                    dep_desc = cre.sub(" ", dep_desc)
                dep_desc = re.sub(r"\s+", " ", dep_desc).strip()
            combined_dep = f"title: {title}\nbody: {dep_desc}".strip()
            if augment_length_buckets:
                bucket = _length_bucket(len(combined_dep.split()))
                combined_dep = f"{combined_dep} {bucket}"
            department_texts.append(combined_dep)

        # Priority model -----------------------------------------------------
        pr_encoder = LabelEncoder()
        y_priority = pr_encoder.fit_transform(df["priority"].tolist())
        pr_vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=2)
        X_priority = pr_vectorizer.fit_transform(priority_texts)
        pr_model = LogisticRegression(
            max_iter=200,
            class_weight=class_weight if class_weight == "balanced" else None,
        )
        pr_model.fit(X_priority, y_priority)

        # Department model ---------------------------------------------------
        dep_encoder = LabelEncoder()
        y_department = dep_encoder.fit_transform(df["department"].tolist())
        dep_vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=2)
        X_department = dep_vectorizer.fit_transform(department_texts)
        dep_model = LogisticRegression(
            max_iter=200,
            class_weight=class_weight if class_weight == "balanced" else None,
        )
        dep_model.fit(X_department, y_department)

        self.priority_bundle = _ModelBundle(pr_vectorizer, pr_model, pr_encoder)
        self.department_bundle = _ModelBundle(dep_vectorizer, dep_model, dep_encoder)
        self.is_trained = True

    def predict(self, title: str, description: str) -> Tuple[str, str, float, float]:
        if not self.is_trained or not self.priority_bundle or not self.department_bundle:
            raise RuntimeError("Model not trained or bundles missing")

        title_p = self._preprocess(title)
        desc_p_original = self._preprocess(description)
        combined_priority = f"title: {title_p}\nbody: {desc_p_original}".strip()
        if self.augment_length_buckets:
            combined_priority = f"{combined_priority} {_length_bucket(len(combined_priority.split()))}"

        # Priority
        Xp = self.priority_bundle.vectorizer.transform([combined_priority])
        pr_probs = self.priority_bundle.model.predict_proba(Xp)[0]
        pr_idx = int(np.argmax(pr_probs))
        pr_label = self.priority_bundle.label_encoder.inverse_transform([pr_idx])[0]
        # Department (apply exclusions if configured)
        dep_text = desc_p_original
        if self.department_exclude_patterns:
            for pattern in self.department_exclude_patterns:
                dep_text = re.sub(pattern, " ", dep_text)
            dep_text = re.sub(r"\s+", " ", dep_text).strip()
        combined_dep = f"title: {title_p}\nbody: {dep_text}".strip()
        if self.augment_length_buckets:
            combined_dep = f"{combined_dep} {_length_bucket(len(combined_dep.split()))}"
        Xd = self.department_bundle.vectorizer.transform([combined_dep])
        dep_probs = self.department_bundle.model.predict_proba(Xd)[0]
        dep_idx = int(np.argmax(dep_probs))
        dep_label = self.department_bundle.label_encoder.inverse_transform([dep_idx])[0]

        return pr_label, dep_label, float(pr_probs[pr_idx]), float(dep_probs[dep_idx])

    def save_models(self, output_dir: str) -> None:
        if not self.is_trained:
            raise RuntimeError("Cannot save before training")
        os.makedirs(output_dir, exist_ok=True)
        assert self.priority_bundle and self.department_bundle
        joblib.dump(self.priority_bundle.vectorizer, os.path.join(output_dir, "priority_vectorizer.joblib"))
        joblib.dump(self.priority_bundle.model, os.path.join(output_dir, "priority_model.joblib"))
        joblib.dump(self.priority_bundle.label_encoder, os.path.join(output_dir, "priority_encoder.joblib"))
        joblib.dump(self.department_bundle.vectorizer, os.path.join(output_dir, "department_vectorizer.joblib"))
        joblib.dump(self.department_bundle.model, os.path.join(output_dir, "department_model.joblib"))
        joblib.dump(self.department_bundle.label_encoder, os.path.join(output_dir, "department_encoder.joblib"))
        config = {
            "class_weight": self.class_weight,
            "augment_length_buckets": self.augment_length_buckets,
            "version": 1,
            "model_type": "sklearn_logreg_tfidf",
            "department_exclude_patterns": self.department_exclude_patterns,
        }
        with open(os.path.join(output_dir, "classifier_config.json"), "w", encoding="utf-8") as f:
            json.dump(config, f, indent=2)

    def load_models(self, input_dir: str) -> None:
        self.priority_bundle = _ModelBundle(
            joblib.load(os.path.join(input_dir, "priority_vectorizer.joblib")),
            joblib.load(os.path.join(input_dir, "priority_model.joblib")),
            joblib.load(os.path.join(input_dir, "priority_encoder.joblib")),
        )
        self.department_bundle = _ModelBundle(
            joblib.load(os.path.join(input_dir, "department_vectorizer.joblib")),
            joblib.load(os.path.join(input_dir, "department_model.joblib")),
            joblib.load(os.path.join(input_dir, "department_encoder.joblib")),
        )
        # Load config if present
        cfg_path = os.path.join(input_dir, "classifier_config.json")
        if os.path.exists(cfg_path):
            try:
                with open(cfg_path, "r", encoding="utf-8") as f:
                    cfg = json.load(f)
                self.class_weight = cfg.get("class_weight")
                self.augment_length_buckets = cfg.get("augment_length_buckets", False)
                self.department_exclude_patterns = cfg.get("department_exclude_patterns", []) or []
            except Exception:
                pass
        self.is_trained = True

    # --------------------------- Internal Helpers ---------------------------
    def _preprocess(self, text: Any) -> str:
        if not text:
            return ""
        if not isinstance(text, str):
            text = str(text)
        text = text.strip().lower()
        text = re.sub(r"\s+", " ", text)
        return text


__all__ = ["TicketClassifier", "PRIORITY_LEVELS"]
