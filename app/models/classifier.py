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
from scipy import sparse


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
        # Enable extra engineered tokens for priority model only
        self.enable_priority_extra: bool = False
        # Enable interaction composite tokens for priority
        self.enable_priority_interactions: bool = False
        # Optional character n-gram vectorizers (disabled by default)
        self.enable_priority_char: bool = False
        self.enable_department_char: bool = False
        self.priority_char_vectorizer: Optional[TfidfVectorizer] = None
        self.department_char_vectorizer: Optional[TfidfVectorizer] = None
        self.priority_char_ngram_range = (3,5)
        self.department_char_ngram_range = (3,5)
        # Max feature caps for char n-gram vectorizers (kept moderate to avoid huge sparse matrices)
        self.priority_char_max_features = 10000
        self.department_char_max_features = 10000
        # Priority sample weighting (cost-sensitive emphasis per class)
        self.priority_class_weights_override: Optional[dict[str,float]] = None
        # Per-target hyperparameters (currently only C for LogisticRegression)
        self.priority_C: float = 1.0
        self.department_C: float = 1.0
        # Probability calibration flag
        self.calibrate_probabilities: bool = False

    # ------------------------------ Public API ------------------------------
    def train(
        self,
        df: pd.DataFrame,
        class_weight: Optional[str] = None,
        augment_length_buckets: bool = False,
        department_exclude_regexes: Optional[list[str]] = None,
        enable_priority_extra: bool = False,
        priority_C: float = 1.0,
        department_C: float = 1.0,
        calibrate_probabilities: bool = False,
        enable_priority_interactions: bool = False,
        enable_priority_char: bool = False,
        enable_department_char: bool = False,
        priority_char_ngram_range: tuple[int,int] = (3,5),
        department_char_ngram_range: tuple[int,int] = (3,5),
        priority_cost_weights: Optional[dict[str,float]] = None,
        priority_char_max_features: Optional[int] = 10000,
        department_char_max_features: Optional[int] = 10000,
        priority_word_ngram_max: int = 2,
        department_word_ngram_max: int = 2,
        priority_penalty: str = "l2",
        department_penalty: str = "l2",
        priority_l1_ratio: Optional[float] = None,
        department_l1_ratio: Optional[float] = None,
    ) -> None:
        self.class_weight = class_weight
        self.augment_length_buckets = augment_length_buckets
        if department_exclude_regexes:
            self.department_exclude_patterns = department_exclude_regexes
        self.enable_priority_extra = enable_priority_extra
        self.enable_priority_interactions = enable_priority_interactions
        self.enable_priority_char = enable_priority_char
        self.enable_department_char = enable_department_char
        self.priority_char_ngram_range = priority_char_ngram_range
        self.department_char_ngram_range = department_char_ngram_range
        self.priority_class_weights_override = priority_cost_weights
        self.priority_C = priority_C
        self.department_C = department_C
        self.calibrate_probabilities = calibrate_probabilities
        # Persist max feature limits for reproducibility
        self.priority_char_max_features = priority_char_max_features
        self.department_char_max_features = department_char_max_features
        # Store new ngram / penalty settings
        self._priority_word_ngram_max = priority_word_ngram_max
        self._department_word_ngram_max = department_word_ngram_max
        self._priority_penalty = priority_penalty
        self._department_penalty = department_penalty
        self._priority_l1_ratio = priority_l1_ratio
        self._department_l1_ratio = department_l1_ratio

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
            if self.enable_priority_extra:
                extra_tokens = self._priority_extra_tokens(title, desc_original)
                if extra_tokens:
                    combined_priority = f"{combined_priority} {' '.join(extra_tokens)}"
            if self.enable_priority_interactions:
                inter_tokens = self._priority_interaction_tokens(title, desc_original)
                if inter_tokens:
                    combined_priority = f"{combined_priority} {' '.join(inter_tokens)}"
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
        pr_vectorizer = TfidfVectorizer(ngram_range=(1, priority_word_ngram_max), min_df=2)
        X_priority = pr_vectorizer.fit_transform(priority_texts)
        if self.enable_priority_char:
            # Limit dimensionality so training remains tractable; high-dimensional char space can explode
            self.priority_char_vectorizer = TfidfVectorizer(
                analyzer='char',
                ngram_range=self.priority_char_ngram_range,
                min_df=5,
                max_features=self.priority_char_max_features,
            )
            Xp_char = self.priority_char_vectorizer.fit_transform(priority_texts)
            X_priority = sparse.hstack([X_priority, Xp_char], format='csr')
        sample_weight = None
        if self.priority_class_weights_override:
            # map original string labels to weight
            sw = []
            for lbl in df['priority'].tolist():
                sw.append(self.priority_class_weights_override.get(str(lbl), 1.0))
            sample_weight = np.array(sw)
        # Configure solver/penalty
        pr_solver = "saga" if priority_penalty in ("l1", "elasticnet") or self.enable_priority_char else "lbfgs"
        pr_kwargs = {
            'max_iter': 250,
            'class_weight': class_weight if class_weight == "balanced" else None,
            'C': self.priority_C,
            'solver': pr_solver,
            'penalty': priority_penalty,
            'multi_class': 'auto'
        }
        if priority_penalty == 'elasticnet':
            pr_kwargs['l1_ratio'] = priority_l1_ratio if priority_l1_ratio is not None else 0.5
        pr_base = LogisticRegression(**pr_kwargs)
        if self.calibrate_probabilities:
            from sklearn.calibration import CalibratedClassifierCV
            pr_model = CalibratedClassifierCV(pr_base, method="sigmoid", cv=3)
        else:
            pr_model = pr_base
        if sample_weight is not None and not self.calibrate_probabilities:
            pr_model.fit(X_priority, y_priority, sample_weight=sample_weight)
        else:
            pr_model.fit(X_priority, y_priority)

        # Department model ---------------------------------------------------
        dep_encoder = LabelEncoder()
        y_department = dep_encoder.fit_transform(df["department"].tolist())
        dep_vectorizer = TfidfVectorizer(ngram_range=(1, department_word_ngram_max), min_df=2)
        X_department = dep_vectorizer.fit_transform(department_texts)
        if self.enable_department_char:
            self.department_char_vectorizer = TfidfVectorizer(
                analyzer='char',
                ngram_range=self.department_char_ngram_range,
                min_df=5,
                max_features=self.department_char_max_features,
            )
            Xd_char = self.department_char_vectorizer.fit_transform(department_texts)
            X_department = sparse.hstack([X_department, Xd_char], format='csr')
        dep_solver = "saga" if department_penalty in ("l1", "elasticnet") or self.enable_department_char else "lbfgs"
        dep_kwargs = {
            'max_iter': 250,
            'class_weight': class_weight if class_weight == "balanced" else None,
            'C': self.department_C,
            'solver': dep_solver,
            'penalty': department_penalty,
            'multi_class': 'auto'
        }
        if department_penalty == 'elasticnet':
            dep_kwargs['l1_ratio'] = department_l1_ratio if department_l1_ratio is not None else 0.5
        dep_base = LogisticRegression(**dep_kwargs)
        if self.calibrate_probabilities:
            from sklearn.calibration import CalibratedClassifierCV
            dep_model = CalibratedClassifierCV(dep_base, method="sigmoid", cv=3)
        else:
            dep_model = dep_base
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
        if self.enable_priority_extra:
            extra_tokens = self._priority_extra_tokens(title_p, desc_p_original)
            if extra_tokens:
                combined_priority = f"{combined_priority} {' '.join(extra_tokens)}"
        if self.enable_priority_interactions:
            inter_tokens = self._priority_interaction_tokens(title_p, desc_p_original)
            if inter_tokens:
                combined_priority = f"{combined_priority} {' '.join(inter_tokens)}"
        if self.augment_length_buckets:
            combined_priority = f"{combined_priority} {_length_bucket(len(combined_priority.split()))}"

        # Priority
        Xp = self.priority_bundle.vectorizer.transform([combined_priority])
        if self.enable_priority_char and self.priority_char_vectorizer is not None:
            Xp_char = self.priority_char_vectorizer.transform([combined_priority])
            Xp = sparse.hstack([Xp, Xp_char], format='csr')
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
        if self.enable_department_char and self.department_char_vectorizer is not None:
            Xd_char = self.department_char_vectorizer.transform([combined_dep])
            Xd = sparse.hstack([Xd, Xd_char], format='csr')
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
            "enable_priority_extra": self.enable_priority_extra,
            "enable_priority_interactions": self.enable_priority_interactions,
            "enable_priority_char": self.enable_priority_char,
            "enable_department_char": self.enable_department_char,
            "priority_char_ngram_range": self.priority_char_ngram_range,
            "department_char_ngram_range": self.department_char_ngram_range,
            "priority_cost_weights": self.priority_class_weights_override,
            "priority_C": self.priority_C,
            "department_C": self.department_C,
            "calibrate_probabilities": self.calibrate_probabilities,
            # Note: max_features not persisted previously; include now for reproducibility
            "priority_char_max_features": self.priority_char_max_features,
            "department_char_max_features": self.department_char_max_features,
            "priority_word_ngram_max": self._priority_word_ngram_max,
            "department_word_ngram_max": self._department_word_ngram_max,
            "priority_penalty": self._priority_penalty,
            "department_penalty": self._department_penalty,
            "priority_l1_ratio": self._priority_l1_ratio,
            "department_l1_ratio": self._department_l1_ratio,
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
                self.enable_priority_extra = cfg.get("enable_priority_extra", False)
                self.enable_priority_interactions = cfg.get("enable_priority_interactions", False)
                self.enable_priority_char = cfg.get("enable_priority_char", False)
                self.enable_department_char = cfg.get("enable_department_char", False)
                self.priority_char_ngram_range = tuple(cfg.get("priority_char_ngram_range", (3,5)))
                self.department_char_ngram_range = tuple(cfg.get("department_char_ngram_range", (3,5)))
                self.priority_class_weights_override = cfg.get("priority_cost_weights") or None
                self.priority_C = cfg.get("priority_C", 1.0)
                self.department_C = cfg.get("department_C", 1.0)
                self.calibrate_probabilities = cfg.get("calibrate_probabilities", False)
                self.priority_char_max_features = cfg.get("priority_char_max_features", 10000)
                self.department_char_max_features = cfg.get("department_char_max_features", 10000)
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

    def _priority_extra_tokens(self, title: str, desc: str) -> list[str]:
        """Generate additional engineered tokens to aid priority discrimination.

        Heuristics:
          - Keyword groups (urgent/outage, billing/payment, access/security)
          - Structural signals (caps ratio, punctuation density, exclamations, digits)
        """
        text = f"{title} {desc}".lower()
        tokens: list[str] = []
        # Keyword groups (keep compact & general)
        urgent_kw = ["outage", "down", "critical", "severe", "emergency", "unresponsive"]
        high_kw = ["failure", "crash", "error", "corrupt", "broken"]
        billing_kw = ["invoice", "charged", "billing", "payment", "refund"]
        access_kw = ["login", "credential", "password", "access", "locked"]
        def any_kw(words):
            return any(w in text for w in words)
        if any_kw(urgent_kw):
            tokens.append("__KWD_URGENT__")
        if any_kw(high_kw):
            tokens.append("__KWD_HIGH__")
        if any_kw(billing_kw):
            tokens.append("__KWD_BILLING_CTX__")
        if any_kw(access_kw):
            tokens.append("__KWD_ACCESS__")
        # Structural features
        total_chars = max(len(desc), 1)
        caps_chars = sum(1 for c in desc if c.isupper())
        caps_ratio = caps_chars / total_chars
        if caps_ratio > 0.2:
            tokens.append("__STRUCT_CAPS_HEAVY__")
        punct_ratio = sum(1 for c in desc if c in "!?") / total_chars
        if punct_ratio > 0.02:
            tokens.append("__STRUCT_PUNCT_ATTENTION__")
        if desc.count("!") >= 2:
            tokens.append("__STRUCT_MULTI_EXCL__")
        if any(ch.isdigit() for ch in desc):
            tokens.append("__STRUCT_DIGITS__")
        return tokens

    def _priority_interaction_tokens(self, title: str, desc: str) -> list[str]:
        """Composite interaction tokens combining enrichment signals + keyword intents.

        Examples:
          __INT_CSAT_LOW_URGENT__  if low CSAT marker + urgent/outage wording.
          __INT_BILLING_ESCALATE__ if billing/refund keywords + escalation punctuation.
          __INT_ACCESS_URGENT__    if access keywords + urgent/outage wording.
        """
        text_lower = f"{title} {desc}".lower()
        tokens: list[str] = []
        urgent_kw = ["outage", "down", "critical", "severe", "emergency", "unresponsive"]
        billing_kw = ["invoice", "charged", "billing", "payment", "refund"]
        access_kw = ["login", "credential", "password", "access", "locked"]
        refund_kw = ["refund", "chargeback", "reimburs"]
        def any_kw(words):
            return any(w in text_lower for w in words)
        # CSAT low + urgent
        if "__csat_low__" in text_lower and any_kw(urgent_kw):
            tokens.append("__INT_CSAT_LOW_URGENT__")
        # Billing escalation punctuation
        if any_kw(billing_kw) and text_lower.count("!") >= 2:
            tokens.append("__INT_BILLING_ESCALATE__")
        # Refund + urgent
        if any_kw(refund_kw) and any_kw(urgent_kw):
            tokens.append("__INT_REFUND_URGENT__")
        # Access + urgent
        if any_kw(access_kw) and any_kw(urgent_kw):
            tokens.append("__INT_ACCESS_URGENT__")
        return tokens


__all__ = ["TicketClassifier", "PRIORITY_LEVELS"]
