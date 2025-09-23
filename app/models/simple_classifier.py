"""Simple rule and keyword based ticket classifier.

This file was (re)created because tests reference
`app.models.simple_classifier.SimpleTicketClassifier` but the file was
missing from the repository. It provides a lightweight, dependency-free
classifier implementation with the interface expected by tests and the
`simple_main` FastAPI fallback application.

Interface methods expected by tests:
 - predict(title, description) -> (priority, department, p_conf, d_conf)
 - save_models(output_dir)
 - load_models(input_dir)
 - _preprocess_text(text)
 - attribute: is_trained (bool)

The implementation uses straightforward keyword heuristics and assigns
confidence scores based on matched rule strength. This is intentionally
simple so the project can run in constrained environments (e.g. where
scientific Python stack cannot be installed) while the more advanced
training pipeline (TF-IDF + Logistic Regression) can be built separately.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from typing import Dict, List, Tuple


PRIORITY_LEVELS = ["Urgent", "High", "Medium", "Low"]
DEPARTMENTS = ["Tech Support", "Billing", "Sales"]


@dataclass
class Rule:
    pattern: re.Pattern
    priority_delta: int = 0  # Negative lowers, positive raises urgency
    department: str | None = None
    weight: float = 1.0


class SimpleTicketClassifier:
    """Heuristic / keyword based classifier.

    The model assigns:
      - Priority: derived from keyword scores (higher score -> higher urgency)
      - Department: best matching department keyword rules
    """

    def __init__(self) -> None:
        self.is_trained = True  # Rule-based system is always 'trained'
        self.config: Dict[str, any] = {}

        # Keyword rule sets
        self.priority_rules: List[Rule] = [
            Rule(re.compile(r"\b(outage|down|unreachable|data loss)\b", re.I), priority_delta=3, weight=2.0),
            Rule(re.compile(r"\b(error|failed|failure|crash)\b", re.I), priority_delta=2, weight=1.5),
            Rule(re.compile(r"\b(slow|degraded|delay)\b", re.I), priority_delta=1, weight=1.0),
            Rule(re.compile(r"\b(question|inquiry|info|information)\b", re.I), priority_delta=-1, weight=0.8),
            Rule(re.compile(r"\b(feature request|enhancement|nice to have)\b", re.I), priority_delta=-2, weight=1.2),
        ]

        self.department_rules: List[Rule] = [
            Rule(re.compile(r"\b(server|infrastructure|login|bug|issue|timeout|database)\b", re.I), department="Tech Support", weight=2.0),
            Rule(re.compile(r"\b(invoice|billing|payment|credit card|charge|refund)\b", re.I), department="Billing", weight=2.2),
            Rule(re.compile(r"\b(pricing|quote|demo|trial|purchase|plan|upgrade)\b", re.I), department="Sales", weight=1.8),
        ]

        # Base priority mapping (score thresholds)
        self.priority_thresholds = {
            "Urgent": 5.0,
            "High": 3.0,
            "Medium": 1.0,
            "Low": -10.0,  # fallback
        }

    # Public API -----------------------------------------------------
    def predict(self, title: str, description: str) -> Tuple[str, str, float, float]:
        title = self._preprocess_text(title)
        description = self._preprocess_text(description)
        combined = f"{title} {description}".strip()

        priority_score = self._score_priority(combined)
        priority = self._map_priority(priority_score)
        department, dept_score, max_possible = self._assign_department(combined)

        # Confidence heuristics
        priority_conf = self._priority_confidence(priority_score)
        dept_conf = (dept_score / max_possible) if max_possible > 0 else 0.0

        return priority, department, round(priority_conf, 3), round(dept_conf, 3)

    def save_models(self, output_dir: str) -> None:
        os.makedirs(output_dir, exist_ok=True)
        config = {
            "priority_thresholds": self.priority_thresholds,
            "version": 1,
            "type": "rule_based_simple",
        }
        with open(os.path.join(output_dir, "classifier_config.json"), "w", encoding="utf-8") as f:
            json.dump(config, f, indent=2)

    def load_models(self, input_dir: str) -> None:
        try:
            with open(os.path.join(input_dir, "classifier_config.json"), "r", encoding="utf-8") as f:
                data = json.load(f)
            if "priority_thresholds" in data:
                self.priority_thresholds = data["priority_thresholds"]
            self.is_trained = True
        except FileNotFoundError:
            # Non-fatal: remain with defaults
            self.is_trained = True

    # Internal helpers ------------------------------------------------
    def _preprocess_text(self, text) -> str:  # noqa: D401 (simple helper)
        if not text:
            return ""
        if not isinstance(text, str):
            text = str(text)
        return text.strip().lower()

    def _score_priority(self, text: str) -> float:
        score = 0.0
        for rule in self.priority_rules:
            if rule.pattern.search(text):
                score += rule.priority_delta * rule.weight
        # Length heuristic: very short tickets less likely urgent
        length_tokens = len(text.split())
        if length_tokens < 4:
            score -= 0.5
        return score

    def _map_priority(self, score: float) -> str:
        for level in ("Urgent", "High", "Medium"):
            if score >= self.priority_thresholds[level]:
                return level
        return "Low"

    def _assign_department(self, text: str) -> Tuple[str, float, float]:
        dept_scores = {d: 0.0 for d in DEPARTMENTS}
        max_possible = 0.0
        for rule in self.department_rules:
            max_possible += rule.weight
            if rule.pattern.search(text):
                if rule.department:
                    dept_scores[rule.department] += rule.weight
        # Pick department with highest score; default to Tech Support
        department = max(dept_scores, key=dept_scores.get)
        return department, dept_scores[department], max_possible

    def _priority_confidence(self, score: float) -> float:
        # Map raw score to 0-1 range using piecewise scaling
        if score <= 0:
            return max(0.05, min(0.3, 0.1 + (score / 10)))
        if score >= 6:
            return 0.95
        return 0.3 + (score / 6) * 0.6  # between 0.3 and 0.9


# Alias for compatibility (some imports expect TicketClassifier name)
TicketClassifier = SimpleTicketClassifier

__all__ = [
    "SimpleTicketClassifier",
    "TicketClassifier",
    "PRIORITY_LEVELS",
    "DEPARTMENTS",
]
