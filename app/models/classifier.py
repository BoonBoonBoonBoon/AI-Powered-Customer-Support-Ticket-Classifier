import os
import joblib
import pandas as pd
from typing import Tuple
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder

PRIORITY_LABELS = ["Urgent", "High", "Medium", "Low"]
DEPARTMENT_LABELS = ["Tech Support", "Billing", "Sales"]

class TicketClassifier:
    def __init__(self):
        self.priority_model: Pipeline | None = None
        self.department_model: Pipeline | None = None
        self.priority_encoder = LabelEncoder()
        self.department_encoder = LabelEncoder()
        self.is_trained = False

    def _preprocess_text(self, text: str | None) -> str:
        if text is None:
            return ""
        return str(text).strip().lower()

    def _prepare_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        required = ["title", "description", "priority", "department"]
        missing = [c for c in required if c not in df.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")
        df = df.copy()
        df["title"].fillna("", inplace=True)
        df["description"].fillna("", inplace=True)
        df["text"] = (df["title"].astype(str) + " " + df["description"].astype(str)).str.lower()
        return df

    def train(self, df: pd.DataFrame):
        df_prepared = self._prepare_dataframe(df)
        X = df_prepared["text"]

        # Encode labels
        y_priority = self.priority_encoder.fit_transform(df_prepared["priority"])\
            if set(df_prepared["priority"]) else []
        y_department = self.department_encoder.fit_transform(df_prepared["department"])\
            if set(df_prepared["department"]) else []

        # Build pipelines
        self.priority_model = Pipeline([
            ("tfidf", TfidfVectorizer(ngram_range=(1,2), min_df=1)),
            ("clf", LogisticRegression(max_iter=1000))
        ])
        self.department_model = Pipeline([
            ("tfidf", TfidfVectorizer(ngram_range=(1,2), min_df=1)),
            ("clf", LogisticRegression(max_iter=1000))
        ])

        self.priority_model.fit(X, y_priority)
        self.department_model.fit(X, y_department)
        self.is_trained = True
        return self

    def predict(self, title: str, description: str) -> Tuple[str, str, float, float]:
        if not self.is_trained:
            raise ValueError("Model not trained yet!")
        title_p = self._preprocess_text(title)
        desc_p = self._preprocess_text(description)
        text = (title_p + " " + desc_p).strip()
        if not text:
            raise ValueError("Text required for prediction")

        # Priority prediction
        p_proba = self.priority_model.predict_proba([text])[0]
        p_idx = p_proba.argmax()
        priority = self.priority_encoder.inverse_transform([p_idx])[0]
        p_conf = float(p_proba[p_idx])

        # Department prediction
        d_proba = self.department_model.predict_proba([text])[0]
        d_idx = d_proba.argmax()
        department = self.department_encoder.inverse_transform([d_idx])[0]
        d_conf = float(d_proba[d_idx])

        return priority, department, p_conf, d_conf

    def save_models(self, path: str):
        if not self.is_trained:
            raise ValueError("Cannot save untrained models")
        os.makedirs(path, exist_ok=True)
        joblib.dump({
            "model": self.priority_model,
            "encoder": self.priority_encoder
        }, os.path.join(path, "priority_model.pkl"))
        joblib.dump({
            "model": self.department_model,
            "encoder": self.department_encoder
        }, os.path.join(path, "department_model.pkl"))

    def load_models(self, path: str):
        p_file = os.path.join(path, "priority_model.pkl")
        d_file = os.path.join(path, "department_model.pkl")
        if not (os.path.exists(p_file) and os.path.exists(d_file)):
            raise FileNotFoundError("Model files not found")
        p_bundle = joblib.load(p_file)
        d_bundle = joblib.load(d_file)
        self.priority_model = p_bundle["model"]
        self.priority_encoder = p_bundle["encoder"]
        self.department_model = d_bundle["model"]
        self.department_encoder = d_bundle["encoder"]
        self.is_trained = True
        return self
