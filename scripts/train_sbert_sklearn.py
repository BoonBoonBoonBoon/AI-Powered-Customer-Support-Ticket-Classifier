import argparse
import json
import os
from typing import Dict, Any

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm

# We import lazily to avoid import cost if user only wants help text
from sentence_transformers import SentenceTransformer

SEED = 42


def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {"title", "description", "priority", "department"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in {path}: {missing}")
    return df


def build_texts(df: pd.DataFrame) -> np.ndarray:
    # Concatenate title and description with a separator for better signal
    return (df["title"].astype(str).str.strip() + " \n\n" + df["description"].astype(str).str.strip()).values


def encode_texts(model: SentenceTransformer, texts: np.ndarray, batch_size: int = 64, show_progress: bool = True) -> np.ndarray:
    embeddings = model.encode(list(texts), batch_size=batch_size, show_progress_bar=show_progress, normalize_embeddings=True)
    return np.asarray(embeddings, dtype=np.float32)


def train_and_eval_clf(X_train: np.ndarray, y_train: np.ndarray, X_val: np.ndarray, y_val: np.ndarray) -> Dict[str, Any]:
    # Use a strong but fast linear classifier; class_weight='balanced' helps for skew
    clf = LogisticRegression(max_iter=2000, n_jobs=-1, class_weight="balanced", solver="lbfgs", multi_class="auto")
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_val)
    rep = classification_report(y_val, y_pred, output_dict=True, zero_division=0)
    return {"model": clf, "report": rep}


def main():
    parser = argparse.ArgumentParser(description="Train SBERT + sklearn classifiers for priority and department")
    parser.add_argument("--data", default="data/enriched_customer_tickets.csv", help="Path to dataset CSV")
    parser.add_argument("--encoder", default="sentence-transformers/all-MiniLM-L6-v2", help="SentenceTransformer model name")
    parser.add_argument("--output-version", default=None, help="Version tag for output dir, e.g., s1.0.0")
    parser.add_argument("--output-dir", default=None, help="Explicit output directory. Overrides version if set.")
    parser.add_argument("--val-size", type=float, default=0.2, help="Validation split size")
    parser.add_argument("--batch-size", type=int, default=64, help="Encoding batch size")
    args = parser.parse_args()

    df = load_data(args.data)

    # Prepare labels
    pri_le = LabelEncoder()
    dep_le = LabelEncoder()
    df["priority_id"] = pri_le.fit_transform(df["priority"].astype(str))
    df["department_id"] = dep_le.fit_transform(df["department"].astype(str))

    # Stratify by priority to keep class balance in split
    train_df, val_df = train_test_split(
        df, test_size=args.val_size, random_state=SEED, stratify=df["priority_id"]
    )

    # Texts and labels
    X_train_text = build_texts(train_df)
    X_val_text = build_texts(val_df)
    y_p_train = train_df["priority_id"].values
    y_p_val = val_df["priority_id"].values
    y_d_train = train_df["department_id"].values
    y_d_val = val_df["department_id"].values

    # Encode
    model = SentenceTransformer(args.encoder)
    X_train = encode_texts(model, X_train_text, batch_size=args.batch_size)
    X_val = encode_texts(model, X_val_text, batch_size=args.batch_size)

    # Train classifiers
    pri_out = train_and_eval_clf(X_train, y_p_train, X_val, y_p_val)
    dep_out = train_and_eval_clf(X_train, y_d_train, X_val, y_d_val)

    # Metrics
    metrics = {
        "encoder": args.encoder,
        "priority": {
            "macro_f1": pri_out["report"]["macro avg"]["f1-score"],
            "accuracy": pri_out["report"]["accuracy"],
            "report": pri_out["report"],
        },
        "department": {
            "macro_f1": dep_out["report"]["macro avg"]["f1-score"],
            "accuracy": dep_out["report"]["accuracy"],
            "report": dep_out["report"],
        },
    }

    # Output dir
    if args.output_dir:
        out_dir = args.output_dir
    else:
        ver = args.output_version or "s1.0.0"
        out_dir = os.path.join("models", "sbert", ver)
    os.makedirs(out_dir, exist_ok=True)

    # Save metrics & artifacts
    with open(os.path.join(out_dir, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    # Save label encoders and simple metadata to reconstruct pipeline
    label_maps = {
        "priority_id2label": {int(i): lbl for i, lbl in enumerate(pri_le.classes_.tolist())},
        "priority_label2id": {lbl: int(i) for i, lbl in enumerate(pri_le.classes_.tolist())},
        "department_id2label": {int(i): lbl for i, lbl in enumerate(dep_le.classes_.tolist())},
        "department_label2id": {lbl: int(i) for i, lbl in enumerate(dep_le.classes_.tolist())},
    }
    with open(os.path.join(out_dir, "label_mappings.json"), "w", encoding="utf-8") as f:
        json.dump(label_maps, f, indent=2)

    # Persist models
    import joblib

    joblib.dump(pri_out["model"], os.path.join(out_dir, "priority_clf.joblib"))
    joblib.dump(dep_out["model"], os.path.join(out_dir, "department_clf.joblib"))

    # Record encoder name (we can reload it dynamically at predict time)
    with open(os.path.join(out_dir, "encoder.json"), "w", encoding="utf-8") as f:
        json.dump({"model": args.encoder}, f, indent=2)

    print("Done. Artifacts at:", out_dir)
    print(json.dumps({
        "priority_macro_f1": metrics["priority"]["macro_f1"],
        "priority_accuracy": metrics["priority"]["accuracy"],
        "department_macro_f1": metrics["department"]["macro_f1"],
        "department_accuracy": metrics["department"]["accuracy"],
    }, indent=2))


if __name__ == "__main__":
    main()
