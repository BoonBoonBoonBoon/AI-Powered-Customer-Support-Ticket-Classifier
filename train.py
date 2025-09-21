#!/usr/bin/env python3

import sys
import os
import json
import hashlib
import time
import pandas as pd
import argparse
from pathlib import Path
from datetime import datetime, timezone
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np

# Add the project root to the Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from app.models.classifier import TicketClassifier
from app.config import settings
from data.generate_dataset import save_sample_dataset


def _hash_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(8192), b''):
            h.update(chunk)
    return h.hexdigest()


def train_model(data_path: str, models_dir: str = "models"):
    """Train the ticket classifier with the provided dataset and produce metrics/metadata"""
    print(f"Training model with data from: {data_path}")
    print(f"Model version: {settings.MODEL_VERSION}")
    print(f"Validation split: {settings.VALIDATION_SPLIT}")
    print("Setting random seeds for reproducibility...")
    np.random.seed(settings.RANDOM_SEED)
    os.environ["PYTHONHASHSEED"] = str(settings.RANDOM_SEED)
    
    # Check if data file exists
    if not os.path.exists(data_path):
        print(f"Data file not found: {data_path}")
        print("Generating sample dataset...")
        save_sample_dataset(data_path, 1000)
    
    # Load data
    print("Loading dataset...")
    df = pd.read_csv(data_path)
    print(f"Loaded {len(df)} tickets")
    print(f"Columns: {list(df.columns)}")
    
    # Validate required columns
    required_columns = ['title', 'description', 'priority', 'department']
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")
    
    # Show data distribution
    print("\nData distribution:")
    print("Priority distribution:")
    print(df['priority'].value_counts())
    print("\nDepartment distribution:")
    print(df['department'].value_counts())
    
    # Train/validation split
    if settings.ENABLE_EVAL:
        train_df, val_df = train_test_split(
            df,
            test_size=settings.VALIDATION_SPLIT,
            random_state=settings.RANDOM_SEED,
            shuffle=True,
            stratify=df['priority'] if 'priority' in df else None
        )
    else:
        train_df, val_df = df, None

    print(f"Training samples: {len(train_df)} | Validation samples: {len(val_df) if val_df is not None else 0}")

    # Initialize and train classifier
    classifier = TicketClassifier()
    classifier.train(train_df)
    
    # Versioned model directory
    version_dir = os.path.join(models_dir, f"v{settings.MODEL_VERSION}")
    os.makedirs(version_dir, exist_ok=True)
    classifier.save_models(version_dir)

    # Evaluation
    metrics = {}
    if settings.ENABLE_EVAL and val_df is not None and len(val_df) > 0:
        print("\nEvaluating on validation set...")
        y_true_priority = val_df['priority'].tolist()
        y_true_department = val_df['department'].tolist()
        pred_priority = []
        pred_department = []
        for _, row in val_df.iterrows():
            p, d, _, _ = classifier.predict(row['title'], row['description'])
            pred_priority.append(p)
            pred_department.append(d)

        pr_report = classification_report(y_true_priority, pred_priority, output_dict=True)
        dep_report = classification_report(y_true_department, pred_department, output_dict=True)
        pr_cm = confusion_matrix(y_true_priority, pred_priority).tolist()
        dep_cm = confusion_matrix(y_true_department, pred_department).tolist()
        metrics = {
            'priority': {
                'report': pr_report,
                'confusion_matrix': pr_cm
            },
            'department': {
                'report': dep_report,
                'confusion_matrix': dep_cm
            }
        }
        with open(os.path.join(version_dir, 'metrics.json'), 'w', encoding='utf-8') as f:
            json.dump(metrics, f, indent=2)
        print("Validation metrics written to metrics.json")

    # Metadata
    metadata = {
        'model_version': settings.MODEL_VERSION,
        'created_at': datetime.now(timezone.utc).isoformat(),
        'data_path': data_path,
        'data_sha256': _hash_file(data_path),
        'n_samples_total': int(len(df)),
        'n_train_samples': int(len(train_df)),
        'n_val_samples': int(len(val_df) if val_df is not None else 0),
        'priority_distribution': df['priority'].value_counts().to_dict(),
        'department_distribution': df['department'].value_counts().to_dict(),
        'metrics_file': 'metrics.json' if metrics else None
    }
    with open(os.path.join(version_dir, 'model_metadata.json'), 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2)
    print("Model metadata written to model_metadata.json")
    
    print(f"\nTraining completed! Models saved to {version_dir}/")
    
    # Test with sample predictions
    print("\nTesting with sample predictions:")
    test_cases = [
        ("Server is down", "The main server is not responding, all users affected"),
        ("Billing question", "I have a question about my monthly invoice"),
        ("Need product demo", "Can someone show me how the product works?")
    ]
    
    for title, description in test_cases:
        priority, department, p_conf, d_conf = classifier.predict(title, description)
        print(f"Title: {title}")
        print(f"  -> Priority: {priority} (confidence: {p_conf:.3f})")
        print(f"  -> Department: {department} (confidence: {d_conf:.3f})")
        print()


def main():
    parser = argparse.ArgumentParser(description="Train the ticket classifier")
    parser.add_argument(
        "--data", 
        default="data/sample_tickets.csv",
        help="Path to the training data CSV file"
    )
    parser.add_argument(
        "--models-dir",
        default="models",
        help="Directory to save trained models"
    )
    
    args = parser.parse_args()
    
    try:
        train_model(args.data, args.models_dir)
    except Exception as e:
        print(f"Error during training: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()