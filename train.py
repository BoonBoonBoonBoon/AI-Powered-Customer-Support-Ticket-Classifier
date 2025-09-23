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


def train_model(
    data_path: str,
    models_dir: str = "models",
    holdout_frac: float = 0.0,
    class_weight: str | None = None,
    augment_length: bool = False,
    min_class_samples: int = 1,
    department_exclude_regex: list[str] | None = None,
):
    """Train the ticket classifier with the provided dataset and produce metrics/metadata.

    Parameters:
        data_path: Path to CSV containing columns title, description, priority, department
        models_dir: Base directory for versioned artifacts
        holdout_frac: Fraction of entire dataset to reserve as final holdout (0 disables)
        class_weight: 'balanced' or None passed to sklearn LogisticRegression
        augment_length: If True, append length bucket synthetic tokens
        min_class_samples: Minimum samples required per class (priority & department)
    """
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
    
    # Optional holdout split (stratified by priority where possible)
    holdout_df = None
    if holdout_frac and holdout_frac > 0:
        if holdout_frac >= 0.5:
            raise ValueError("holdout_frac should be < 0.5")
        holdout_df, df_work = train_test_split(
            df,
            test_size=1 - holdout_frac,
            random_state=settings.RANDOM_SEED,
            shuffle=True,
            stratify=df['priority'] if 'priority' in df else None
        )
        print(f"Holdout reserved: {len(holdout_df)} samples")
    else:
        df_work = df

    # Train/validation split
    if settings.ENABLE_EVAL:
        train_df, val_df = train_test_split(
            df_work,
            test_size=settings.VALIDATION_SPLIT,
            random_state=settings.RANDOM_SEED,
            shuffle=True,
            stratify=df_work['priority'] if 'priority' in df_work else None
        )
    else:
        train_df, val_df = df_work, None

    # Class minimum enforcement (simple warning / filtering if needed)
    for col in ["priority", "department"]:
        counts = train_df[col].value_counts()
        too_small = counts[counts < min_class_samples]
        if len(too_small) > 0:
            print(f"WARNING: Classes below min samples in training set for {col}: {too_small.to_dict()}")

    print(f"Training samples: {len(train_df)} | Validation samples: {len(val_df) if val_df is not None else 0}")

    # Initialize and train classifier
    classifier = TicketClassifier()
    classifier.train(
        train_df,
        class_weight=class_weight,
        augment_length_buckets=augment_length,
        department_exclude_regexes=department_exclude_regex,
    )
    
    # Versioned model directory
    version_dir = os.path.join(models_dir, f"v{settings.MODEL_VERSION}")
    os.makedirs(version_dir, exist_ok=True)
    classifier.save_models(version_dir)
    print(f"Saved model artifacts (vectorizers, models, encoders) to {version_dir}")

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
                'confusion_matrix': pr_cm,
                'summary': {
                    'macro_f1': pr_report.get('macro avg', {}).get('f1-score'),
                    'weighted_f1': pr_report.get('weighted avg', {}).get('f1-score'),
                    'accuracy': pr_report.get('accuracy')
                }
            },
            'department': {
                'report': dep_report,
                'confusion_matrix': dep_cm,
                'summary': {
                    'macro_f1': dep_report.get('macro avg', {}).get('f1-score'),
                    'weighted_f1': dep_report.get('weighted avg', {}).get('f1-score'),
                    'accuracy': dep_report.get('accuracy')
                }
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
        'n_holdout_samples': int(len(holdout_df) if holdout_df is not None else 0),
        'priority_distribution': df['priority'].value_counts().to_dict(),
        'department_distribution': df['department'].value_counts().to_dict(),
        'metrics_file': 'metrics.json' if metrics else None,
        'priority_macro_f1': metrics.get('priority', {}).get('summary', {}).get('macro_f1'),
        'department_macro_f1': metrics.get('department', {}).get('summary', {}).get('macro_f1'),
        'class_weight': class_weight,
        'augment_length_buckets': augment_length,
        'department_exclude_regex': department_exclude_regex,
    }
    with open(os.path.join(version_dir, 'model_metadata.json'), 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2)
    print("Model metadata written to model_metadata.json")
    
    # Holdout evaluation (never used for training or validation tuning)
    if holdout_df is not None and len(holdout_df) > 0:
        print("\nEvaluating on holdout set...")
        ho_priority_true = holdout_df['priority'].tolist()
        ho_department_true = holdout_df['department'].tolist()
        ho_pr_pred, ho_dep_pred = [], []
        for _, row in holdout_df.iterrows():
            p, d, _, _ = classifier.predict(row['title'], row['description'])
            ho_pr_pred.append(p)
            ho_dep_pred.append(d)
        from sklearn.metrics import classification_report as _cr
        ho_pr_report = _cr(ho_priority_true, ho_pr_pred, output_dict=True)
        ho_dep_report = _cr(ho_department_true, ho_dep_pred, output_dict=True)
        holdout_metrics = {
            'priority': {
                'macro_f1': ho_pr_report.get('macro avg', {}).get('f1-score'),
                'accuracy': ho_pr_report.get('accuracy')
            },
            'department': {
                'macro_f1': ho_dep_report.get('macro avg', {}).get('f1-score'),
                'accuracy': ho_dep_report.get('accuracy')
            },
            'n_holdout_samples': len(holdout_df)
        }
        with open(os.path.join(version_dir, 'holdout_metrics.json'), 'w', encoding='utf-8') as f:
            json.dump(holdout_metrics, f, indent=2)
        print("Holdout metrics written to holdout_metrics.json")
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
    parser = argparse.ArgumentParser(description="Train the ticket classifier (advanced)")
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
    parser.add_argument(
        "--holdout-frac",
        type=float,
        default=0.0,
        help="Fraction of full dataset to reserve as final holdout (0 disables)"
    )
    parser.add_argument(
        "--class-weight",
        choices=["balanced", "none"],
        default="none",
        help="Apply class_weight='balanced' to logistic regression models"
    )
    parser.add_argument(
        "--augment-length",
        action="store_true",
        help="Append length bucket tokens to text inputs"
    )
    parser.add_argument(
        "--min-class-samples",
        type=int,
        default=1,
        help="Warn if any class has fewer than this many training samples"
    )
    parser.add_argument(
        "--department-exclude-regex",
        action="append",
        default=[],
        help="Regex pattern(s) to remove from description when training/predicting department model (can be repeated)"
    )
    
    args = parser.parse_args()
    
    try:
        train_model(
            args.data,
            args.models_dir,
            holdout_frac=args.holdout_frac,
            class_weight=(None if args.class_weight == 'none' else 'balanced'),
            augment_length=args.augment_length,
            min_class_samples=args.min_class_samples,
            department_exclude_regex=args.department_exclude_regex or None,
        )
    except Exception as e:
        print(f"Error during training: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()