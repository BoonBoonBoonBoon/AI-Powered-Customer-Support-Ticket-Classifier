#!/usr/bin/env python3

import sys
import os
import pandas as pd
import argparse
from pathlib import Path

# Add the project root to the Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from app.models.classifier import TicketClassifier
from data.generate_dataset import save_sample_dataset


def train_model(data_path: str, models_dir: str = "models"):
    """Train the ticket classifier with the provided dataset"""
    print(f"Training model with data from: {data_path}")
    
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
    
    # Initialize and train classifier
    classifier = TicketClassifier()
    classifier.train(df)
    
    # Save trained models
    os.makedirs(models_dir, exist_ok=True)
    classifier.save_models(models_dir)
    
    print(f"\nTraining completed! Models saved to {models_dir}/")
    
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