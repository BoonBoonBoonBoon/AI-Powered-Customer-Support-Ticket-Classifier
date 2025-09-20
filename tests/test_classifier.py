import pytest
import pandas as pd
import os
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from app.models.classifier import TicketClassifier


@pytest.fixture
def sample_data():
    """Create sample training data"""
    data = [
        {"title": "Server down", "description": "The server is not responding", "priority": "Urgent", "department": "Tech Support"},
        {"title": "Billing issue", "description": "Wrong amount charged", "priority": "High", "department": "Billing"},
        {"title": "Product demo", "description": "Need a demonstration", "priority": "Medium", "department": "Sales"},
        {"title": "Password reset", "description": "Can't reset password", "priority": "Medium", "department": "Tech Support"},
        {"title": "Refund request", "description": "Want money back", "priority": "High", "department": "Billing"},
        {"title": "Feature inquiry", "description": "Questions about features", "priority": "Low", "department": "Sales"},
    ]
    return pd.DataFrame(data)


@pytest.fixture
def trained_classifier(sample_data):
    """Create a trained classifier"""
    classifier = TicketClassifier()
    classifier.train(sample_data)
    return classifier


def test_classifier_initialization():
    """Test classifier initialization"""
    classifier = TicketClassifier()
    assert classifier.priority_model is None
    assert classifier.department_model is None
    assert not classifier.is_trained


def test_classifier_training(sample_data):
    """Test classifier training"""
    classifier = TicketClassifier()
    classifier.train(sample_data)
    
    assert classifier.priority_model is not None
    assert classifier.department_model is not None
    assert classifier.is_trained


def test_classifier_prediction(trained_classifier):
    """Test classifier predictions"""
    priority, department, p_conf, d_conf = trained_classifier.predict(
        "Server is down", "Critical system failure"
    )
    
    assert priority in ["Urgent", "High", "Medium", "Low"]
    assert department in ["Tech Support", "Billing", "Sales"]
    assert 0 <= p_conf <= 1
    assert 0 <= d_conf <= 1


def test_classifier_save_load(trained_classifier, tmp_path):
    """Test saving and loading models"""
    models_dir = tmp_path / "test_models"
    
    # Save models
    trained_classifier.save_models(str(models_dir))
    
    # Check files exist
    assert (models_dir / "priority_model.pkl").exists()
    assert (models_dir / "department_model.pkl").exists()
    
    # Load models
    new_classifier = TicketClassifier()
    new_classifier.load_models(str(models_dir))
    
    assert new_classifier.is_trained
    
    # Test predictions work
    priority, department, p_conf, d_conf = new_classifier.predict(
        "Test ticket", "Test description"
    )
    assert priority in ["Urgent", "High", "Medium", "Low"]
    assert department in ["Tech Support", "Billing", "Sales"]


def test_prediction_without_training():
    """Test that prediction fails without training"""
    classifier = TicketClassifier()
    
    with pytest.raises(ValueError, match="Model not trained yet!"):
        classifier.predict("Test", "Test")


def test_text_preprocessing(trained_classifier):
    """Test text preprocessing"""
    # Test with various inputs
    processed = trained_classifier._preprocess_text("  UPPERCASE TEXT  ")
    assert processed == "uppercase text"
    
    processed = trained_classifier._preprocess_text("")
    assert processed == ""
    
    processed = trained_classifier._preprocess_text(None)
    assert processed == ""


def test_predictions_consistency(trained_classifier):
    """Test that predictions are consistent"""
    title = "Server outage"
    description = "All services are down"
    
    # Make same prediction multiple times
    results = []
    for _ in range(3):
        result = trained_classifier.predict(title, description)
        results.append(result)
    
    # All results should be identical
    for result in results[1:]:
        assert result == results[0]