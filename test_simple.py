#!/usr/bin/env python3

"""
Simple test script that doesn't require external dependencies
"""

import sys
import os
sys.path.insert(0, '.')

from app.models.simple_classifier import SimpleTicketClassifier


def test_classifier_basic():
    """Test basic classifier functionality"""
    classifier = SimpleTicketClassifier()
    
    # Test initialization
    assert classifier.is_trained == True
    print("✓ Classifier initialization test passed")
    
    # Test basic prediction
    priority, department, p_conf, d_conf = classifier.predict(
        "Server down", "The server is not responding"
    )
    
    assert priority in ["Urgent", "High", "Medium", "Low"]
    assert department in ["Tech Support", "Billing", "Sales"]
    assert 0 <= p_conf <= 1
    assert 0 <= d_conf <= 1
    print("✓ Basic prediction test passed")


def test_priority_classification():
    """Test priority classification"""
    classifier = SimpleTicketClassifier()
    
    # Test urgent priority
    priority, _, _, _ = classifier.predict(
        "Server outage", "Critical server down, all users affected"
    )
    assert priority == "Urgent"
    print("✓ Urgent priority classification test passed")
    
    # Test high priority
    priority, _, _, _ = classifier.predict(
        "Payment failed", "Credit card payment declined"
    )
    assert priority == "High"
    print("✓ High priority classification test passed")
    
    # Test low priority
    priority, _, _, _ = classifier.predict(
        "General question", "I have a question about documentation"
    )
    assert priority == "Low"
    print("✓ Low priority classification test passed")


def test_department_classification():
    """Test department classification"""
    classifier = SimpleTicketClassifier()
    
    # Test tech support
    _, department, _, _ = classifier.predict(
        "Login issue", "Cannot login to the website"
    )
    assert department == "Tech Support"
    print("✓ Tech Support classification test passed")
    
    # Test billing
    _, department, _, _ = classifier.predict(
        "Invoice question", "I have a question about my bill"
    )
    assert department == "Billing"
    print("✓ Billing classification test passed")
    
    # Test sales
    _, department, _, _ = classifier.predict(
        "Demo request", "I want to see a product demonstration"
    )
    assert department == "Sales"
    print("✓ Sales classification test passed")


def test_edge_cases():
    """Test edge cases"""
    classifier = SimpleTicketClassifier()
    
    # Test empty input
    priority, department, p_conf, d_conf = classifier.predict("", "")
    assert priority in ["Urgent", "High", "Medium", "Low"]
    assert department in ["Tech Support", "Billing", "Sales"]
    print("✓ Empty input test passed")
    
    # Test None input (converted to empty string)
    processed = classifier._preprocess_text(None)
    assert processed == ""
    print("✓ None input preprocessing test passed")


def test_save_load():
    """Test save and load functionality"""
    classifier = SimpleTicketClassifier()
    
    # Test save
    test_dir = "/tmp/test_models"
    classifier.save_models(test_dir)
    assert os.path.exists(f"{test_dir}/classifier_config.json")
    print("✓ Model saving test passed")
    
    # Test load
    new_classifier = SimpleTicketClassifier()
    new_classifier.load_models(test_dir)
    assert new_classifier.is_trained
    print("✓ Model loading test passed")


def run_all_tests():
    """Run all tests"""
    print("Running Simple Ticket Classifier Tests")
    print("=" * 50)
    
    try:
        test_classifier_basic()
        test_priority_classification()
        test_department_classification()
        test_edge_cases()
        test_save_load()
        
        print("\n" + "=" * 50)
        print("🎉 All tests passed!")
        return True
        
    except AssertionError as e:
        print(f"\n❌ Test failed: {e}")
        return False
    except Exception as e:
        print(f"\n💥 Unexpected error: {e}")
        return False


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)