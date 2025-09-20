import pytest
from fastapi.testclient import TestClient
import sys
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from app.main import app
from app.models.classifier import TicketClassifier
from data.generate_dataset import generate_sample_dataset


@pytest.fixture
def client():
    """Create test client"""
    return TestClient(app)


@pytest.fixture
def trained_app():
    """Create app with trained classifier"""
    # Generate small training dataset
    df = generate_sample_dataset(50)
    
    # Train classifier
    classifier = TicketClassifier()
    classifier.train(df)
    
    # Replace global classifier in app
    from app import main
    main.classifier = classifier
    
    return TestClient(app)


def test_root_endpoint(client):
    """Test root endpoint"""
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "AI-Powered Customer Support Ticket Classifier" in data["message"]


def test_health_endpoint_without_classifier(client):
    """Test health endpoint when classifier is not ready"""
    response = client.get("/health")
    # Should return 503 since classifier is not initialized
    assert response.status_code == 503


def test_classify_endpoint_without_classifier(client):
    """Test classify endpoint when classifier is not ready"""
    ticket_data = {
        "title": "Test ticket",
        "description": "Test description",
        "customer_email": "test@example.com"
    }
    
    response = client.post("/classify", json=ticket_data)
    assert response.status_code == 503


def test_health_endpoint_with_classifier(trained_app):
    """Test health endpoint with trained classifier"""
    response = trained_app.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"


def test_classify_endpoint_success(trained_app):
    """Test successful ticket classification"""
    ticket_data = {
        "title": "Server is down",
        "description": "The main server is not responding to requests",
        "customer_email": "customer@example.com"
    }
    
    response = trained_app.post("/classify", json=ticket_data)
    assert response.status_code == 200
    
    data = response.json()
    assert data["title"] == ticket_data["title"]
    assert data["description"] == ticket_data["description"]
    assert data["customer_email"] == ticket_data["customer_email"]
    assert data["predicted_priority"] in ["Urgent", "High", "Medium", "Low"]
    assert data["predicted_department"] in ["Tech Support", "Billing", "Sales"]
    assert 0 <= data["priority_confidence"] <= 1
    assert 0 <= data["department_confidence"] <= 1


def test_classify_endpoint_minimal_data(trained_app):
    """Test classification with minimal required data"""
    ticket_data = {
        "title": "Help needed",
        "description": "I need assistance"
    }
    
    response = trained_app.post("/classify", json=ticket_data)
    assert response.status_code == 200
    
    data = response.json()
    assert data["customer_email"] is None


def test_classify_endpoint_validation(trained_app):
    """Test request validation"""
    # Missing required fields
    response = trained_app.post("/classify", json={})
    assert response.status_code == 422
    
    # Missing description
    response = trained_app.post("/classify", json={"title": "Test"})
    assert response.status_code == 422
    
    # Missing title
    response = trained_app.post("/classify", json={"description": "Test"})
    assert response.status_code == 422


def test_model_status_endpoint(trained_app):
    """Test model status endpoint"""
    response = trained_app.get("/model/status")
    assert response.status_code == 200
    
    data = response.json()
    assert data["is_trained"] is True
    assert data["models_loaded"] is True


def test_multiple_classifications(trained_app):
    """Test multiple ticket classifications"""
    test_tickets = [
        {
            "title": "Payment failed",
            "description": "My credit card was declined but amount was deducted"
        },
        {
            "title": "Product demo request",
            "description": "I would like to see a demonstration of your product"
        },
        {
            "title": "System outage",
            "description": "All services are down and users cannot access anything"
        }
    ]
    
    for ticket in test_tickets:
        response = trained_app.post("/classify", json=ticket)
        assert response.status_code == 200
        
        data = response.json()
        assert data["predicted_priority"] in ["Urgent", "High", "Medium", "Low"]
        assert data["predicted_department"] in ["Tech Support", "Billing", "Sales"]


def test_cors_headers(trained_app):
    """Test CORS headers are present"""
    response = trained_app.get("/")
    assert "access-control-allow-origin" in response.headers


def test_openapi_docs(client):
    """Test that OpenAPI documentation is available"""
    response = client.get("/docs")
    assert response.status_code == 200
    
    response = client.get("/openapi.json")
    assert response.status_code == 200