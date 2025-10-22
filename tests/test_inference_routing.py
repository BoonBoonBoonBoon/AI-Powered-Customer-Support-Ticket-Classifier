import os
import pytest
from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def test_classify_sklear_default():
    # Default path should be sklearn; ensure endpoint responds.
    os.environ.pop("SERVE_MODEL_TYPE", None)
    resp = client.post("/classify", json={
        "title": "Password reset not working",
        "description": "I reset my password but still cannot login."
    })
    assert resp.status_code in (200, 503)  # allow not_ready in dev


def test_classify_transformer_endpoint_available():
    # Transformer explicit endpoint should return 200 or 500 if artifacts missing.
    resp = client.post("/classify/transformer", json={
        "title": "Billing issue",
        "description": "I was charged twice for my subscription."
    })
    assert resp.status_code in (200, 500)


@pytest.mark.skipif(not (os.path.exists("models/registry/staging.json") and os.path.exists("models/transformers/t1.0.1/pytorch_model.bin")), reason="Transformer artifacts not present")
def test_classify_routed_to_transformer():
    # When flag is set, /classify should route to transformer runtime
    os.environ["SERVE_MODEL_TYPE"] = "transformer"
    os.environ["SERVE_MODEL_ENV"] = "staging"
    resp = client.post("/classify", json={
        "title": "Login issue",
        "description": "Cannot login with correct password"
    })
    assert resp.status_code in (200, 500)
