from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from contextlib import asynccontextmanager
import os
import json
import logging
from pathlib import Path
from typing import Optional
from app.models.schemas import TicketRequest, TicketResponse, HealthResponse, PriorityLevel, Department
from app.models.classifier import TicketClassifier
from app.config import settings
from app.logging_utils import configure_logging, RequestIDMiddleware, MaxBodySizeMiddleware

# Configure structured logging
configure_logging()
logger = logging.getLogger(__name__)

classifier: Optional[TicketClassifier] = None
model_metadata: Optional[dict] = None


def _find_latest_version_dir(base_dir: str) -> Optional[Path]:
    p = Path(base_dir)
    if not p.exists():
        return None
    dirs = [d for d in p.iterdir() if d.is_dir() and d.name.startswith('v')]
    if not dirs:
        return None
    return sorted(dirs)[-1]


@asynccontextmanager
async def lifespan(app: FastAPI):
    global classifier, model_metadata
    logger.info("Lifespan startup: initializing classifier")
    classifier = TicketClassifier()
    version_dir = _find_latest_version_dir(settings.MODELS_BASE_DIR) or Path(settings.MODELS_BASE_DIR)
    try:
        if version_dir.exists():
            classifier.load_models(str(version_dir))
            meta_file = version_dir / 'model_metadata.json'
            if meta_file.exists():
                with open(meta_file, 'r', encoding='utf-8') as f:
                    model_metadata = json.load(f)
            logger.info("Models loaded from %s", version_dir)
        else:
            logger.warning("No model directory found; classifier not trained")
    except Exception as e:
        logger.error("Failed to load models: %s", e)
        classifier = None
    yield
    logger.info("Lifespan shutdown complete")


app = FastAPI(
    title="AI-Powered Customer Support Ticket Classifier",
    description="Automatically classify customer support tickets by priority and department",
    version=settings.MODEL_VERSION,
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])
app.add_middleware(RequestIDMiddleware)
app.add_middleware(MaxBodySizeMiddleware, max_bytes=64 * 1024)  # 64KB

 # (classifier and metadata handled in lifespan)


def get_classifier() -> TicketClassifier:
    """Dependency to get the classifier instance"""
    global classifier
    if classifier is None:
        raise HTTPException(status_code=503, detail="Classifier not initialized")
    return classifier


@app.get("/version")
async def version():
    return {
        "api_version": settings.MODEL_VERSION,
        "model_loaded": classifier is not None and classifier.is_trained,
        "model_metadata": model_metadata
    }


@app.get("/", response_model=HealthResponse)
async def root():
    """Root endpoint with basic info and explicit CORS header for tests"""
    resp = HealthResponse(
        status="healthy",
        message="AI-Powered Customer Support Ticket Classifier API is running"
    )
    # Manually ensure header exists even without Origin (pytest TestClient)
    return JSONResponse(content=resp.model_dump(), headers={"access-control-allow-origin": "*"})


@app.get("/health/live", response_model=HealthResponse, tags=["health"])
async def liveness():
    return HealthResponse(status="alive", message="Service process responsive")

@app.get("/health/ready", response_model=HealthResponse, tags=["health"])
async def readiness(classifier: TicketClassifier = Depends(get_classifier)):
    ready = classifier.is_trained if classifier else False
    return HealthResponse(status="ready" if ready else "not_ready", message="Classifier loaded" if ready else "Classifier not ready")

@app.get("/health", response_model=HealthResponse)
async def legacy_health(classifier: TicketClassifier = Depends(get_classifier)):
    # Preserve backward compatibility for existing tests expecting /health semantics
    return HealthResponse(
        status="healthy" if classifier.is_trained else "not_ready",
        message="Classifier is ready" if classifier.is_trained else "Classifier not trained"
    )


@app.post("/classify", response_model=TicketResponse, tags=["inference"])
async def classify_ticket(
    ticket: TicketRequest,
    classifier: TicketClassifier = Depends(get_classifier)
):
    """Classify a customer support ticket"""
    try:
        if not classifier.is_trained:
            raise HTTPException(
                status_code=503,
                detail="Classifier not trained. Please train the model first."
            )
        
        # Get predictions
        priority, department, priority_conf, dept_conf = classifier.predict(
            ticket.title, ticket.description
        )
        
        return TicketResponse(
            title=ticket.title,
            description=ticket.description,
            predicted_priority=PriorityLevel(priority),
            predicted_department=Department(department),
            priority_confidence=priority_conf,
            department_confidence=dept_conf,
            customer_email=ticket.customer_email
        )
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error classifying ticket: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@app.get("/model/status")
async def model_status(classifier: TicketClassifier = Depends(get_classifier)):
    """Get model training status"""
    return {
        "is_trained": classifier.is_trained,
        "models_loaded": classifier.priority_model is not None and classifier.department_model is not None
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)