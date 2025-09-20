from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
import os
import logging
from app.models.schemas import TicketRequest, TicketResponse, HealthResponse, PriorityLevel, Department
from app.models.classifier import TicketClassifier

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="AI-Powered Customer Support Ticket Classifier",
    description="Automatically classify customer support tickets by priority and department",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global classifier instance
classifier = None


def get_classifier() -> TicketClassifier:
    """Dependency to get the classifier instance"""
    global classifier
    if classifier is None:
        raise HTTPException(status_code=503, detail="Classifier not initialized")
    return classifier


@app.on_event("startup")
async def startup_event():
    """Initialize the classifier on startup"""
    global classifier
    logger.info("Starting up the application...")
    
    classifier = TicketClassifier()
    
    # Try to load pre-trained models
    models_dir = "models"
    try:
        if os.path.exists(models_dir):
            classifier.load_models(models_dir)
            logger.info("Pre-trained models loaded successfully")
        else:
            logger.warning("No pre-trained models found. Please train the model first.")
    except Exception as e:
        logger.error(f"Failed to load models: {e}")
        classifier = None


@app.get("/", response_model=HealthResponse)
async def root():
    """Root endpoint with basic info"""
    return HealthResponse(
        status="healthy",
        message="AI-Powered Customer Support Ticket Classifier API is running"
    )


@app.get("/health", response_model=HealthResponse)
async def health_check(classifier: TicketClassifier = Depends(get_classifier)):
    """Health check endpoint"""
    return HealthResponse(
        status="healthy" if classifier.is_trained else "not_ready",
        message="Classifier is ready" if classifier.is_trained else "Classifier not trained"
    )


@app.post("/classify", response_model=TicketResponse)
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