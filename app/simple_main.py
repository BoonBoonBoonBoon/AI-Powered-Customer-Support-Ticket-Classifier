"""
Minimal FastAPI application that works with basic Python libraries
This is a fallback version when external dependencies can't be installed
"""

import json
import os
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

try:
    from fastapi import FastAPI, HTTPException
    from fastapi.middleware.cors import CORSMiddleware
    FASTAPI_AVAILABLE = True
except ImportError:
    FASTAPI_AVAILABLE = False

try:
    from app.models.simple_classifier import TicketClassifier
except ImportError:
    # Import directly
    import sys
    sys.path.append('.')
    from app.models.simple_classifier import SimpleTicketClassifier as TicketClassifier


# Define models without Pydantic if FastAPI is not available
class TicketRequest:
    def __init__(self, title: str, description: str, customer_email: str = None):
        self.title = title
        self.description = description
        self.customer_email = customer_email


class TicketResponse:
    def __init__(self, title: str, description: str, predicted_priority: str, 
                 predicted_department: str, priority_confidence: float, 
                 department_confidence: float, customer_email: str = None):
        self.title = title
        self.description = description
        self.predicted_priority = predicted_priority
        self.predicted_department = predicted_department
        self.priority_confidence = priority_confidence
        self.department_confidence = department_confidence
        self.customer_email = customer_email
    
    def dict(self):
        return {
            "title": self.title,
            "description": self.description,
            "predicted_priority": self.predicted_priority,
            "predicted_department": self.predicted_department,
            "priority_confidence": self.priority_confidence,
            "department_confidence": self.department_confidence,
            "customer_email": self.customer_email
        }


if FASTAPI_AVAILABLE:
    # FastAPI version
    from pydantic import BaseModel
    from typing import Optional
    from enum import Enum

    class PriorityLevel(str, Enum):
        URGENT = "Urgent"
        HIGH = "High"
        MEDIUM = "Medium"
        LOW = "Low"

    class Department(str, Enum):
        TECH_SUPPORT = "Tech Support"
        BILLING = "Billing"
        SALES = "Sales"

    class TicketRequestModel(BaseModel):
        title: str
        description: str
        customer_email: Optional[str] = None

    class TicketResponseModel(BaseModel):
        title: str
        description: str
        predicted_priority: PriorityLevel
        predicted_department: Department
        priority_confidence: float
        department_confidence: float
        customer_email: Optional[str] = None

    class HealthResponse(BaseModel):
        status: str
        message: str

    # Initialize FastAPI app
    app = FastAPI(
        title="AI-Powered Customer Support Ticket Classifier",
        description="Automatically classify customer support tickets by priority and department",
        version="1.0.0"
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Global classifier instance
    classifier = None

    @app.on_event("startup")
    async def startup_event():
        global classifier
        classifier = TicketClassifier()
        # Try to load configuration if available
        try:
            if os.path.exists("models"):
                classifier.load_models("models")
        except Exception as e:
            print(f"Could not load models: {e}")

    @app.get("/", response_model=HealthResponse)
    async def root():
        return HealthResponse(
            status="healthy",
            message="AI-Powered Customer Support Ticket Classifier API is running (Simple Mode)"
        )

    @app.get("/health", response_model=HealthResponse)
    async def health_check():
        global classifier
        if classifier is None:
            raise HTTPException(status_code=503, detail="Classifier not initialized")
        
        return HealthResponse(
            status="healthy" if classifier.is_trained else "not_ready",
            message="Simple rule-based classifier is ready"
        )

    @app.post("/classify", response_model=TicketResponseModel)
    async def classify_ticket(ticket: TicketRequestModel):
        global classifier
        if classifier is None:
            raise HTTPException(status_code=503, detail="Classifier not initialized")
        
        try:
            priority, department, priority_conf, dept_conf = classifier.predict(
                ticket.title, ticket.description
            )
            
            return TicketResponseModel(
                title=ticket.title,
                description=ticket.description,
                predicted_priority=PriorityLevel(priority),
                predicted_department=Department(department),
                priority_confidence=priority_conf,
                department_confidence=dept_conf,
                customer_email=ticket.customer_email
            )
            
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Classification error: {str(e)}")

    @app.get("/model/status")
    async def model_status():
        global classifier
        if classifier is None:
            raise HTTPException(status_code=503, detail="Classifier not initialized")
        
        return {
            "is_trained": classifier.is_trained,
            "model_type": "rule_based",
            "models_loaded": True
        }

else:
    # Simple HTTP server version for when FastAPI is not available
    from http.server import HTTPServer, BaseHTTPRequestHandler
    import urllib.parse

    class SimpleTicketHandler(BaseHTTPRequestHandler):
        classifier = TicketClassifier()
        
        def do_GET(self):
            if self.path == '/':
                self._send_json_response({
                    "status": "healthy",
                    "message": "Simple Ticket Classifier is running"
                })
            elif self.path == '/health':
                self._send_json_response({
                    "status": "healthy",
                    "message": "Simple rule-based classifier is ready"
                })
            elif self.path == '/model/status':
                self._send_json_response({
                    "is_trained": True,
                    "model_type": "rule_based",
                    "models_loaded": True
                })
            else:
                self._send_error(404, "Not Found")
        
        def do_POST(self):
            if self.path == '/classify':
                try:
                    content_length = int(self.headers['Content-Length'])
                    post_data = self.rfile.read(content_length)
                    data = json.loads(post_data.decode('utf-8'))
                    
                    title = data.get('title', '')
                    description = data.get('description', '')
                    customer_email = data.get('customer_email')
                    
                    if not title or not description:
                        self._send_error(400, "Title and description are required")
                        return
                    
                    priority, department, p_conf, d_conf = self.classifier.predict(title, description)
                    
                    response = {
                        "title": title,
                        "description": description,
                        "predicted_priority": priority,
                        "predicted_department": department,
                        "priority_confidence": p_conf,
                        "department_confidence": d_conf,
                        "customer_email": customer_email
                    }
                    
                    self._send_json_response(response)
                    
                except json.JSONDecodeError:
                    self._send_error(400, "Invalid JSON")
                except Exception as e:
                    self._send_error(500, f"Server error: {str(e)}")
            else:
                self._send_error(404, "Not Found")
        
        def _send_json_response(self, data, status_code=200):
            self.send_response(status_code)
            self.send_header('Content-type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            self.wfile.write(json.dumps(data, indent=2).encode('utf-8'))
        
        def _send_error(self, status_code, message):
            self.send_response(status_code)
            self.send_header('Content-type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            error_response = {"error": message}
            self.wfile.write(json.dumps(error_response).encode('utf-8'))

    def run_simple_server(port=8000):
        server_address = ('', port)
        httpd = HTTPServer(server_address, SimpleTicketHandler)
        print(f"Simple Ticket Classifier server running on port {port}")
        print(f"Visit http://localhost:{port}/ to test the API")
        httpd.serve_forever()


if __name__ == "__main__":
    if FASTAPI_AVAILABLE:
        import uvicorn
        uvicorn.run(app, host="0.0.0.0", port=8000)
    else:
        print("FastAPI not available, starting simple HTTP server...")
        run_simple_server(8000)