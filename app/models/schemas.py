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

class TicketRequest(BaseModel):
    title: str
    description: str
    customer_email: Optional[str] = None

class TicketResponse(BaseModel):
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

class ErrorResponse(BaseModel):
    detail: str
    code: str | None = None
    request_id: str | None = None
