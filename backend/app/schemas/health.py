"""Health response schemas."""

from pydantic import BaseModel


class HealthResponse(BaseModel):
    """API health check response."""

    status: str
