from __future__ import annotations

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: str = "ok"
    service: str
    version: str
    model_loaded: bool
    ready: bool
    artifact_version: Optional[str] = None
    detail: Optional[str] = None


class RecommendRequest(BaseModel):
    customer_id: str = Field(..., examples=["3a90a1b9c8b3cc6a73ed007b774c868113af5e4b9ff4ce214f673a8102a2da44"])
    top_k: int = Field(10, ge=1, le=50)


class RecommendationItem(BaseModel):
    article_id: str
    score: float


class RecommendResponse(BaseModel):
    customer_id: str
    top_k: int
    is_fallback: bool
    recommendations: List[RecommendationItem]


class BaselineRequest(BaseModel):
    top_k: int = Field(10, ge=1, le=50)


class BaselineResponse(BaseModel):
    top_k: int
    is_fallback: bool = True
    recommendations: List[RecommendationItem]


class MetadataResponse(BaseModel):
    metadata: Dict[str, Any]
