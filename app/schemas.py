from __future__ import annotations

from typing import List
from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: str = "ok"
    model_loaded: bool
    artifact_version: str | None = None


class RecommendRequest(BaseModel):
    customer_id: str = Field(..., example="3a90a1b9c8b3cc6a...")
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
    recommendations: List[RecommendationItem]
