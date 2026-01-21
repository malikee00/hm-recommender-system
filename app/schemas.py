from __future__ import annotations

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: str = "ok"
    app_name: str
    app_version: str
    default_top_k: int
    feature_store_dir: str
    registry_run_dir: str
    images_dir: str
    detail: Optional[str] = None


class RecommendRequest(BaseModel):
    customer_id: str = Field(
        ...,
        min_length=5,
        examples=["3a90a1b9c8b3cc6a73ed007b774c868113af5e4b9ff4ce214f673a8102a2da44"],
    )
    top_k: int = Field(10, ge=1, le=50)


class BaselineRequest(BaseModel):
    top_k: int = Field(10, ge=1, le=50)


class RecommendationItem(BaseModel):
    article_id: str
    score: float = 0.0
    image_url: str
    meta: Dict[str, Any] = Field(default_factory=dict)


class RecommendResponse(BaseModel):
    customer_id: str
    top_k: int
    is_fallback: bool
    recommendations: List[RecommendationItem]


class BaselineResponse(BaseModel):
    top_k: int
    source: str = "popular"
    recommendations: List[RecommendationItem]


class UserProfile(BaseModel):
    customer_id: str
    age: Optional[int] = None
    club_member_status: Optional[str] = None
    fashion_news_frequency: Optional[str] = None
    active: Optional[bool] = None


class UserSummary(BaseModel):
    total_purchases: Optional[int] = None
    last_purchase_date: Optional[str] = None
    avg_price: Optional[float] = None
    top_product_group_name: Optional[str] = None


class UserLastTransactionItem(BaseModel):
    t_dat: Optional[str] = None
    article_id: str
    price: Optional[float] = None
    image_url: str
    product_type_name: Optional[str] = None
    product_group_name: Optional[str] = None
    department_name: Optional[str] = None
    colour_group_name: Optional[str] = None
    section_name: Optional[str] = None
    garment_group_name: Optional[str] = None


class UserResponse(BaseModel):
    profile: UserProfile
    summary: Dict[str, Any] = Field(default_factory=dict)
    last_transactions: List[UserLastTransactionItem] = Field(default_factory=list)


class MetadataResponse(BaseModel):
    metadata: Dict[str, Any]
