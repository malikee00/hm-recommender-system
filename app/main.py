from __future__ import annotations

from fastapi import FastAPI, HTTPException

from app.config import AppConfig
from app.schemas import (
    HealthResponse,
    RecommendRequest,
    RecommendResponse,
    BaselineRequest,
    BaselineResponse,
    RecommendationItem,
)
from app.service import AppService

app = FastAPI(title="H&M Recommender API", version="1.0.0")

cfg = AppConfig.from_env()
app_service = AppService(cfg)


@app.on_event("startup")
def startup_event():
    try:
        app_service.load_artifacts()
    except Exception as e:
        raise RuntimeError(f"Startup failed: {e}")


@app.get("/health", response_model=HealthResponse)
def health():
    return HealthResponse(
        status="ok",
        model_loaded=app_service.is_loaded,
        artifact_version=app_service.artifact_version(),
    )


@app.post("/recommend", response_model=RecommendResponse)
def recommend(req: RecommendRequest):
    if not app_service.is_loaded:
        raise HTTPException(status_code=503, detail="Service not ready")

    result = app_service.recommend(req.customer_id, req.top_k)

    return RecommendResponse(
        customer_id=result.customer_id,
        top_k=result.top_k,
        is_fallback=result.is_fallback,
        recommendations=[
            RecommendationItem(article_id=r.article_id, score=r.score)
            for r in result.recommendations
        ],
    )


@app.post("/baseline", response_model=BaselineResponse)
def baseline(req: BaselineRequest):
    if not app_service.is_loaded:
        raise HTTPException(status_code=503, detail="Service not ready")

    result = app_service.baseline(req.top_k)

    return BaselineResponse(
        top_k=req.top_k,
        recommendations=[
            RecommendationItem(article_id=r.article_id, score=r.score)
            for r in result.recommendations
        ],
    )
