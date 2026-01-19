from __future__ import annotations

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from app.config import AppConfig
from app.schemas import (
    BaselineRequest,
    BaselineResponse,
    HealthResponse,
    MetadataResponse,
    RecommendRequest,
    RecommendResponse,
    RecommendationItem,
)
from app.service import AppService

cfg = AppConfig.from_env()
app_service = AppService(cfg)

app = FastAPI(title=cfg.service_name, version=cfg.service_version)

app.add_middleware(
    CORSMiddleware,
    allow_origins=cfg.api_cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def startup_event():
    try:
        app_service.load_artifacts()
    except Exception:
        # error sudah disimpan ke app_service.last_error
        return


@app.get("/health", response_model=HealthResponse)
def health():
    loaded = app_service.is_loaded
    detail = None if loaded else (app_service.last_error or "Service not ready")

    return HealthResponse(
        status="ok",
        service=cfg.service_name,
        version=cfg.service_version,
        model_loaded=loaded,
        ready=loaded,
        artifact_version=app_service.artifact_version(),
        detail=detail,
    )


@app.get("/metadata", response_model=MetadataResponse)
def metadata():
    if not app_service.is_loaded:
        raise HTTPException(status_code=503, detail="Service not ready")
    return MetadataResponse(metadata=app_service.get_metadata())


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
            RecommendationItem(
                article_id=r.article_id,
                score=r.score,
                image_url=app_service.build_image_url(r.article_id),
            )
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
        is_fallback=True,
        recommendations=[
            RecommendationItem(
                article_id=r.article_id,
                score=r.score,
                image_url=app_service.build_image_url(r.article_id),
            )
            for r in result.recommendations
        ],
    )
