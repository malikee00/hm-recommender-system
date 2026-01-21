from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field

from app.config import (
    APP_NAME,
    APP_VERSION,
    DEFAULT_TOP_K,
    FEATURE_STORE_DIR,
    REGISTRY_RUN_DIR,
    IMAGES_DIR,
)
from app.service import LocalHMService

app = FastAPI(title=APP_NAME, version=APP_VERSION)

templates = Jinja2Templates(directory=str(Path(__file__).resolve().parents[1] / "templates"))

svc = LocalHMService(feature_store_dir=FEATURE_STORE_DIR, registry_run_dir=REGISTRY_RUN_DIR)

app.mount("/img", StaticFiles(directory=str(IMAGES_DIR)), name="img")


class RecommendReq(BaseModel):
    customer_id: str = Field(..., min_length=5)
    top_k: int = Field(DEFAULT_TOP_K, ge=1, le=50)


@app.on_event("startup")
def startup() -> None:
    try:
        svc.load()
    except Exception as e:
        raise RuntimeError(f"Failed to load service artifacts: {e}") from e


@app.get("/health")
def health() -> dict:
    return {
        "status": "ok",
        "app_name": APP_NAME,
        "app_version": APP_VERSION,
        "default_top_k": DEFAULT_TOP_K,
        "feature_store_dir": str(FEATURE_STORE_DIR),
        "registry_run_dir": str(REGISTRY_RUN_DIR),
        "images_dir": str(IMAGES_DIR),
    }


@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request, "default_top_k": DEFAULT_TOP_K})


@app.get("/download/sample_customers.xlsx")
def download_sample_customers():
    sample_path = Path(__file__).resolve().parents[1] / "data" / "reference" / "sample_customers.xlsx"
    if not sample_path.exists():
        raise HTTPException(
            status_code=404,
            detail="sample_customers.xlsx not found. Jalankan generator: python scripts/make_sample_customers.py",
        )
    return FileResponse(
        str(sample_path),
        media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        filename="sample_customers.xlsx",
    )


@app.get("/api/user/{customer_id}")
def api_user(customer_id: str):
    try:
        return svc.user_profile_and_history(customer_id=customer_id, last_n=5)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/api/recommend")
def api_recommend(req: RecommendReq):
    try:
        return svc.recommend(customer_id=req.customer_id, top_k=req.top_k)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
