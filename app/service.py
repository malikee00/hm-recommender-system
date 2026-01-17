from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Dict, Optional, List

from app.config import AppConfig
from ml.recommender.inference import RecommenderService, RecommendResult


@dataclass
class LoadedArtifacts:
    service: RecommenderService
    metadata: Dict[str, Any]


class AppService:
    def __init__(self, cfg: AppConfig):
        self.cfg = cfg
        self._loaded: Optional[LoadedArtifacts] = None

    def load_artifacts(self) -> None:
        if self._loaded is not None:
            return

        svc = RecommenderService(
            registry_dir=str(self.cfg.registry_run_dir),
            feature_store_dir=str(self.cfg.feature_store_dir),
        )
        svc.load()

        with open(self.cfg.metadata_path, "r", encoding="utf-8") as f:
            meta = json.load(f)

        self._loaded = LoadedArtifacts(service=svc, metadata=meta)

    @property
    def is_loaded(self) -> bool:
        return self._loaded is not None

    def artifact_version(self) -> Optional[str]:
        if self._loaded is None:
            return None
        meta = self._loaded.metadata
        return meta.get("model_type") or meta.get("objective")

    def recommend(self, customer_id: str, top_k: int) -> RecommendResult:
        if self._loaded is None:
            raise RuntimeError("Service not loaded")
        return self._loaded.service.recommend(customer_id, top_k)

    def baseline(self, top_k: int) -> RecommendResult:
        if self._loaded is None:
            raise RuntimeError("Service not loaded")
        return self._loaded.service._fallback("baseline", top_k)

    def get_metadata(self) -> Dict[str, Any]:
        if self._loaded is None:
            raise RuntimeError("Service not loaded")
        return dict(self._loaded.metadata)
