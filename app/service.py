from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Dict, Optional, List

import pandas as pd

from app.config import AppConfig
from ml.recommender.inference import RecommenderService, RecommendResult, Recommendation


@dataclass
class LoadedArtifacts:
    service: RecommenderService
    metadata: Dict[str, Any]
    popularity_df: pd.DataFrame


class AppService:
    def __init__(self, cfg: AppConfig):
        self.cfg = cfg
        self._loaded: Optional[LoadedArtifacts] = None
        self._last_error: Optional[str] = None

    def load_artifacts(self) -> None:
        if self._loaded is not None:
            return

        try:
            svc = RecommenderService(
                registry_dir=str(self.cfg.registry_run_dir),
                feature_store_dir=str(self.cfg.feature_store_dir),
            )
            svc.load()

            with open(self.cfg.metadata_path, "r", encoding="utf-8") as f:
                meta = json.load(f)

            pop = pd.read_csv(self.cfg.item_popularity_path)
            if "article_id" not in pop.columns:
                raise ValueError(f"item_popularity.csv must have 'article_id' column, got: {list(pop.columns)}")

            score_col = None
            for c in ["popularity", "cnt", "count", "score"]:
                if c in pop.columns:
                    score_col = c
                    break
            if score_col is None:
                pop = pop.copy()
                pop["popularity"] = list(range(len(pop), 0, -1))
                score_col = "popularity"

            pop = pop[["article_id", score_col]].rename(columns={score_col: "score"})
            pop["article_id"] = pop["article_id"].astype(str)
            pop["score"] = pop["score"].astype(float)

            self._loaded = LoadedArtifacts(service=svc, metadata=meta, popularity_df=pop)
            self._last_error = None

        except Exception as e:
            self._loaded = None
            self._last_error = str(e)
            raise

    @property
    def is_loaded(self) -> bool:
        return self._loaded is not None

    @property
    def last_error(self) -> Optional[str]:
        return self._last_error

    def artifact_version(self) -> Optional[str]:
        if self._loaded is None:
            return None
        meta = self._loaded.metadata
        return meta.get("run_id") or meta.get("model_type") or meta.get("objective")

    def recommend(self, customer_id: str, top_k: int) -> RecommendResult:
        if self._loaded is None:
            raise RuntimeError("Service not loaded")
        return self._loaded.service.recommend(customer_id, top_k)

    def baseline(self, top_k: int) -> RecommendResult:
        if self._loaded is None:
            raise RuntimeError("Service not loaded")

        df = self._loaded.popularity_df.head(top_k)
        recs: List[Recommendation] = [
            Recommendation(article_id=row["article_id"], score=float(row["score"]))
            for _, row in df.iterrows()
        ]

        return RecommendResult(
            customer_id="baseline",
            top_k=top_k,
            is_fallback=True,
            recommendations=recs,
        )

    def get_metadata(self) -> Dict[str, Any]:
        if self._loaded is None:
            raise RuntimeError("Service not loaded")
        return dict(self._loaded.metadata)
