from __future__ import annotations

import hashlib
import json
import time
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
import requests

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

    def build_image_url(self, article_id: str) -> str:
        if not self.cfg.image_base_url:
            return "https://via.placeholder.com/128?text=No+Image"

        article_id = str(article_id)
        prefix = article_id[:3]
        return f"{self.cfg.image_base_url}/{prefix}/{article_id}.jpg"

    def ensure_artifacts_present(self) -> None:
        required_files = [
            self.cfg.two_tower_model_path,
            self.cfg.faiss_index_path,
            self.cfg.encoders_path,
            self.cfg.item_popularity_path,
            self.cfg.metadata_path,
        ]
        if all(p.exists() for p in required_files):
            return

        if not self.cfg.artifacts_url:
            missing = [str(p) for p in required_files if not p.exists()]
            raise FileNotFoundError(
                "Artifacts missing and ARTIFACTS_URL not set:\n- " + "\n- ".join(missing)
            )

        self.cfg.registry_run_dir.mkdir(parents=True, exist_ok=True)

        zip_cache_dir = self.cfg.registry_base_dir / "_downloads"
        zip_cache_dir.mkdir(parents=True, exist_ok=True)

        url_hash = hashlib.sha1(self.cfg.artifacts_url.encode("utf-8")).hexdigest()[:12]
        zip_path = zip_cache_dir / f"{self.cfg.run_id}_{url_hash}.zip"

        if not zip_path.exists() or zip_path.stat().st_size == 0:
            self._download_zip_with_retry(self.cfg.artifacts_url, zip_path)

        self._extract_zip_into_registry(zip_path, self.cfg.registry_run_dir)

        missing_after = [str(p) for p in required_files if not p.exists()]
        if missing_after:
            raise FileNotFoundError(
                "Artifacts download/extract completed but required files still missing:\n- "
                + "\n- ".join(missing_after)
            )

    def _download_zip_with_retry(self, url: str, zip_path: Path) -> None:
        tmp_path = zip_path.with_suffix(zip_path.suffix + ".part")

        max_attempts = 8
        backoff_cap = 30

        for attempt in range(1, max_attempts + 1):
            try:
                headers = {"User-Agent": "hm-recommender/1.0"}
                resume_from = tmp_path.stat().st_size if tmp_path.exists() else 0
                if resume_from > 0:
                    headers["Range"] = f"bytes={resume_from}-"

                with requests.get(
                    url,
                    stream=True,
                    timeout=(15, 300),
                    headers=headers,
                    allow_redirects=True,
                ) as r:
                    if r.status_code == 416:
                        if tmp_path.exists() and tmp_path.stat().st_size > 0:
                            tmp_path.replace(zip_path)
                            return
                        raise RuntimeError("HTTP 416 (Range Not Satisfiable) but no partial file exists.")

                    r.raise_for_status()

                    mode = "ab" if resume_from > 0 else "wb"
                    with open(tmp_path, mode) as f:
                        for chunk in r.iter_content(chunk_size=1024 * 1024):
                            if chunk:
                                f.write(chunk)

                if tmp_path.exists() and tmp_path.stat().st_size > 0:
                    tmp_path.replace(zip_path)
                    return

                raise RuntimeError("Download completed but zip file is empty.")

            except Exception as e:
                if attempt == max_attempts:
                    raise RuntimeError(f"Failed to download artifacts zip from {url}: {e}")

                time.sleep(min(2 ** attempt, backoff_cap))

    def _extract_zip_into_registry(self, zip_path: Path, target_dir: Path) -> None:
        with zipfile.ZipFile(zip_path, "r") as zf:
            names = [n for n in zf.namelist() if not n.endswith("/")]

            normalized = [n.replace("\\", "/") for n in names]
            run_prefix = None

            candidates = [
                "two_tower_model.pt",
                "faiss.index",
                "feature_encoders.json",
                "item_popularity.csv",
                "metadata.json",
            ]

            for n in normalized:
                for c in candidates:
                    if n.endswith("/" + c):
                        run_prefix = n[: -len(c)]
                        break
                if run_prefix is not None:
                    break

            if run_prefix is None:
                run_prefix = ""

            for member in names:
                m = member.replace("\\", "/")
                if m.endswith("/"):
                    continue

                if run_prefix and m.startswith(run_prefix):
                    rel = m[len(run_prefix) :]
                else:
                    rel = m

                rel = rel.lstrip("/")
                out_path = target_dir / rel
                out_path.parent.mkdir(parents=True, exist_ok=True)

                with zf.open(member) as src, open(out_path, "wb") as dst:
                    dst.write(src.read())

    def load_artifacts(self) -> None:
        if self._loaded is not None:
            return

        try:
            self.ensure_artifacts_present()

            svc = RecommenderService(
                registry_dir=str(self.cfg.registry_run_dir),
                feature_store_dir=str(self.cfg.feature_store_dir),
            )
            svc.load()

            with open(self.cfg.metadata_path, "r", encoding="utf-8") as f:
                meta = json.load(f)

            pop = pd.read_csv(self.cfg.item_popularity_path)
            if "article_id" not in pop.columns:
                raise ValueError(
                    f"item_popularity.csv must have 'article_id' column, got: {list(pop.columns)}"
                )

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
