from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import List


def _parse_csv_env(name: str, default: str) -> List[str]:
    raw = os.getenv(name, default).strip()
    if not raw:
        return []
    return [x.strip() for x in raw.split(",") if x.strip()]


@dataclass(frozen=True)
class AppConfig:
    service_name: str
    service_version: str

    project_root: Path
    feature_store_dir: Path
    registry_run_dir: Path

    two_tower_model_path: Path
    faiss_index_path: Path
    encoders_path: Path
    item_popularity_path: Path
    metadata_path: Path

    # CORS
    api_cors_origins: List[str]

    @staticmethod
    def from_env() -> "AppConfig":
        project_root = Path(os.getenv("PROJECT_ROOT", Path(__file__).resolve().parents[1])).resolve()

        feature_store_dir = Path(
            os.getenv("FEATURE_STORE_DIR", str(project_root / "data" / "feature_store"))
        ).resolve()

        registry_base_dir = Path(
            os.getenv("REGISTRY_DIR", str(project_root / "ml" / "registry" / "recommender"))
        ).resolve()

        run_id = os.getenv("RUN_ID", "run_2m_e2_v4_bpr")

        registry_run_dir = Path(
            os.getenv("REGISTRY_RUN_DIR", str(registry_base_dir / run_id))
        ).resolve()

        cors_origins = _parse_csv_env(
            "CORS_ORIGINS",
            "http://localhost:3000,http://127.0.0.1:3000",
        )

        cfg = AppConfig(
            service_name=os.getenv("SERVICE_NAME", "H&M Recommender API"),
            service_version=os.getenv("SERVICE_VERSION", "1.0.0"),
            project_root=project_root,
            feature_store_dir=feature_store_dir,
            registry_run_dir=registry_run_dir,
            two_tower_model_path=registry_run_dir / "two_tower_model.pt",
            faiss_index_path=registry_run_dir / "faiss.index",
            encoders_path=registry_run_dir / "feature_encoders.json",
            item_popularity_path=registry_run_dir / "item_popularity.csv",
            metadata_path=registry_run_dir / "metadata.json",
            api_cors_origins=cors_origins,
        )

        cfg.validate()
        return cfg

    def validate(self) -> None:
        required_files = [
            self.two_tower_model_path,
            self.faiss_index_path,
            self.encoders_path,
            self.item_popularity_path,
            self.metadata_path,
        ]
        missing = [str(p) for p in required_files if not p.exists()]
        if missing:
            raise FileNotFoundError(
                "Missing required artifacts:\n- " + "\n- ".join(missing)
            )

        if not self.feature_store_dir.exists():
            raise FileNotFoundError(f"Feature store dir not found: {self.feature_store_dir}")
