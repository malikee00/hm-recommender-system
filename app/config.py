from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional


def _try_load_dotenv(project_root: Path) -> None:
    env_path = project_root / ".env"
    if not env_path.exists():
        return

    try:
        from dotenv import load_dotenv  # type: ignore
    except Exception:
        return

    load_dotenv(dotenv_path=str(env_path), override=False)


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

    registry_base_dir: Path
    run_id: str
    registry_run_dir: Path

    two_tower_model_path: Path
    faiss_index_path: Path
    encoders_path: Path
    item_popularity_path: Path
    metadata_path: Path

    image_base_url: str
    artifacts_url: Optional[str]

    api_cors_origins: List[str]

    @staticmethod
    def from_env() -> "AppConfig":
        project_root = Path(os.getenv("PROJECT_ROOT", Path(__file__).resolve().parents[1])).resolve()

        # load .env if exists
        _try_load_dotenv(project_root)

        feature_store_dir = Path(
            os.getenv("FEATURE_STORE_DIR", str(project_root / "data" / "feature_store"))
        ).resolve()

        registry_base_dir = Path(
            os.getenv("REGISTRY_DIR", str(project_root / "ml" / "registry" / "recommender"))
        ).resolve()

        run_id = os.getenv("RUN_ID", "run_2m_e2_v4_bpr").strip()

        registry_run_dir = Path(
            os.getenv("REGISTRY_RUN_DIR", str(registry_base_dir / run_id))
        ).resolve()

        cors_origins = _parse_csv_env(
            "CORS_ORIGINS",
            "http://localhost:8501,http://127.0.0.1:8501,http://localhost:3000,http://127.0.0.1:3000",
        )

        image_base_url = os.getenv("IMAGE_BASE_URL", "").strip().rstrip("/")
        artifacts_url = os.getenv("ARTIFACTS_URL", "").strip() or None

        cfg = AppConfig(
            service_name=os.getenv("SERVICE_NAME", "H&M Recommender API"),
            service_version=os.getenv("SERVICE_VERSION", "1.0.0"),
            project_root=project_root,
            feature_store_dir=feature_store_dir,
            registry_base_dir=registry_base_dir,
            run_id=run_id,
            registry_run_dir=registry_run_dir,
            two_tower_model_path=registry_run_dir / "two_tower_model.pt",
            faiss_index_path=registry_run_dir / "faiss.index",
            encoders_path=registry_run_dir / "feature_encoders.json",
            item_popularity_path=registry_run_dir / "item_popularity.csv",
            metadata_path=registry_run_dir / "metadata.json",
            image_base_url=image_base_url,
            artifacts_url=artifacts_url,
            api_cors_origins=cors_origins,
        )

        cfg.validate_non_artifact()
        return cfg

    def validate_non_artifact(self) -> None:
        if not self.feature_store_dir.exists():
            raise FileNotFoundError(f"Feature store dir not found: {self.feature_store_dir}")
        return
