from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class AppConfig:
    project_root: Path
    feature_store_dir: Path
    registry_run_dir: Path

    two_tower_model_path: Path
    faiss_index_path: Path
    encoders_path: Path
    item_popularity_path: Path
    metadata_path: Path

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

        cfg = AppConfig(
            project_root=project_root,
            feature_store_dir=feature_store_dir,
            registry_run_dir=registry_run_dir,
            two_tower_model_path=registry_run_dir / "two_tower_model.pt",
            faiss_index_path=registry_run_dir / "faiss.index",
            encoders_path=registry_run_dir / "feature_encoders.json",
            item_popularity_path=registry_run_dir / "item_popularity.csv",
            metadata_path=registry_run_dir / "metadata.json",
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
