from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass(frozen=True)
class ImageConfig:
    images_dir: Path
    placeholder_path: Path


def article_id_to_path(article_id: str, images_dir: Path) -> Path:
    return images_dir / f"{str(article_id)}.jpg"


def get_image_path(article_id: str, cfg: ImageConfig) -> Path:
    p = article_id_to_path(article_id, cfg.images_dir)
    if p.exists():
        return p
    return cfg.placeholder_path
