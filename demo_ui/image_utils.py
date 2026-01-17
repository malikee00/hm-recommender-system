from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ImageConfig:
    images_dir: Path
    placeholder_path: Path


def article_id_to_path(article_id: str, images_dir: Path) -> Path:
    aid = str(article_id)
    aid10 = aid.zfill(10)
    folder = aid10[:3]
    return images_dir / folder / f"{aid10}.jpg"


def get_image_path(article_id: str, cfg: ImageConfig) -> Path:
    p = article_id_to_path(article_id, cfg.images_dir)
    if p.exists():
        return p
    return cfg.placeholder_path
