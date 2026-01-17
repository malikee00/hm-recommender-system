from __future__ import annotations

import os
from dataclasses import dataclass
from typing import List, Optional, Dict, Any

import numpy as np
import pandas as pd
import torch

try:
    import faiss
except Exception:
    faiss = None

from features.preprocess import load_encoders, transform_user_features


def load_user_history_agg(feature_store_dir: str) -> Optional[pd.DataFrame]:
    path = os.path.join(feature_store_dir, "user_history_agg.parquet")
    if not os.path.exists(path):
        return None

    df = pd.read_parquet(path)
    if "customer_id" not in df.columns or "top_product_group_name" not in df.columns:
        return None

    out = df[["customer_id", "top_product_group_name"]].copy()
    out["customer_id"] = out["customer_id"].astype(str)
    out["top_product_group_name"] = out["top_product_group_name"].astype(str)
    return out


@dataclass
class Recommendation:
    article_id: str
    score: float


@dataclass
class RecommendResult:
    customer_id: str
    top_k: int
    is_fallback: bool
    recommendations: List[Recommendation]


class RecommenderService:
    def __init__(self, registry_dir: str, feature_store_dir: str, debug: bool = False):
        self.registry_dir = registry_dir
        self.feature_store_dir = feature_store_dir
        self.debug = debug

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._loaded = False

        self.enc = None
        self.inv_item: List[Optional[str]] = []
        self.index = None
        self.pop_list: List[str] = []
        self.user_feat_t: Optional[torch.Tensor] = None
        self.model = None
        self._last_debug: Dict[str, Any] = {}

    def load(self):
        if self._loaded:
            return

        if faiss is None:
            raise RuntimeError("faiss is not installed")

        enc_path = os.path.join(self.registry_dir, "feature_encoders.json")
        idx_path = os.path.join(self.registry_dir, "faiss.index")
        pop_path = os.path.join(self.registry_dir, "item_popularity.csv")
        ckpt_path = os.path.join(self.registry_dir, "two_tower_model.pt")

        for p in (enc_path, idx_path, pop_path, ckpt_path):
            if not os.path.exists(p):
                raise FileNotFoundError(f"Missing artifact: {p}")

        self.enc = load_encoders(enc_path)

        self.inv_item = [None] * len(self.enc.item_id_map)
        for item_id, idx in self.enc.item_id_map.items():
            self.inv_item[idx] = str(item_id)

        self.index = faiss.read_index(idx_path)

        pop = pd.read_csv(pop_path)
        if "article_id" not in pop.columns:
            raise ValueError("item_popularity.csv must contain 'article_id'")
        pop["article_id"] = pop["article_id"].astype(str)
        self.pop_list = pop["article_id"].tolist()

        user_feat_path = os.path.join(self.feature_store_dir, "user_features.parquet")
        if not os.path.exists(user_feat_path):
            raise FileNotFoundError(f"Missing user_features.parquet: {user_feat_path}")

        user_df = pd.read_parquet(user_feat_path)
        user_hist = load_user_history_agg(self.feature_store_dir)

        user_feat_mat, _ = transform_user_features(
            user_df, self.enc, user_history_agg_df=user_hist
        )
        self.user_feat_t = torch.as_tensor(user_feat_mat, dtype=torch.long, device="cpu")

        ckpt = torch.load(ckpt_path, map_location="cpu")
        from .train import TwoTower

        self.model = TwoTower(
            user_cat_sizes=ckpt["user_cat_sizes"],
            item_cat_sizes=ckpt["item_cat_sizes"],
            age_num_buckets=ckpt["age_num_buckets"],
            embedding_dim=ckpt["embedding_dim"],
            user_cat_cols=ckpt["user_cat_cols"],
            item_cat_cols=ckpt["item_cat_cols"],
        )
        self.model.load_state_dict(ckpt["state_dict"])
        self.model = self.model.to(self.device)
        self.model.eval()

        self._loaded = True

    def _fallback(self, customer_id: str, top_k: int) -> RecommendResult:
        recs = [
            Recommendation(article_id=a, score=1.0 / (i + 1))
            for i, a in enumerate(self.pop_list[:top_k])
        ]
        return RecommendResult(
            customer_id=str(customer_id),
            top_k=top_k,
            is_fallback=True,
            recommendations=recs,
        )

    def recommend(self, customer_id: str, top_k: int = 10) -> RecommendResult:
        self.load()
        customer_id = str(customer_id)
        self._last_debug = {}

        if customer_id not in self.enc.user_id_map:
            return self._fallback(customer_id, top_k)

        u_idx = self.enc.user_id_map[customer_id]
        if self.user_feat_t is None or u_idx < 0 or u_idx >= self.user_feat_t.shape[0]:
            return self._fallback(customer_id, top_k)

        feat = self.user_feat_t[u_idx : u_idx + 1].to(self.device)

        with torch.no_grad():
            u_emb = (
                self.model.user_forward(feat)
                .detach()
                .cpu()
                .numpy()
                .astype(np.float32)
            )

        k_search = max(top_k, 50)
        D, I = self.index.search(u_emb, k_search)

        if self.debug and len(D) > 0:
            self._last_debug["faiss_scores_head"] = [float(x) for x in D[0][:10]]

        recs: List[Recommendation] = []

        for rank, (score, internal_idx) in enumerate(
            zip(D[0].tolist(), I[0].tolist()), start=1
        ):
            if internal_idx < 0 or internal_idx >= len(self.inv_item):
                continue

            item_id = self.inv_item[internal_idx]
            if item_id is None:
                continue

            calibrated_score = float(score) * (1.0 - 0.001 * (rank - 1))

            recs.append(
                Recommendation(article_id=item_id, score=calibrated_score)
            )

            if len(recs) >= top_k:
                break

        if not recs:
            return self._fallback(customer_id, top_k)

        return RecommendResult(
            customer_id=customer_id,
            top_k=top_k,
            is_fallback=False,
            recommendations=recs,
        )

    def get_last_debug(self) -> Dict[str, Any]:
        return dict(self._last_debug)
