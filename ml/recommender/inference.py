from __future__ import annotations

import os
from dataclasses import dataclass
from typing import List

import numpy as np
import pandas as pd
import torch

try:
    import faiss
except Exception:
    faiss = None

from features.preprocess import load_encoders, transform_user_features


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
    def __init__(self, registry_dir: str, feature_store_dir: str):
        self.registry_dir = registry_dir
        self.feature_store_dir = feature_store_dir
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._loaded = False

    def load(self):
        if self._loaded:
            return

        if faiss is None:
            raise RuntimeError("faiss tidak terinstall. Install faiss-cpu terlebih dahulu.")

        self.enc = load_encoders(os.path.join(self.registry_dir, "feature_encoders.json"))

        self.inv_item = [None] * len(self.enc.item_id_map)
        for iid, idx in self.enc.item_id_map.items():
            self.inv_item[idx] = iid

        self.index = faiss.read_index(os.path.join(self.registry_dir, "faiss.index"))

        pop_path = os.path.join(self.registry_dir, "item_popularity.csv")
        pop = pd.read_csv(pop_path)
        pop["article_id"] = pop["article_id"].astype(str)
        self.pop_list = pop["article_id"].tolist()

        user_df = pd.read_parquet(os.path.join(self.feature_store_dir, "user_features.parquet"))
        user_feat_mat, _ = transform_user_features(user_df, self.enc)
        self.user_feat_t = torch.as_tensor(user_feat_mat, dtype=torch.long, device="cpu")

        ckpt = torch.load(os.path.join(self.registry_dir, "two_tower_model.pt"), map_location="cpu")
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

    def recommend(self, customer_id: str, top_k: int = 10) -> RecommendResult:
        self.load()
        customer_id = str(customer_id)

        if customer_id not in self.enc.user_id_map:
            recs = [Recommendation(article_id=a, score=0.0) for a in self.pop_list[:top_k]]
            return RecommendResult(customer_id=customer_id, top_k=top_k, is_fallback=True, recommendations=recs)

        u_idx = self.enc.user_id_map[customer_id]
        feat = self.user_feat_t[u_idx : u_idx + 1].to(self.device)

        with torch.no_grad():
            u_emb = self.model.user_forward(feat).detach().cpu().numpy().astype(np.float32)

        k_search = max(top_k, 50)
        D, I = self.index.search(u_emb, k_search)

        recs: List[Recommendation] = []
        for score, internal_idx in zip(D[0].tolist(), I[0].tolist()):
            if internal_idx < 0 or internal_idx >= len(self.inv_item):
                continue
            iid = self.inv_item[internal_idx]
            if iid is None:
                continue
            recs.append(Recommendation(article_id=iid, score=float(score)))
            if len(recs) >= top_k:
                break

        if not recs:
            recs = [Recommendation(article_id=a, score=0.0) for a in self.pop_list[:top_k]]
            return RecommendResult(customer_id=customer_id, top_k=top_k, is_fallback=True, recommendations=recs)

        return RecommendResult(customer_id=customer_id, top_k=top_k, is_fallback=False, recommendations=recs)
