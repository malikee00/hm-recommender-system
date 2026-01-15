from __future__ import annotations

import argparse
import json
import os
from datetime import datetime
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

try:
    import faiss
except Exception:
    faiss = None

from features.preprocess import (
    fit_encoders,
    transform_user_features,
    transform_item_features,
    save_encoders,
)


# -------------------------
# Data loading (single source of truth)
# -------------------------
def load_training_tables(
    feature_store_dir: str,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    interactions = pd.read_parquet(os.path.join(feature_store_dir, "interactions.parquet"))
    user_features = pd.read_parquet(os.path.join(feature_store_dir, "user_features.parquet"))
    item_features = pd.read_parquet(os.path.join(feature_store_dir, "item_features.parquet"))
    item_popularity = pd.read_parquet(os.path.join(feature_store_dir, "item_popularity.parquet"))
    return interactions, user_features, item_features, item_popularity


# -------------------------
# Dataset: return indices (int64)
# -------------------------
class InteractionsDataset(Dataset):
    def __init__(self, df: pd.DataFrame, user_id_map: Dict[str, int], item_id_map: Dict[str, int]):
        u = df["customer_id"].astype(str).map(user_id_map)
        i = df["article_id"].astype(str).map(item_id_map)

        mask = u.notna() & i.notna()
        u = u[mask].astype(np.int64).to_numpy()
        i = i[mask].astype(np.int64).to_numpy()

        self.u = u
        self.i = i

    def __len__(self) -> int:
        return len(self.u)

    def __getitem__(self, idx: int):
        return int(self.u[idx]), int(self.i[idx])


# -------------------------
# Two-Tower model
# -------------------------
class TwoTower(nn.Module):
    def __init__(
        self,
        user_cat_sizes: Dict[str, int],
        item_cat_sizes: Dict[str, int],
        age_num_buckets: int,
        embedding_dim: int,
    ):
        super().__init__()

        # user fields: age_bucket + N user cats
        self.user_age_emb = nn.Embedding(age_num_buckets, embedding_dim)
        self.user_cat_embs = nn.ModuleDict(
            {name: nn.Embedding(size, embedding_dim) for name, size in user_cat_sizes.items()}
        )

        # item fields: M item cats
        self.item_cat_embs = nn.ModuleDict(
            {name: nn.Embedding(size, embedding_dim) for name, size in item_cat_sizes.items()}
        )

        # simple MLP heads
        self.user_proj = nn.Sequential(
            nn.Linear((1 + len(user_cat_sizes)) * embedding_dim, embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, embedding_dim),
        )
        self.item_proj = nn.Sequential(
            nn.Linear(len(item_cat_sizes) * embedding_dim, embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, embedding_dim),
        )

    def user_forward(self, user_feat_batch: torch.Tensor) -> torch.Tensor:
        age_bucket = user_feat_batch[:, 0]
        parts = [self.user_age_emb(age_bucket)]

        cat_names = list(self.user_cat_embs.keys())
        for j, name in enumerate(cat_names, start=1):
            parts.append(self.user_cat_embs[name](user_feat_batch[:, j]))

        x = torch.cat(parts, dim=1)  
        z = self.user_proj(x)
        return nn.functional.normalize(z, dim=1)

    def item_forward(self, item_feat_batch: torch.Tensor) -> torch.Tensor:
        parts = []
        cat_names = list(self.item_cat_embs.keys())
        for j, name in enumerate(cat_names):
            parts.append(self.item_cat_embs[name](item_feat_batch[:, j]))

        x = torch.cat(parts, dim=1)  
        z = self.item_proj(x)
        return nn.functional.normalize(z, dim=1)


# -------------------------
# FAISS
# -------------------------
def build_faiss_index(item_emb: np.ndarray, out_path: str) -> None:
    if faiss is None:
        raise RuntimeError("faiss tidak terinstall. Install faiss-cpu terlebih dahulu.")
    dim = item_emb.shape[1]
    index = faiss.IndexFlatIP(dim)  
    index.add(item_emb.astype(np.float32))
    faiss.write_index(index, out_path)


# -------------------------
# Main
# -------------------------
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--feature_store_dir", default="data/feature_store")
    ap.add_argument("--registry_dir", default="ml/registry/recommender")
    ap.add_argument("--embedding_dim", type=int, default=64)
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--batch_size", type=int, default=2048)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--max_interactions", type=int, default=2_000_000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--item_embed_batch_size", type=int, default=8192)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    os.makedirs(args.registry_dir, exist_ok=True)

    # device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_cuda = device.type == "cuda"
    print("[INFO] device:", device)

    # load tables
    interactions, user_df, item_df, item_popularity = load_training_tables(args.feature_store_dir)

    # basic clean
    interactions = interactions.dropna(subset=["customer_id", "article_id", "t_dat"]).copy()
    interactions["customer_id"] = interactions["customer_id"].astype(str)
    interactions["article_id"] = interactions["article_id"].astype(str)

    # subsample for safety
    if len(interactions) > args.max_interactions:
        interactions = interactions.sample(args.max_interactions, random_state=args.seed).reset_index(drop=True)

    # fit encoders (FASE 3.3)
    enc = fit_encoders(user_df=user_df, item_df=item_df, interactions_df=interactions)

    # save id mappings
    with open(os.path.join(args.registry_dir, "user_id_mapping.json"), "w", encoding="utf-8") as f:
        json.dump(enc.user_id_map, f, ensure_ascii=False)
    with open(os.path.join(args.registry_dir, "item_id_mapping.json"), "w", encoding="utf-8") as f:
        json.dump(enc.item_id_map, f, ensure_ascii=False)

    # save encoders
    enc_path = os.path.join(args.registry_dir, "feature_encoders.json")
    save_encoders(enc, enc_path)

    # transform feature tables -> aligned matrices (CPU)
    user_feat_mat, _ = transform_user_features(user_df, enc)
    item_feat_mat, _ = transform_item_features(item_df, enc)

    # store on CPU 
    user_feat_mat_t = torch.as_tensor(user_feat_mat, dtype=torch.long, device="cpu")
    item_feat_mat_t = torch.as_tensor(item_feat_mat, dtype=torch.long, device="cpu")

    # build dataset/loader
    ds = InteractionsDataset(interactions, enc.user_id_map, enc.item_id_map)
    dl = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        drop_last=True,
        pin_memory=use_cuda, 
    )

    # sizes for embeddings
    user_cat_sizes = {k: (max(v.values()) + 1) for k, v in enc.user_cat_maps.items()}
    item_cat_sizes = {k: (max(v.values()) + 1) for k, v in enc.item_cat_maps.items()}

    model = TwoTower(
        user_cat_sizes=user_cat_sizes,
        item_cat_sizes=item_cat_sizes,
        age_num_buckets=enc.age_num_buckets,
        embedding_dim=args.embedding_dim,
    ).to(device)

    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = nn.CrossEntropyLoss()

    # -------------------------
    # TRAIN
    # -------------------------
    model.train()
    for ep in range(1, args.epochs + 1):
        pbar = tqdm(dl, desc=f"epoch {ep}/{args.epochs}")
        total_loss = 0.0
        n = 0

        for u_idx, i_idx in pbar:
            u_idx = u_idx.long()
            i_idx = i_idx.long()

            u_idx_cpu = u_idx.cpu()
            i_idx_cpu = i_idx.cpu()

            u_feat = user_feat_mat_t.index_select(0, u_idx_cpu).to(device, non_blocking=True)
            i_feat = item_feat_mat_t.index_select(0, i_idx_cpu).to(device, non_blocking=True)

            # forward
            u_emb = model.user_forward(u_feat)  # [B, D]
            i_emb = model.item_forward(i_feat)  # [B, D]

            # in-batch negatives
            logits = u_emb @ i_emb.t()  # [B, B]
            target = torch.arange(logits.size(0), device=device, dtype=torch.long)

            loss = loss_fn(logits, target)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            opt.step()

            total_loss += float(loss.item())
            n += 1
            pbar.set_postfix(loss=total_loss / n)

    # -------------------------
    # SAVE MODEL
    # -------------------------
    model_path = os.path.join(args.registry_dir, "two_tower_model.pt")
    torch.save(
        {
            "state_dict": model.state_dict(),
            "embedding_dim": args.embedding_dim,
            "user_cat_sizes": user_cat_sizes,
            "item_cat_sizes": item_cat_sizes,
            "age_num_buckets": enc.age_num_buckets,
            "user_cat_cols": list(enc.user_cat_maps.keys()),
            "item_cat_cols": list(enc.item_cat_maps.keys()),
        },
        model_path,
    )

    # -------------------------
    # PRECOMPUTE ITEM EMBEDDINGS
    # -------------------------
    model.eval()
    with torch.no_grad():
        num_items = item_feat_mat_t.shape[0]
        all_item_emb = np.zeros((num_items, args.embedding_dim), dtype=np.float32)

        bs = int(args.item_embed_batch_size)
        for start in tqdm(range(0, num_items, bs), desc="item-embeddings"):
            end = min(start + bs, num_items)
            feat = item_feat_mat_t[start:end].to(device, non_blocking=True)
            emb = model.item_forward(feat).detach().cpu().numpy().astype(np.float32)
            all_item_emb[start:end] = emb

    np.save(os.path.join(args.registry_dir, "item_embeddings.npy"), all_item_emb)

    # -------------------------
    # FAISS INDEX
    # -------------------------
    index_path = os.path.join(args.registry_dir, "faiss.index")
    build_faiss_index(all_item_emb, index_path)

    # -------------------------
    # POPULARITY BASELINE EXPORT
    # -------------------------
    pop_out = os.path.join(args.registry_dir, "item_popularity.csv")
    pop = item_popularity.copy()
    pop["article_id"] = pop["article_id"].astype(str)

    sort_cols = [c for c in ["purchase_count", "popularity_rank"] if c in pop.columns]
    if sort_cols:
        asc = [False] + ([True] * (len(sort_cols) - 1))
        pop.sort_values(by=sort_cols, ascending=asc, inplace=True)

    pop.to_csv(pop_out, index=False)

    # -------------------------
    # METADATA
    # -------------------------
    meta = {
        "model_type": "two-tower",
        "embedding_dim": args.embedding_dim,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "train_rows": int(len(interactions)),
        "num_users": int(len(enc.user_id_map)),
        "num_items": int(len(enc.item_id_map)),
        "trained_at": datetime.utcnow().isoformat() + "Z",
        "device": str(device),
        "feature_store_dir": args.feature_store_dir,
        "artifacts": {
            "model": "two_tower_model.pt",
            "item_embeddings": "item_embeddings.npy",
            "faiss_index": "faiss.index",
            "user_id_mapping": "user_id_mapping.json",
            "item_id_mapping": "item_id_mapping.json",
            "feature_encoders": "feature_encoders.json",
            "item_popularity": "item_popularity.csv",
        },
    }
    with open(os.path.join(args.registry_dir, "metadata.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print("[DONE] Training done. Artifacts saved to:", args.registry_dir)


if __name__ == "__main__":
    main()
