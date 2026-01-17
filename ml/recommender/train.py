from __future__ import annotations

import argparse
import json
import os
from datetime import datetime
from typing import Dict, Tuple, List, Optional

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


def load_training_tables(feature_store_dir: str) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    paths = {
        "interactions": os.path.join(feature_store_dir, "interactions.parquet"),
        "user_features": os.path.join(feature_store_dir, "user_features.parquet"),
        "item_features": os.path.join(feature_store_dir, "item_features.parquet"),
        "item_popularity": os.path.join(feature_store_dir, "item_popularity.parquet"),
    }
    for _, p in paths.items():
        if not os.path.exists(p):
            raise FileNotFoundError(p)

    interactions = pd.read_parquet(paths["interactions"])
    user_features = pd.read_parquet(paths["user_features"])
    item_features = pd.read_parquet(paths["item_features"])
    item_popularity = pd.read_parquet(paths["item_popularity"])
    return interactions, user_features, item_features, item_popularity


def load_user_history_agg(feature_store_dir: str) -> Optional[pd.DataFrame]:
    p = os.path.join(feature_store_dir, "user_history_agg.parquet")
    if not os.path.exists(p):
        return None
    df = pd.read_parquet(p)
    if "customer_id" not in df.columns:
        return None
    if "top_product_group_name" not in df.columns:
        return None
    out = df[["customer_id", "top_product_group_name"]].copy()
    out["customer_id"] = out["customer_id"].astype(str)
    out["top_product_group_name"] = out["top_product_group_name"].astype(str)
    return out


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
        return int(len(self.u))

    def __getitem__(self, idx: int):
        return int(self.u[idx]), int(self.i[idx])


class TwoTower(nn.Module):
    def __init__(
        self,
        user_cat_sizes: Dict[str, int],
        item_cat_sizes: Dict[str, int],
        age_num_buckets: int,
        embedding_dim: int,
        user_cat_cols: List[str],
        item_cat_cols: List[str],
    ):
        super().__init__()

        self.user_cat_cols = list(user_cat_cols)
        self.item_cat_cols = list(item_cat_cols)

        self.user_age_emb = nn.Embedding(int(age_num_buckets), int(embedding_dim))

        self.user_cat_embs = nn.ModuleDict()
        for name in self.user_cat_cols:
            self.user_cat_embs[name] = nn.Embedding(int(user_cat_sizes[name]), int(embedding_dim))

        self.item_cat_embs = nn.ModuleDict()
        for name in self.item_cat_cols:
            self.item_cat_embs[name] = nn.Embedding(int(item_cat_sizes[name]), int(embedding_dim))

        self.user_proj = nn.Sequential(
            nn.Linear((1 + len(self.user_cat_cols)) * embedding_dim, embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, embedding_dim),
        )
        self.item_proj = nn.Sequential(
            nn.Linear(len(self.item_cat_cols) * embedding_dim, embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, embedding_dim),
        )

    def user_forward(self, user_feat_batch: torch.Tensor) -> torch.Tensor:
        user_feat_batch = user_feat_batch.long()
        age_bucket = user_feat_batch[:, 0]
        parts = [self.user_age_emb(age_bucket)]
        for j, name in enumerate(self.user_cat_cols, start=1):
            parts.append(self.user_cat_embs[name](user_feat_batch[:, j]))
        x = torch.cat(parts, dim=1)
        z = self.user_proj(x)
        return nn.functional.normalize(z, dim=1)

    def item_forward(self, item_feat_batch: torch.Tensor) -> torch.Tensor:
        item_feat_batch = item_feat_batch.long()
        parts = []
        for j, name in enumerate(self.item_cat_cols):
            parts.append(self.item_cat_embs[name](item_feat_batch[:, j]))
        x = torch.cat(parts, dim=1)
        z = self.item_proj(x)
        return nn.functional.normalize(z, dim=1)


def build_faiss_index(item_emb: np.ndarray, out_path: str) -> None:
    if faiss is None:
        raise RuntimeError("faiss tidak terinstall")
    dim = int(item_emb.shape[1])
    index = faiss.IndexFlatIP(dim)
    index.add(item_emb.astype(np.float32))
    faiss.write_index(index, out_path)


def sample_negatives_uniform(pos_i: torch.Tensor, num_items: int) -> torch.Tensor:
    neg = torch.randint(low=0, high=num_items, size=pos_i.shape, device=pos_i.device, dtype=torch.long)
    mask = neg.eq(pos_i)
    if mask.any():
        neg2 = torch.randint(low=0, high=num_items, size=pos_i.shape, device=pos_i.device, dtype=torch.long)
        neg = torch.where(mask, neg2, neg)
    mask = neg.eq(pos_i)
    if mask.any():
        neg = (neg + 1) % num_items
    return neg


def bpr_loss(u: torch.Tensor, pos: torch.Tensor, neg: torch.Tensor) -> torch.Tensor:
    pos_scores = (u * pos).sum(dim=1)
    neg_scores = (u * neg).sum(dim=1)
    x = pos_scores - neg_scores
    return -torch.log(torch.sigmoid(x) + 1e-12).mean()


class PopularNegSampler:
    def __init__(
        self,
        item_popularity_df: pd.DataFrame,
        item_id_map: Dict[str, int],
        power: float = 0.75,
        top_n: int = 0,
    ):
        df = item_popularity_df.copy()
        if "article_id" not in df.columns:
            raise ValueError("item_popularity.parquet harus punya kolom 'article_id'")
        df["article_id"] = df["article_id"].astype(str)

        if "purchase_count" in df.columns:
            w = df["purchase_count"].astype(float).to_numpy()
        elif "popularity_rank" in df.columns:
            r = df["popularity_rank"].astype(float).to_numpy()
            w = 1.0 / np.maximum(r, 1.0)
        else:
            w = np.ones(len(df), dtype=float)

        idxs: List[int] = []
        weights: List[float] = []
        for aid, ww in zip(df["article_id"].tolist(), w.tolist()):
            j = item_id_map.get(aid, None)
            if j is None:
                continue
            if ww <= 0 or not np.isfinite(ww):
                continue
            idxs.append(int(j))
            weights.append(float(ww))

        if not idxs:
            self.idxs = None
            self.probs = None
            return

        idxs = np.asarray(idxs, dtype=np.int64)
        weights = np.asarray(weights, dtype=np.float64)

        if power != 1.0:
            weights = np.power(weights, float(power))

        if top_n and top_n > 0 and top_n < len(idxs):
            order = np.argsort(weights)[::-1]
            order = order[: int(top_n)]
            idxs = idxs[order]
            weights = weights[order]

        s = weights.sum()
        if s <= 0:
            self.idxs = None
            self.probs = None
            return

        self.idxs = idxs
        self.probs = (weights / s).astype(np.float64)

    def sample(self, batch_size: int, device: torch.device) -> torch.Tensor:
        if self.idxs is None or self.probs is None:
            raise RuntimeError("popular sampler tidak siap (idxs/probs kosong)")
        choice = np.random.choice(self.idxs, size=int(batch_size), replace=True, p=self.probs)
        return torch.as_tensor(choice, dtype=torch.long, device=device)


def sample_negatives(
    pos_i: torch.Tensor,
    num_items: int,
    mode: str,
    popular_sampler: Optional[PopularNegSampler],
    device: torch.device,
    mixed_alpha: float,
) -> torch.Tensor:
    mode = str(mode).lower()
    bs = int(pos_i.numel())

    if mode == "uniform" or popular_sampler is None:
        return sample_negatives_uniform(pos_i, num_items)

    if mode == "popular":
        neg = popular_sampler.sample(bs, device=device).view_as(pos_i)

    elif mode == "mixed":
        neg_pop = popular_sampler.sample(bs, device=device).view_as(pos_i)
        neg_uni = sample_negatives_uniform(pos_i, num_items)
        m = torch.rand_like(pos_i.float()) < float(mixed_alpha)
        neg = torch.where(m, neg_pop, neg_uni)

    else:
        raise ValueError("neg_sampling harus salah satu dari: uniform, popular, mixed")

    mask = neg.eq(pos_i)
    if mask.any():
        if mode == "popular":
            neg2 = popular_sampler.sample(bs, device=device).view_as(pos_i)
        else:
            neg2 = sample_negatives_uniform(pos_i, num_items)
        neg = torch.where(mask, neg2, neg)

    mask = neg.eq(pos_i)
    if mask.any():
        neg = (neg + 1) % num_items

    return neg


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--feature_store_dir", default="data/feature_store")
    ap.add_argument("--registry_dir", default="ml/registry/recommender")
    ap.add_argument("--reports_dir", default="ml/reports/recommender")
    ap.add_argument("--embedding_dim", type=int, default=64)
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--batch_size", type=int, default=2048)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--max_interactions", type=int, default=2_000_000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--item_embed_batch_size", type=int, default=8192)
    ap.add_argument("--skip_faiss", action="store_true")
    ap.add_argument("--objective", choices=["inbatch_ce", "bpr"], default="bpr")
    ap.add_argument("--weight_decay", type=float, default=0.0)

    ap.add_argument("--neg_sampling", choices=["uniform", "popular", "mixed"], default="mixed")
    ap.add_argument("--popular_power", type=float, default=0.75)
    ap.add_argument("--popular_top_n", type=int, default=0)
    ap.add_argument("--mixed_alpha", type=float, default=0.5)

    args = ap.parse_args()

    torch.manual_seed(int(args.seed))
    np.random.seed(int(args.seed))

    os.makedirs(args.registry_dir, exist_ok=True)
    os.makedirs(args.reports_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_cuda = device.type == "cuda"

    print("device:", device)
    print("feature_store_dir:", args.feature_store_dir)
    print("registry_dir:", args.registry_dir)
    print("reports_dir:", args.reports_dir)
    print("objective:", args.objective)
    print("neg_sampling:", args.neg_sampling)

    interactions, user_df, item_df, item_popularity = load_training_tables(args.feature_store_dir)
    user_hist = load_user_history_agg(args.feature_store_dir)

    print("raw interactions rows:", int(len(interactions)))
    print("raw user_features rows:", int(len(user_df)))
    print("raw item_features rows:", int(len(item_df)))
    print("user_history_agg loaded:", bool(user_hist is not None))

    interactions = interactions.dropna(subset=["customer_id", "article_id", "t_dat"]).copy()
    interactions["customer_id"] = interactions["customer_id"].astype(str)
    interactions["article_id"] = interactions["article_id"].astype(str)

    if len(interactions) > int(args.max_interactions):
        interactions = interactions.sample(int(args.max_interactions), random_state=int(args.seed)).reset_index(drop=True)

    print("train interactions rows:", int(len(interactions)))
    print("unique customers in interactions:", int(interactions["customer_id"].nunique()))
    print("unique articles in interactions:", int(interactions["article_id"].nunique()))

    enc = fit_encoders(
        user_df=user_df,
        item_df=item_df,
        interactions_df=interactions,
        user_history_agg_df=user_hist,
    )

    with open(os.path.join(args.registry_dir, "user_id_mapping.json"), "w", encoding="utf-8") as f:
        json.dump(enc.user_id_map, f, ensure_ascii=False)
    with open(os.path.join(args.registry_dir, "item_id_mapping.json"), "w", encoding="utf-8") as f:
        json.dump(enc.item_id_map, f, ensure_ascii=False)

    enc_path = os.path.join(args.registry_dir, "feature_encoders.json")
    save_encoders(enc, enc_path)

    user_feat_mat, _ = transform_user_features(user_df, enc, user_history_agg_df=user_hist)
    item_feat_mat, _ = transform_item_features(item_df, enc)

    user_feat_mat_t = torch.as_tensor(user_feat_mat, dtype=torch.long, device="cpu")
    item_feat_mat_t = torch.as_tensor(item_feat_mat, dtype=torch.long, device="cpu")

    ds = InteractionsDataset(interactions, enc.user_id_map, enc.item_id_map)
    print("dataset pairs:", int(len(ds)))

    effective_bs = int(args.batch_size)
    if len(ds) < effective_bs:
        effective_bs = max(1, min(int(args.batch_size), int(len(ds))))
    drop_last = True if len(ds) >= effective_bs else False

    dl = DataLoader(
        ds,
        batch_size=effective_bs,
        shuffle=True,
        num_workers=0,
        drop_last=drop_last,
        pin_memory=use_cuda,
    )

    user_cat_cols = list(enc.user_cat_maps.keys())
    item_cat_cols = list(enc.item_cat_maps.keys())
    print("user_cat_cols:", user_cat_cols)
    print("item_cat_cols:", item_cat_cols)

    user_cat_sizes = {k: (max(v.values()) + 1) for k, v in enc.user_cat_maps.items()}
    item_cat_sizes = {k: (max(v.values()) + 1) for k, v in enc.item_cat_maps.items()}

    model = TwoTower(
        user_cat_sizes=user_cat_sizes,
        item_cat_sizes=item_cat_sizes,
        age_num_buckets=enc.age_num_buckets,
        embedding_dim=int(args.embedding_dim),
        user_cat_cols=user_cat_cols,
        item_cat_cols=item_cat_cols,
    ).to(device)

    opt = torch.optim.Adam(model.parameters(), lr=float(args.lr), weight_decay=float(args.weight_decay))
    loss_fn = nn.CrossEntropyLoss()

    num_items_total = int(len(enc.item_id_map))
    popular_sampler = None
    if args.neg_sampling in ("popular", "mixed"):
        popular_sampler = PopularNegSampler(
            item_popularity_df=item_popularity,
            item_id_map=enc.item_id_map,
            power=float(args.popular_power),
            top_n=int(args.popular_top_n),
        )
        if popular_sampler.idxs is None:
            print("popular sampler fallback -> uniform (mapping kosong)")

    model.train()
    for ep in range(1, int(args.epochs) + 1):
        pbar = tqdm(dl, desc=f"epoch {ep}/{int(args.epochs)}")
        total_loss = 0.0
        n = 0

        for u_idx, pos_i_idx in pbar:
            u_idx_cpu = u_idx.long().cpu()
            pos_i_idx_cpu = pos_i_idx.long().cpu()

            u_feat = user_feat_mat_t.index_select(0, u_idx_cpu).to(device, non_blocking=True)
            pos_feat = item_feat_mat_t.index_select(0, pos_i_idx_cpu).to(device, non_blocking=True)

            u_emb = model.user_forward(u_feat)
            pos_emb = model.item_forward(pos_feat)

            if args.objective == "inbatch_ce":
                logits = u_emb @ pos_emb.t()
                target = torch.arange(logits.size(0), device=device, dtype=torch.long)
                loss = loss_fn(logits, target)
            else:
                neg_i_idx = sample_negatives(
                    pos_i=pos_i_idx_cpu.to(device, non_blocking=True),
                    num_items=num_items_total,
                    mode=str(args.neg_sampling),
                    popular_sampler=popular_sampler,
                    device=device,
                    mixed_alpha=float(args.mixed_alpha),
                )
                neg_feat = item_feat_mat_t.index_select(0, neg_i_idx.detach().cpu()).to(device, non_blocking=True)
                neg_emb = model.item_forward(neg_feat)
                loss = bpr_loss(u_emb, pos_emb, neg_emb)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            opt.step()

            total_loss += float(loss.item())
            n += 1
            pbar.set_postfix(loss=(total_loss / n if n else 0.0))

    model_path = os.path.join(args.registry_dir, "two_tower_model.pt")
    torch.save(
        {
            "state_dict": model.state_dict(),
            "embedding_dim": int(args.embedding_dim),
            "user_cat_sizes": user_cat_sizes,
            "item_cat_sizes": item_cat_sizes,
            "age_num_buckets": int(enc.age_num_buckets),
            "user_cat_cols": user_cat_cols,
            "item_cat_cols": item_cat_cols,
        },
        model_path,
    )
    print("saved:", model_path)

    model.eval()
    with torch.no_grad():
        num_items = int(item_feat_mat_t.shape[0])
        all_item_emb = np.zeros((num_items, int(args.embedding_dim)), dtype=np.float32)

        bs = int(args.item_embed_batch_size)
        for start in tqdm(range(0, num_items, bs), desc="item-embeddings"):
            end = min(start + bs, num_items)
            feat = item_feat_mat_t[start:end].to(device, non_blocking=True)
            emb = model.item_forward(feat).detach().cpu().numpy().astype(np.float32)
            all_item_emb[start:end] = emb

    emb_path = os.path.join(args.registry_dir, "item_embeddings.npy")
    np.save(emb_path, all_item_emb)
    print("saved:", emb_path)

    index_path = os.path.join(args.registry_dir, "faiss.index")
    if args.skip_faiss or faiss is None:
        print("faiss skipped")
    else:
        build_faiss_index(all_item_emb, index_path)
        print("saved:", index_path)

    pop_out = os.path.join(args.registry_dir, "item_popularity.csv")
    pop = item_popularity.copy()
    pop["article_id"] = pop["article_id"].astype(str)

    sort_cols = [c for c in ["purchase_count", "popularity_rank"] if c in pop.columns]
    if sort_cols:
        asc = [False] + ([True] * (len(sort_cols) - 1))
        pop.sort_values(by=sort_cols, ascending=asc, inplace=True)

    pop.to_csv(pop_out, index=False)
    print("saved:", pop_out)

    meta = {
        "model_type": "two-tower",
        "objective": str(args.objective),
        "neg_sampling": str(args.neg_sampling),
        "popular_power": float(args.popular_power),
        "popular_top_n": int(args.popular_top_n),
        "mixed_alpha": float(args.mixed_alpha),
        "embedding_dim": int(args.embedding_dim),
        "epochs": int(args.epochs),
        "batch_size": int(effective_bs),
        "lr": float(args.lr),
        "weight_decay": float(args.weight_decay),
        "max_interactions": int(args.max_interactions),
        "train_rows": int(len(interactions)),
        "num_users": int(len(enc.user_id_map)),
        "num_items": int(len(enc.item_id_map)),
        "trained_at": datetime.utcnow().isoformat() + "Z",
        "device": str(device),
        "feature_store_dir": str(args.feature_store_dir),
        "uses_user_history_agg": bool(user_hist is not None),
        "artifacts": {
            "model": "two_tower_model.pt",
            "item_embeddings": "item_embeddings.npy",
            "faiss_index": ("faiss.index" if (not args.skip_faiss and faiss is not None) else None),
            "user_id_mapping": "user_id_mapping.json",
            "item_id_mapping": "item_id_mapping.json",
            "feature_encoders": "feature_encoders.json",
            "item_popularity": "item_popularity.csv",
        },
    }

    meta_path = os.path.join(args.registry_dir, "metadata.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    print("saved:", meta_path)

    print("training artifacts saved to:", args.registry_dir)


if __name__ == "__main__":
    main()
