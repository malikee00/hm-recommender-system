from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch

try:
    import faiss
except Exception:
    faiss = None

from features.preprocess import load_encoders, transform_user_features


def load_training_tables(feature_store_dir: str) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    interactions = pd.read_parquet(os.path.join(feature_store_dir, "interactions.parquet"))
    user_features = pd.read_parquet(os.path.join(feature_store_dir, "user_features.parquet"))
    item_features = pd.read_parquet(os.path.join(feature_store_dir, "item_features.parquet"))
    item_popularity = pd.read_parquet(os.path.join(feature_store_dir, "item_popularity.parquet"))
    return interactions, user_features, item_features, item_popularity


def leave_last_out_split(interactions: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    df = interactions.copy()
    df["t_dat"] = pd.to_datetime(df["t_dat"])
    df = df.sort_values(["customer_id", "t_dat"])

    last_idx = df.groupby("customer_id")["t_dat"].idxmax()
    test = df.loc[last_idx].copy()

    cnt = df.groupby("customer_id").size()
    eligible = set(cnt[cnt >= 2].index)
    test = test[test["customer_id"].isin(eligible)]

    train = df.drop(test.index).copy()
    return train, test


def precision_recall_at_k(recs: List[str], gt: str, k: int) -> Tuple[float, float]:
    topk = recs[:k]
    hit = 1.0 if gt in topk else 0.0
    return hit / float(k), hit


def ndcg_at_k(recs: List[str], gt: str, k: int) -> float:
    topk = recs[:k]
    if gt not in topk:
        return 0.0
    rank = topk.index(gt) + 1
    return 1.0 / np.log2(rank + 1.0)


def compute_metrics(reco_map: Dict[str, List[str]], gt_map: Dict[str, str], ks=(5, 10, 20)) -> Dict[str, float]:
    out: Dict[str, float] = {}
    eval_users = list(gt_map.keys())

    for k in ks:
        precs, recs, ndcgs = [], [], []
        for u in eval_users:
            gt = gt_map[u]
            recs_u = reco_map.get(u, [])
            if not recs_u:
                continue
            p, r = precision_recall_at_k(recs_u, gt, k)
            n = ndcg_at_k(recs_u, gt, k)
            precs.append(p)
            recs.append(r)
            ndcgs.append(n)

        out[f"precision@{k}"] = float(np.mean(precs)) if precs else 0.0
        out[f"recall@{k}"] = float(np.mean(recs)) if recs else 0.0
        out[f"ndcg@{k}"] = float(np.mean(ndcgs)) if ndcgs else 0.0

    out["num_eval_users"] = int(len(eval_users))
    out["num_users_with_recs"] = int(sum(1 for u in eval_users if len(reco_map.get(u, [])) > 0))
    return out


def load_faiss_index(path: str):
    if faiss is None:
        raise RuntimeError("faiss tidak terinstall. Install faiss-cpu terlebih dahulu.")
    return faiss.read_index(path)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--feature_store_dir", default="data/feature_store")
    ap.add_argument("--registry_dir", default="ml/registry/recommender")
    ap.add_argument("--reports_dir", default="ml/reports/recommender")
    ap.add_argument("--topk_max", type=int, default=50)
    ap.add_argument("--ks", default="5,10,20")
    ap.add_argument("--eval_max_users", type=int, default=0)
    ap.add_argument("--seen_cap_per_user", type=int, default=200)
    args = ap.parse_args()

    ks = tuple(int(x.strip()) for x in args.ks.split(","))
    os.makedirs(args.reports_dir, exist_ok=True)

    interactions, user_df, item_df, item_popularity = load_training_tables(args.feature_store_dir)

    interactions = interactions.dropna(subset=["customer_id", "article_id", "t_dat"]).copy()
    interactions["customer_id"] = interactions["customer_id"].astype(str)
    interactions["article_id"] = interactions["article_id"].astype(str)

    train_df, test_df = leave_last_out_split(interactions)

    split_stats = {
        "train_rows": int(len(train_df)),
        "test_rows": int(len(test_df)),
        "num_users_total": int(interactions["customer_id"].nunique()),
        "num_users_eval": int(test_df["customer_id"].nunique()),
        "avg_history_len_train": float(train_df.groupby("customer_id").size().mean()) if len(train_df) else 0.0,
    }
    with open(os.path.join(args.reports_dir, "split_stats.json"), "w", encoding="utf-8") as f:
        json.dump(split_stats, f, indent=2)

    enc = load_encoders(os.path.join(args.registry_dir, "feature_encoders.json"))

    inv_item = [None] * len(enc.item_id_map)
    for iid, idx in enc.item_id_map.items():
        inv_item[idx] = iid

    index = load_faiss_index(os.path.join(args.registry_dir, "faiss.index"))

    user_feat_mat, _ = transform_user_features(user_df, enc)
    user_feat_t = torch.as_tensor(user_feat_mat, dtype=torch.long, device="cpu")

    ckpt = torch.load(os.path.join(args.registry_dir, "two_tower_model.pt"), map_location="cpu")
    from .train import TwoTower

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = TwoTower(
        user_cat_sizes=ckpt["user_cat_sizes"],
        item_cat_sizes=ckpt["item_cat_sizes"],
        age_num_buckets=ckpt["age_num_buckets"],
        embedding_dim=ckpt["embedding_dim"],
        user_cat_cols=ckpt["user_cat_cols"],
        item_cat_cols=ckpt["item_cat_cols"],
    )
    model.load_state_dict(ckpt["state_dict"])
    model = model.to(device)
    model.eval()

    gt_map: Dict[str, str] = dict(zip(test_df["customer_id"].tolist(), test_df["article_id"].tolist()))
    eval_users = list(gt_map.keys())
    if args.eval_max_users and args.eval_max_users > 0:
        eval_users = eval_users[: args.eval_max_users]
        gt_map = {u: gt_map[u] for u in eval_users}

    eval_user_set = set(eval_users)
    train_sub = train_df[train_df["customer_id"].isin(eval_user_set)][["customer_id", "article_id"]]

    seen = defaultdict(list)
    for u, i in zip(train_sub["customer_id"].to_numpy(), train_sub["article_id"].to_numpy()):
        if len(seen[u]) < args.seen_cap_per_user:
            seen[u].append(i)
    seen_sets = {u: set(v) for u, v in seen.items()}

    reco_map: Dict[str, List[str]] = {}
    user_to_index = enc.user_id_map
    bs = 4096

    for start in range(0, len(eval_users), bs):
        batch_users = eval_users[start : start + bs]
        idxs = [user_to_index.get(u, None) for u in batch_users]
        pairs = [(u, idx) for u, idx in zip(batch_users, idxs) if idx is not None]
        if not pairs:
            continue

        u_list, u_idx_list = zip(*pairs)
        idx_t = torch.tensor(u_idx_list, dtype=torch.long, device="cpu")

        feats = user_feat_t.index_select(0, idx_t).to(device, non_blocking=True)
        with torch.no_grad():
            u_emb = model.user_forward(feats).detach().cpu().numpy().astype(np.float32)

        D, I = index.search(u_emb, args.topk_max)

        for u, item_internal_idxs in zip(u_list, I):
            s = seen_sets.get(u, set())
            recs: List[str] = []
            for item_internal_idx in item_internal_idxs.tolist():
                if item_internal_idx < 0 or item_internal_idx >= len(inv_item):
                    continue
                iid = inv_item[item_internal_idx]
                if iid is None:
                    continue
                if iid in s:
                    continue
                recs.append(iid)
            reco_map[u] = recs

    model_metrics = compute_metrics(reco_map, gt_map, ks=ks)
    with open(os.path.join(args.reports_dir, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(model_metrics, f, indent=2)

    pop = item_popularity.copy()
    pop["article_id"] = pop["article_id"].astype(str)

    if "purchase_count" in pop.columns and "popularity_rank" in pop.columns:
        pop = pop.sort_values(["purchase_count", "popularity_rank"], ascending=[False, True])
    elif "purchase_count" in pop.columns:
        pop = pop.sort_values(["purchase_count"], ascending=[False])
    elif "popularity_rank" in pop.columns:
        pop = pop.sort_values(["popularity_rank"], ascending=[True])

    pop_list = pop["article_id"].tolist()
    baseline_reco_map = {u: pop_list for u in gt_map.keys()}

    baseline_metrics = compute_metrics(baseline_reco_map, gt_map, ks=ks)
    with open(os.path.join(args.reports_dir, "baseline_metrics.json"), "w", encoding="utf-8") as f:
        json.dump(baseline_metrics, f, indent=2)

    print("reports dir:", args.reports_dir)
    print("split:", split_stats)
    print("model:", model_metrics)
    print("baseline:", baseline_metrics)


if __name__ == "__main__":
    main()
