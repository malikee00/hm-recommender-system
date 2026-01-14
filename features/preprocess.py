from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Dict, List, Tuple, Any

import numpy as np
import pandas as pd


USER_CAT_COLS = ["club_member_status", "fashion_news_frequency", "active"]
ITEM_CAT_COLS = ["product_group_name", "department_name", "colour_group_name"]


def _clean_str(x: Any) -> str:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "__MISSING__"
    s = str(x).strip()
    return s if s else "__MISSING__"


def bucketize_age(age: Any) -> int:
    """Return age bucket index."""
    if age is None or (isinstance(age, float) and np.isnan(age)):
        return 0  # unknown
    try:
        a = int(age)
    except Exception:
        return 0
    # buckets: 0 unknown, 1: <=18, 2:19-24, 3:25-34, 4:35-44, 5:45-54, 6:55-64, 7:65+
    if a <= 18:
        return 1
    if a <= 24:
        return 2
    if a <= 34:
        return 3
    if a <= 44:
        return 4
    if a <= 54:
        return 5
    if a <= 64:
        return 6
    return 7


def build_id_mapping(values: List[str]) -> Dict[str, int]:
    """0 reserved for PAD/UNK not used; but we keep it simple: start from 0."""
    uniq = pd.Series(values).dropna().astype(str).unique().tolist()
    return {v: i for i, v in enumerate(uniq)}


def build_cat_mapping(series: pd.Series) -> Dict[str, int]:
    uniq = series.map(_clean_str).unique().tolist()
    mapping = {"__MISSING__": 0}
    idx = 1
    for v in uniq:
        v2 = _clean_str(v)
        if v2 not in mapping:
            mapping[v2] = idx
            idx += 1
    return mapping


@dataclass
class Encoders:
    user_id_map: Dict[str, int]
    item_id_map: Dict[str, int]
    user_cat_maps: Dict[str, Dict[str, int]]
    item_cat_maps: Dict[str, Dict[str, int]]
    age_num_buckets: int = 8  # 0..7

    def to_json(self) -> Dict[str, Any]:
        return {
            "user_id_map_size": len(self.user_id_map),
            "item_id_map_size": len(self.item_id_map),
            "user_id_map": self.user_id_map,
            "item_id_map": self.item_id_map,
            "user_cat_maps": self.user_cat_maps,
            "item_cat_maps": self.item_cat_maps,
            "age_num_buckets": self.age_num_buckets,
            "user_cat_cols": USER_CAT_COLS,
            "item_cat_cols": ITEM_CAT_COLS,
        }

    @staticmethod
    def from_json(d: Dict[str, Any]) -> "Encoders":
        return Encoders(
            user_id_map=d["user_id_map"],
            item_id_map=d["item_id_map"],
            user_cat_maps=d["user_cat_maps"],
            item_cat_maps=d["item_cat_maps"],
            age_num_buckets=int(d.get("age_num_buckets", 8)),
        )


def fit_encoders(user_df: pd.DataFrame, item_df: pd.DataFrame, interactions_df: pd.DataFrame) -> Encoders:
    user_ids = interactions_df["customer_id"].astype(str).tolist()
    item_ids = interactions_df["article_id"].astype(str).tolist()

    user_id_map = build_id_mapping(user_ids)
    item_id_map = build_id_mapping(item_ids)

    user_cat_maps = {}
    for c in USER_CAT_COLS:
        if c not in user_df.columns:
            user_df[c] = "__MISSING__"
        user_cat_maps[c] = build_cat_mapping(user_df[c])

    item_cat_maps = {}
    for c in ITEM_CAT_COLS:
        if c not in item_df.columns:
            item_df[c] = "__MISSING__"
        item_cat_maps[c] = build_cat_mapping(item_df[c])

    return Encoders(
        user_id_map=user_id_map,
        item_id_map=item_id_map,
        user_cat_maps=user_cat_maps,
        item_cat_maps=item_cat_maps,
    )


def transform_user_features(user_df: pd.DataFrame, enc: Encoders) -> Tuple[np.ndarray, List[str]]:
    inv_user = [None] * len(enc.user_id_map)
    for uid, idx in enc.user_id_map.items():
        inv_user[idx] = uid

    u = user_df.copy()
    u["customer_id"] = u["customer_id"].astype(str)

    u = u.drop_duplicates("customer_id").set_index("customer_id", drop=False)

    num_users = len(enc.user_id_map)
    num_fields = 1 + len(USER_CAT_COLS)
    mat = np.zeros((num_users, num_fields), dtype=np.int64)

    for idx, uid in enumerate(inv_user):
        if uid is None:
            continue
        if uid in u.index:
            row = u.loc[uid]
            mat[idx, 0] = bucketize_age(row.get("age", None))
            for j, c in enumerate(USER_CAT_COLS, start=1):
                m = enc.user_cat_maps[c]
                mat[idx, j] = m.get(_clean_str(row.get(c, None)), 0)
        else:
            mat[idx, 0] = 0

    return mat, inv_user


def transform_item_features(item_df: pd.DataFrame, enc: Encoders) -> Tuple[np.ndarray, List[str]]:
    inv_item = [None] * len(enc.item_id_map)
    for iid, idx in enc.item_id_map.items():
        inv_item[idx] = iid

    it = item_df.copy()
    it["article_id"] = it["article_id"].astype(str)
    it = it.drop_duplicates("article_id").set_index("article_id", drop=False)

    num_items = len(enc.item_id_map)
    num_fields = len(ITEM_CAT_COLS)
    mat = np.zeros((num_items, num_fields), dtype=np.int64)

    for idx, iid in enumerate(inv_item):
        if iid is None:
            continue
        if iid in it.index:
            row = it.loc[iid]
            for j, c in enumerate(ITEM_CAT_COLS):
                m = enc.item_cat_maps[c]
                mat[idx, j] = m.get(_clean_str(row.get(c, None)), 0)
        else:
            # default all zeros
            pass

    return mat, inv_item


def save_encoders(enc: Encoders, path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(enc.to_json(), f, ensure_ascii=False, indent=2)


def load_encoders(path: str) -> Encoders:
    with open(path, "r", encoding="utf-8") as f:
        d = json.load(f)
    return Encoders.from_json(d)
