from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import inspect
import math
import re

import pandas as pd

from ml.recommender.inference import RecommenderService


@dataclass
class RecItem:
    article_id: str
    score: float
    image_url: str
    meta: Dict[str, Any]


class LocalHMService:
    def __init__(self, feature_store_dir: Path, registry_run_dir: Path):
        self.feature_store_dir = Path(feature_store_dir)
        self.registry_run_dir = Path(registry_run_dir)

        self._customers: Optional[pd.DataFrame] = None
        self._items: Optional[pd.DataFrame] = None
        self._interactions: Optional[pd.DataFrame] = None
        self._user_hist_agg: Optional[pd.DataFrame] = None

        self.reco = self._build_recommender()

    def _build_recommender(self) -> RecommenderService:
        init_sig = inspect.signature(RecommenderService.__init__)
        params = set(init_sig.parameters.keys())

        if "registry_run_dir" in params and "feature_store_dir" in params:
            return RecommenderService(
                registry_run_dir=str(self.registry_run_dir),
                feature_store_dir=str(self.feature_store_dir),
            )

        if "run_dir" in params and "feature_store_dir" in params:
            return RecommenderService(
                run_dir=str(self.registry_run_dir),
                feature_store_dir=str(self.feature_store_dir),
            )

        if "registry_dir" in params and "feature_store_dir" in params:
            return RecommenderService(
                registry_dir=str(self.registry_run_dir),
                feature_store_dir=str(self.feature_store_dir),
            )

        if "registry_run_dir" in params and "feature_store" in params:
            return RecommenderService(
                registry_run_dir=str(self.registry_run_dir),
                feature_store=str(self.feature_store_dir),
            )

        if len(params) <= 1:
            svc = RecommenderService()
            for attr_name in ["registry_run_dir", "run_dir", "registry_dir", "artifacts_dir"]:
                if hasattr(svc, attr_name):
                    setattr(svc, attr_name, str(self.registry_run_dir))
                    break
            for attr_name in ["feature_store_dir", "feature_store", "feature_store_path"]:
                if hasattr(svc, attr_name):
                    setattr(svc, attr_name, str(self.feature_store_dir))
                    break
            return svc

        raise TypeError(
            "RecommenderService.__init__ signature tidak cocok dengan wrapper. "
            f"Param __init__ yang terbaca: {list(params)}"
        )

    def load(self) -> None:
        self.reco.load()

        self._customers = self._safe_read_parquet(self.feature_store_dir / "user_features.parquet")
        self._items = self._safe_read_parquet(self.feature_store_dir / "item_features.parquet")
        self._interactions = self._safe_read_parquet(self.feature_store_dir / "interactions.parquet")
        uh = self.feature_store_dir / "user_history_agg.parquet"
        self._user_hist_agg = self._safe_read_parquet(uh) if uh.exists() else None

        self._prepare_indexes()

    def _safe_read_parquet(self, path: Path) -> pd.DataFrame:
        if not path.exists():
            raise FileNotFoundError(f"Missing required parquet: {path}")
        return pd.read_parquet(path)

    def _normalize_article_id(self, x: Any) -> str:
        if x is None:
            return ""
        s = str(x).strip()

        if re.fullmatch(r"\d+\.0", s):
            s = s.split(".")[0]

        if re.fullmatch(r"\d+", s):
            return s.zfill(10)

        digits = re.sub(r"\D", "", s)
        if digits:
            return digits.zfill(10)

        return s

    def _prepare_indexes(self) -> None:
        if self._items is not None:
            self._items = self._items.copy()
            if "article_id" in self._items.columns:
                self._items["article_id"] = self._items["article_id"].apply(self._normalize_article_id)
                self._items = self._items.drop_duplicates("article_id")
                self._items = self._items.set_index("article_id", drop=False)

        if self._customers is not None:
            self._customers = self._customers.copy()
            if "customer_id" in self._customers.columns:
                self._customers["customer_id"] = self._customers["customer_id"].astype(str)
                self._customers = self._customers.drop_duplicates("customer_id")
                self._customers = self._customers.set_index("customer_id", drop=False)

        if self._interactions is not None:
            self._interactions = self._interactions.copy()
            if "customer_id" in self._interactions.columns:
                self._interactions["customer_id"] = self._interactions["customer_id"].astype(str)
            if "article_id" in self._interactions.columns:
                self._interactions["article_id"] = self._interactions["article_id"].apply(self._normalize_article_id)
            if "t_dat" in self._interactions.columns:
                self._interactions["t_dat"] = pd.to_datetime(self._interactions["t_dat"], errors="coerce")

    def _to_json_safe(self, obj: Any) -> Any:
        if obj is None:
            return None

        if isinstance(obj, float):
            return obj if math.isfinite(obj) else None

        if isinstance(obj, (pd.Timestamp,)):
            return obj.isoformat()

        if isinstance(obj, dict):
            return {str(k): self._to_json_safe(v) for k, v in obj.items()}

        if isinstance(obj, (list, tuple)):
            return [self._to_json_safe(v) for v in obj]

        if hasattr(obj, "item") and callable(getattr(obj, "item")):
            try:
                return self._to_json_safe(obj.item())
            except Exception:
                pass

        try:
            if pd.isna(obj):
                return None
        except Exception:
            pass

        return obj

    def image_url(self, article_id: str) -> str:
        aid = self._normalize_article_id(article_id)
        prefix = aid[:3] if len(aid) >= 3 else "000"
        return f"/img/{prefix}/{aid}.jpg"

    def _item_meta(self, article_id: str) -> Dict[str, Any]:
        if self._items is None:
            return {}
        aid = self._normalize_article_id(article_id)
        if aid in self._items.index:
            row = self._items.loc[aid].to_dict()
            keep = [
                "product_type_name",
                "product_group_name",
                "department_name",
                "colour_group_name",
                "section_name",
                "garment_group_name",
            ]
            return {k: self._to_json_safe(row.get(k)) for k in keep if k in row}
        return {}

    def sample_customers(self, n: int = 200, history_n: int = 5) -> pd.DataFrame:
        if self._interactions is None:
            if hasattr(self.reco, "enc") and hasattr(self.reco.enc, "user_id_map"):
                cids = list(self.reco.enc.user_id_map.keys())[:n]
                return pd.DataFrame({"customer_id": cids, "history": [""] * len(cids)})
            return pd.DataFrame({"customer_id": [], "history": []})

        df = self._interactions.copy()
        cols = ["customer_id", "article_id"]
        if "t_dat" in df.columns:
            cols.append("t_dat")
        df = df[cols].copy()

        if "t_dat" in df.columns:
            df = df.sort_values(["customer_id", "t_dat"], ascending=[True, False])
        else:
            df = df.sort_values(["customer_id"], ascending=[True])

        df["rank"] = df.groupby("customer_id").cumcount() + 1
        df = df[df["rank"] <= history_n]

        hist = (
            df.groupby("customer_id")["article_id"]
            .apply(lambda s: "|".join([str(x) for x in s.tolist()]))
            .reset_index()
            .rename(columns={"article_id": "history"})
        )

        if self._user_hist_agg is not None and "customer_id" in self._user_hist_agg.columns and "total_purchases" in self._user_hist_agg.columns:
            base = self._user_hist_agg[["customer_id", "total_purchases"]].copy()
            base["customer_id"] = base["customer_id"].astype(str)
            hist["customer_id"] = hist["customer_id"].astype(str)
            out = base.merge(hist, on="customer_id", how="left").sort_values("total_purchases", ascending=False)
            out["history"] = out["history"].fillna("")
            return out[["customer_id", "history"]].head(n).reset_index(drop=True)

        return hist.head(n).reset_index(drop=True)

    def user_profile_and_history(self, customer_id: str, last_n: int = 5) -> Dict[str, Any]:
        customer_id = str(customer_id)

        profile: Dict[str, Any] = {"customer_id": customer_id}
        if self._customers is not None and customer_id in self._customers.index:
            row = self._customers.loc[customer_id].to_dict()
            keep = ["customer_id", "age", "club_member_status", "fashion_news_frequency", "active"]
            for k in keep:
                if k in row:
                    profile[k] = self._to_json_safe(row.get(k))

        history_items: List[Dict[str, Any]] = []
        if self._interactions is not None:
            df = self._interactions[self._interactions["customer_id"] == customer_id]
            if "t_dat" in df.columns:
                df = df.sort_values("t_dat", ascending=False)
            df = df.head(last_n)

            for _, r in df.iterrows():
                aid = self._normalize_article_id(r.get("article_id"))
                price_val = r.get("price")
                if price_val is not None and pd.isna(price_val):
                    price_val = None

                item = {
                    "t_dat": None if pd.isna(r.get("t_dat")) else str(r.get("t_dat").date()),
                    "article_id": aid,
                    "price": self._to_json_safe(float(price_val)) if price_val is not None else None,
                }
                item.update(self._item_meta(aid))
                item["image_url"] = self.image_url(aid)
                history_items.append(self._to_json_safe(item))

        summary: Dict[str, Any] = {}
        if self._user_hist_agg is not None and "customer_id" in self._user_hist_agg.columns:
            h = self._user_hist_agg.copy()
            h["customer_id"] = h["customer_id"].astype(str)
            match = h[h["customer_id"] == customer_id]
            if len(match) > 0:
                row = match.iloc[0].to_dict()
                keep = ["total_purchases", "last_purchase_date", "avg_price", "top_product_group_name"]
                summary = {k: self._to_json_safe(row.get(k)) for k in keep if k in row}

        return self._to_json_safe({"profile": profile, "summary": summary, "last_transactions": history_items})

    def _normalize_recommend_output(self, out: Any) -> Tuple[List[Any], bool]:
        if isinstance(out, tuple) and len(out) == 2:
            recs, is_fallback = out
            return list(recs), bool(is_fallback)

        is_fallback = bool(getattr(out, "is_fallback", getattr(out, "fallback", False)))

        for key in ["recommendations", "recs", "items", "results"]:
            if hasattr(out, key):
                recs = getattr(out, key)
                return list(recs), is_fallback

        if hasattr(out, "__iter__"):
            try:
                return list(out), is_fallback
            except Exception:
                pass

        raise TypeError("Output recommend() tidak bisa dinormalisasi jadi list rekomendasi")

    def recommend(self, customer_id: str, top_k: int = 10) -> Dict[str, Any]:
        customer_id = str(customer_id)

        if not hasattr(self.reco, "recommend"):
            raise RuntimeError("RecommenderService tidak punya method recommend(customer_id, top_k)")

        out = self.reco.recommend(customer_id=customer_id, top_k=top_k)
        recs, is_fallback = self._normalize_recommend_output(out)

        items: List[RecItem] = []
        for x in recs:
            if isinstance(x, dict):
                raw_aid = x.get("article_id") or x.get("item_id") or x.get("article") or ""
                aid = self._normalize_article_id(raw_aid)
                score = float(x.get("score", 0.0))
            else:
                raw_aid = getattr(x, "article_id", getattr(x, "item_id", getattr(x, "article", "")))
                aid = self._normalize_article_id(raw_aid)
                score = float(getattr(x, "score", 0.0))

            if not aid:
                continue

            if not math.isfinite(score):
                score = 0.0

            items.append(
                RecItem(
                    article_id=aid,
                    score=score,
                    image_url=self.image_url(aid),
                    meta=self._item_meta(aid),
                )
            )

        payload = {
            "customer_id": customer_id,
            "top_k": top_k,
            "is_fallback": bool(is_fallback),
            "recommendations": [i.__dict__ for i in items],
        }
        return self._to_json_safe(payload)

    def baseline(self, top_k: int = 10) -> Dict[str, Any]:
        recs: List[str]
        if hasattr(self.reco, "popularity"):
            recs = [self._normalize_article_id(x) for x in list(self.reco.popularity)[:top_k]]
        else:
            recs = []

        items: List[RecItem] = []
        for aid in recs:
            items.append(
                RecItem(
                    article_id=aid,
                    score=0.0,
                    image_url=self.image_url(aid),
                    meta=self._item_meta(aid),
                )
            )

        return self._to_json_safe({"top_k": top_k, "recommendations": [i.__dict__ for i in items]})
