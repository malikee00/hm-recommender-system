from __future__ import annotations

import os
import json
from pathlib import Path

import pandas as pd
import requests
import streamlit as st

from image_utils import ImageConfig, get_image_path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
FEATURE_STORE_DIR = PROJECT_ROOT / "data" / "feature_store"
ASSETS_DIR = Path(__file__).resolve().parent / "assets"
IMAGES_DIR = PROJECT_ROOT / "data" / "raw" / "hm" / "images_128_128"
PLACEHOLDER_PATH = ASSETS_DIR / "images" / "placeholder.jpg"

API_BASE_URL = os.getenv("API_BASE_URL", "http://127.0.0.1:8000").rstrip("/")


@st.cache_data
def load_item_features() -> pd.DataFrame:
    p = FEATURE_STORE_DIR / "item_features.parquet"
    df = pd.read_parquet(p)
    df["article_id"] = df["article_id"].astype(str)
    return df.set_index("article_id")


@st.cache_data
def load_user_history_agg() -> pd.DataFrame | None:
    p = FEATURE_STORE_DIR / "user_history_agg.parquet"
    if not p.exists():
        return None
    df = pd.read_parquet(p)
    df["customer_id"] = df["customer_id"].astype(str)
    return df


def build_title(row: pd.Series) -> str:
    pt = row.get("product_type_name", "")
    cg = row.get("colour_group_name", "")
    dep = row.get("department_name", "")
    pt = "" if pd.isna(pt) else str(pt)
    cg = "" if pd.isna(cg) else str(cg)
    dep = "" if pd.isna(dep) else str(dep)

    if pt and cg:
        return f"{pt} — {cg}"
    if pt and dep:
        return f"{pt} ({dep})"
    if pt:
        return pt
    return "Item"


def post_json(path: str, payload: dict) -> dict:
    url = f"{API_BASE_URL}{path}"
    r = requests.post(url, json=payload, timeout=60)
    r.raise_for_status()
    return r.json()


def render_cards(title: str, recs: list[dict], item_feat: pd.DataFrame, img_cfg: ImageConfig, cols: int = 5):
    st.subheader(title)
    if not recs:
        st.info("No results.")
        return

    grid = st.columns(cols)
    for i, rec in enumerate(recs):
        c = grid[i % cols]
        article_id = str(rec["article_id"])
        score = float(rec.get("score", 0.0))

        with c:
            img_path = get_image_path(article_id, img_cfg)
            st.image(str(img_path), use_container_width=True)

            row = item_feat.loc[article_id] if article_id in item_feat.index else None
            if row is not None:
                t = build_title(row)
                st.markdown(f"**{t}**")
                meta_line = []
                for k in ["department_name", "product_group_name", "section_name"]:
                    v = row.get(k, None)
                    if v is not None and not pd.isna(v):
                        meta_line.append(str(v))
                if meta_line:
                    st.caption(" • ".join(meta_line[:3]))
            else:
                st.markdown("**Item**")

            st.caption(f"article_id: {article_id}")
            st.caption(f"score: {score:.6f}")


st.set_page_config(page_title="H&M Recommender Demo", layout="wide")
st.title("H&M Personalized Recommender (Two-Tower + FAISS)")

with st.sidebar:
    st.header("Settings")
    customer_id = st.text_input("customer_id", value="")
    top_k = st.slider("top_k", min_value=1, max_value=20, value=10, step=1)
    run_btn = st.button("Recommend")

st.caption(f"API_BASE_URL: {API_BASE_URL}")

item_feat = load_item_features()
user_hist = load_user_history_agg()
img_cfg = ImageConfig(images_dir=IMAGES_DIR, placeholder_path=PLACEHOLDER_PATH)

if run_btn:
    if not customer_id.strip():
        st.warning("Please input a customer_id.")
        st.stop()

    if user_hist is not None:
        st.subheader("Your recent history")
        u = user_hist[user_hist["customer_id"] == str(customer_id)]
        if len(u) == 0:
            st.info("No history found for this user (may be cold-start).")
        else:
            show_cols = [c for c in ["customer_id", "top_product_group_name"] if c in u.columns]
            st.dataframe(u[show_cols].head(5), use_container_width=True)

    try:
        model_out = post_json("/recommend", {"customer_id": str(customer_id), "top_k": int(top_k)})
        base_out = post_json("/baseline", {"top_k": int(top_k)})
    except requests.HTTPError as e:
        st.error(f"API request failed: {e}")
        st.stop()

    is_fallback = bool(model_out.get("is_fallback", False))

    render_cards(
        "Personalized Recommendations (Model)",
        model_out.get("recommendations", []),
        item_feat,
        img_cfg,
        cols=5,
    )

    render_cards(
        "Popular Baseline (No ML)",
        base_out.get("recommendations", []),
        item_feat,
        img_cfg,
        cols=5,
    )

    if is_fallback:
        st.warning("User not found → showing popular items (cold-start fallback).")
