from __future__ import annotations

import os
import time
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
    pt = "" if pd.isna(row.get("product_type_name", "")) else str(row.get("product_type_name", ""))
    cg = "" if pd.isna(row.get("colour_group_name", "")) else str(row.get("colour_group_name", ""))
    dep = "" if pd.isna(row.get("department_name", "")) else str(row.get("department_name", ""))

    if pt and cg:
        return f"{pt} — {cg}"
    if pt and dep:
        return f"{pt} ({dep})"
    if pt:
        return pt
    return "Item"


def post_json(path: str, payload: dict, retries: int = 3, timeout_s: int = 180) -> dict:
    url = f"{API_BASE_URL}{path}"
    last_err: Exception | None = None

    for _ in range(retries):
        try:
            r = requests.post(url, json=payload, timeout=timeout_s)
            r.raise_for_status()
            return r.json()
        except (requests.exceptions.ReadTimeout, requests.exceptions.ConnectionError) as e:
            last_err = e
            time.sleep(1.5)
        except requests.HTTPError as e:
            try:
                detail = r.text
            except Exception:
                detail = ""
            raise requests.HTTPError(f"{e} | body={detail[:500]}") from e

    raise last_err if last_err is not None else RuntimeError("Unknown request error")


def render_cards(title: str, recs: list[dict], item_feat: pd.DataFrame, img_cfg: ImageConfig, cols: int = 5):
    st.subheader(title)

    if not recs:
        st.info("No results.")
        return

    grid = st.columns(cols)

    for i, rec in enumerate(recs):
        c = grid[i % cols]
        article_id = str(rec.get("article_id", ""))
        score = float(rec.get("score", 0.0))

        with c:
            img_path = get_image_path(article_id, img_cfg)
            st.image(str(img_path), use_container_width=True)

            if article_id in item_feat.index:
                row = item_feat.loc[article_id]
                st.markdown(f"**{build_title(row)}**")

                meta = []
                for k in ["department_name", "product_group_name", "section_name"]:
                    v = row.get(k, None)
                    if v is not None and not pd.isna(v):
                        meta.append(str(v))
                if meta:
                    st.caption(" • ".join(meta[:3]))
            else:
                st.markdown("**Item**")

            st.caption(f"article_id: {article_id}")
            st.caption(f"score: {score:.6f}")


st.set_page_config(page_title="H&M Recommender Demo", layout="wide")
st.title("H&M Personalized Recommender (Two-Tower + FAISS)")
st.caption(f"API_BASE_URL: {API_BASE_URL}")

with st.sidebar:
    st.header("Settings")
    customer_id = st.text_input("customer_id", value="")
    top_k = st.slider("top_k", min_value=1, max_value=20, value=10, step=1)
    run_btn = st.button("Recommend")


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
        api_k = max(int(top_k), 50)
        model_out = post_json("/recommend", {"customer_id": str(customer_id), "top_k": api_k})
        base_out = post_json("/baseline", {"top_k": api_k})
    except Exception as e:
        st.error(str(e))
        st.stop()

    model_recs = model_out.get("recommendations", [])[: int(top_k)]
    base_recs = base_out.get("recommendations", [])[: int(top_k)]
    is_fallback = bool(model_out.get("is_fallback", False))

    render_cards("Personalized Recommendations (Model)", model_recs, item_feat, img_cfg, cols=5)
    render_cards("Popular Baseline (No ML)", base_recs, item_feat, img_cfg, cols=5)

    if is_fallback:
        st.warning("User not found → showing popular items (cold-start fallback).")
