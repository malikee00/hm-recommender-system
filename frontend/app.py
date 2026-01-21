import gradio as gr
import pandas as pd
import requests
import os
from pathlib import Path

# Image base URL from Cloudflare
IMAGE_BASE_URL = "https://pub-37ed1d4c43e6485da7b4aec032e35912.r2.dev/hm-images"

# File paths
PROJECT_ROOT = Path(__file__).resolve().parents[1]
FEATURE_STORE_DIR = PROJECT_ROOT / "data" / "feature_store"
IMAGES_DIR = PROJECT_ROOT / "data" / "raw" / "hm" / "images_128_128"
PLACEHOLDER_PATH = PROJECT_ROOT / "assets" / "images" / "placeholder.jpg"

# Function to get image URL from Cloudflare
def get_image_url(article_id: str) -> str:
    return f"{IMAGE_BASE_URL}/{article_id}.jpg"

# Load item features (using sample CSV)
def load_item_features():
    item_features_file = FEATURE_STORE_DIR / "item_features.parquet"
    df = pd.read_parquet(item_features_file)
    df["article_id"] = df["article_id"].astype(str)
    return df.set_index("article_id")

# Load user history from CSV
def load_user_history_agg():
    user_history_file = FEATURE_STORE_DIR / "user_history_agg.parquet"
    if not user_history_file.exists():
        return None
    df = pd.read_parquet(user_history_file)
    df["customer_id"] = df["customer_id"].astype(str)
    return df

# Function to download sample CSV
def download_sample_csv():
    sample_data = {
        "customer_id": ["C123", "C124", "C125"],
        "top_product_group_name": ["Shirts", "Jeans", "Jackets"]
    }
    df = pd.DataFrame(sample_data)
    df.to_csv("sample_customer_data.csv", index=False)
    return "sample_customer_data.csv"

# Function to make recommendations based on customer ID
def recommend(customer_id, top_k=5):
    # In a real scenario, replace this with a call to your trained model
    recommendations = [
        {"article_id": "A123", "score": 0.95},
        {"article_id": "A456", "score": 0.87},
        {"article_id": "A789", "score": 0.75},
        {"article_id": "A101", "score": 0.88},
        {"article_id": "A202", "score": 0.91},
    ]
    return recommendations[:top_k]

# Function to build title from item features
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

# Function to render recommendation cards
def render_cards(title: str, recs: list, item_feat: pd.DataFrame, cols=5):
    card_list = []
    for rec in recs:
        article_id = rec.get("article_id", "")
        score = rec.get("score", 0.0)
        image_url = get_image_url(article_id)
        
        if article_id in item_feat.index:
            row = item_feat.loc[article_id]
            product_name = build_title(row)
            meta = [str(row.get(k, "")) for k in ["department_name", "product_group_name", "section_name"] if row.get(k, None)]
            description = " • ".join(meta[:3]) if meta else "No description available"
        else:
            product_name = "Item"
            description = "No description available"
        
        card_list.append({
            "image": image_url,
            "article_id": article_id,
            "score": score,
            "product_name": product_name,
            "description": description
        })
    return card_list

# Function to get user history
def get_user_history(customer_id, user_hist):
    if user_hist is not None:
        user_data = user_hist[user_hist["customer_id"] == str(customer_id)]
        if len(user_data) == 0:
            return "No history found for this user (may be cold-start)."
        return user_data[["customer_id", "top_product_group_name"]].head(5).to_dict(orient="records")
    return None

# Gradio interface function
def gradio_interface(customer_id, top_k=5):
    # Load item features and user history
    item_feat = load_item_features()
    user_hist = load_user_history_agg()

    # Get user history
    user_history = get_user_history(customer_id, user_hist)
    
    # Get recommendations
    recommendations = recommend(customer_id, top_k)
    cards = render_cards("Recommendations", recommendations, item_feat)

    return user_history, cards

# Creating Gradio interface
iface = gr.Interface(
    fn=gradio_interface,
    inputs=[
        gr.Textbox(label="Customer ID", placeholder="Enter customer ID here"),
        gr.Slider(minimum=1, maximum=10, step=1, label="Number of Recommendations", value=5)
    ],
    outputs=[
        gr.JSON(label="User History"),
        gr.JSON(label="Recommendations")
    ],
    title="H&M Personalized Recommender"
)

iface.launch()
