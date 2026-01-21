from pathlib import Path
import os
from dotenv import load_dotenv

# load .env from project root
PROJECT_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(PROJECT_ROOT / ".env")

# === APP ===
APP_NAME = os.getenv("APP_NAME", "H&M Local Marketplace Demo")
APP_VERSION = os.getenv("APP_VERSION", "0.1")
DEFAULT_TOP_K = int(os.getenv("DEFAULT_TOP_K", 10))

# === PATHS ===
FEATURE_STORE_DIR = (PROJECT_ROOT / os.getenv(
    "FEATURE_STORE_DIR", "data/feature_store"
)).resolve()

REGISTRY_RUN_DIR = (PROJECT_ROOT / os.getenv(
    "REGISTRY_RUN_DIR", "ml/registry/recommender/run_2m_e2_v4_bpr"
)).resolve()

IMAGES_DIR = (PROJECT_ROOT / os.getenv(
    "IMAGES_DIR", "data/raw/hm/images_128_128"
)).resolve()
