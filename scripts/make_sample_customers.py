from pathlib import Path
import pandas as pd
import re

PROJECT_ROOT = Path(__file__).resolve().parents[1]

FEATURE_STORE_DIR = PROJECT_ROOT / "data" / "feature_store"
OUT_DIR = PROJECT_ROOT / "data" / "reference"
OUT_PATH = OUT_DIR / "sample_customers.xlsx"

def normalize_article_id(x):
    if x is None:
        return ""
    s = str(x).strip()
    if re.fullmatch(r"\d+\.0", s):
        s = s.split(".")[0]
    digits = re.sub(r"\D", "", s)
    if digits:
        return digits.zfill(10)
    return s

def main(n_customers: int = 300, history_n: int = 5) -> None:
    interactions = pd.read_parquet(FEATURE_STORE_DIR / "interactions.parquet").copy()
    items = pd.read_parquet(FEATURE_STORE_DIR / "item_features.parquet").copy()

    if "article_id" not in interactions.columns or "customer_id" not in interactions.columns:
        raise ValueError("interactions.parquet harus punya kolom: customer_id, article_id")

    interactions["customer_id"] = interactions["customer_id"].astype(str)
    interactions["article_id"] = interactions["article_id"].apply(normalize_article_id)

    if "t_dat" in interactions.columns:
        interactions["t_dat"] = pd.to_datetime(interactions["t_dat"], errors="coerce")
        interactions = interactions.sort_values(["customer_id", "t_dat"], ascending=[True, False])
    else:
        interactions = interactions.sort_values(["customer_id"], ascending=[True])

    interactions["rank"] = interactions.groupby("customer_id").cumcount() + 1
    interactions = interactions[interactions["rank"] <= history_n].copy()

    if "article_id" in items.columns:
        items["article_id"] = items["article_id"].apply(normalize_article_id)
    if "product_group_name" not in items.columns:
        items["product_group_name"] = None

    item_map = items.drop_duplicates("article_id").set_index("article_id")["product_group_name"].to_dict()

    interactions["purchase_history"] = interactions["article_id"].map(item_map)
    interactions["purchase_history"] = interactions["purchase_history"].fillna("")

    wide = interactions.pivot_table(
        index="customer_id",
        columns="rank",
        values=["article_id", "purchase_history"],
        aggfunc="first",
        fill_value="",
    )

    wide.columns = [
        ("history_id_" + str(col[1])) if col[0] == "article_id" else ("purchase_history_" + str(col[1]))
        for col in wide.columns
    ]
    wide = wide.reset_index()

    for k in range(1, history_n + 1):
        hid = f"history_id_{k}"
        ph = f"purchase_history_{k}"
        if hid not in wide.columns:
            wide[hid] = ""
        if ph not in wide.columns:
            wide[ph] = ""

    ordered = ["customer_id"]
    for k in range(1, history_n + 1):
        ordered += [f"history_id_{k}", f"purchase_history_{k}"]
    wide = wide[ordered]

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(OUT_PATH, engine="openpyxl") as writer:
        wide.head(n_customers).to_excel(writer, index=False, sheet_name="sample_customers")

    print(f"Saved: {OUT_PATH}")

if __name__ == "__main__":
    main()
