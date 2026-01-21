from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]

FEATURE_STORE_DIR = PROJECT_ROOT / "data" / "feature_store"
OUT_DIR = PROJECT_ROOT / "data" / "reference"
OUT_PATH = OUT_DIR / "sample_customers.csv"

def main(n_customers: int = 300, history_n: int = 5) -> None:
    interactions = pd.read_parquet(FEATURE_STORE_DIR / "interactions.parquet").copy()
    cols = ["customer_id", "article_id"]
    if "t_dat" in interactions.columns:
        cols.append("t_dat")
    interactions = interactions[cols]

    if "t_dat" in interactions.columns:
        interactions = interactions.sort_values(["customer_id", "t_dat"], ascending=[True, False])
    else:
        interactions = interactions.sort_values(["customer_id"], ascending=[True])

    interactions["rank"] = interactions.groupby("customer_id").cumcount() + 1
    interactions = interactions[interactions["rank"] <= history_n]

    hist = (
        interactions.groupby("customer_id")["article_id"]
        .apply(lambda s: "|".join([str(x) for x in s.tolist()]))
        .reset_index()
        .rename(columns={"article_id": "history"})
    )

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    hist.head(n_customers).to_csv(OUT_PATH, index=False)
    print(f"Saved: {OUT_PATH}")

if __name__ == "__main__":
    main()
