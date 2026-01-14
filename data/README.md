# Data Policy (H&M Personalized Fashion Recommendations)

This project uses the **H&M Personalized Fashion Recommendations** dataset from Kaggle.

Because the dataset is large, **raw files must NOT be committed** to GitHub.
We keep raw data locally (or in Colab) and generate clean + feature-store outputs via the pipeline.

---

## Folder layout

Inside `ml_hm_project/`:

- `data/raw/hm/reference/`  
  Raw CSVs downloaded from Kaggle (do not commit)

- `data/processed/`  
  Canonical cleaned datasets (Parquet) produced by `transform.py` (Phase 2.3)

- `data/feature_store/`  
  Final tables used for ML training + serving (Parquet) produced by `build_features.py` (Phase 2.4)

---

## Required raw files (minimum)

Place these files here:

`data/raw/hm/reference/`
- `customers.csv`
- `articles.csv`
- `transactions_train.csv`

> Note: Kaggle login is required to download the dataset.

---

## Git policy (important)

✅ Allowed to commit:
- This `data/README.md`
- Small synthetic samples for testing (optional)

🚫 Do NOT commit:
- `customers.csv`, `articles.csv`, `transactions_train.csv`
- Any large raw datasets

Recommended `.gitignore` entries:
data/raw/
data/processed/
data/feature_store/
*.parquet


