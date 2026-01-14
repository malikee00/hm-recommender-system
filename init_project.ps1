# ==========================================
# Init folder structure INSIDE existing H&M
# ==========================================

$folders = @(
    # data
    "data",
    "data/raw",
    "data/raw/hm",
    "data/raw/hm/reference",
    "data/processed",
    "data/feature_store",

    # features
    "features",

    # ml
    "ml",
    "ml/recommender",
    "ml/registry",
    "ml/registry/recommender",
    "ml/reports",
    "ml/reports/recommender",

    # app (FastAPI)
    "app",

    # demo UI (Streamlit)
    "demo_ui",

    # ops
    "ops",

    # docs
    "docs",
    "docs/model_cards"
)

foreach ($folder in $folders) {
    if (-not (Test-Path $folder)) {
        New-Item -ItemType Directory -Path $folder | Out-Null
        Write-Host "Created: $folder"
    } else {
        Write-Host "Exists : $folder"
    }
}

Write-Host ""
Write-Host "[DONE] Folder structure initialized inside H&M."
