$ErrorActionPreference = "Stop"

Write-Host "=== H&M Pipeline Runner (Phase 2.1 - 2.4) ===" -ForegroundColor Cyan

# Ensure we run from project root
$PROJECT_ROOT = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location (Join-Path $PROJECT_ROOT "..")
Write-Host "Project root: $(Get-Location)" -ForegroundColor DarkGray

# Optional: ensure venv is activated
if (-not (Test-Path ".\.venv\Scripts\python.exe")) {
    Write-Host "ERROR: .venv not found. Create venv and install deps first." -ForegroundColor Red
    exit 1
}
$PY = ".\.venv\Scripts\python.exe"

# Step 2.1 fetch/check raw files exists
Write-Host "`n[1/4] fetch_data (check raw files exist)" -ForegroundColor Yellow
& $PY -m pipelines_spark.fetch_data --raw-dir data/raw/hm/reference

# Step 2.2 schema validation
Write-Host "`n[2/4] validate_schema" -ForegroundColor Yellow
& $PY -m pipelines_spark.validate_schema --raw-dir data/raw/hm/reference

# Step 2.3 transform
Write-Host "`n[3/4] transform (canonical parquet)" -ForegroundColor Yellow
& $PY -m pipelines_spark.transform --raw-dir data/raw/hm/reference --processed-dir data/processed

# Step 2.4 build features
Write-Host "`n[4/4] build_features (feature_store parquet)" -ForegroundColor Yellow
& $PY -m pipelines_spark.build_features --processed-dir data/processed --feature-store-dir data/feature_store

Write-Host "`n [DONE] Pipeline completed." -ForegroundColor Green

# Final checks: required outputs
$required = @(
    "data/feature_store/interactions.parquet",
    "data/feature_store/user_features.parquet",
    "data/feature_store/item_features.parquet",
    "data/feature_store/user_history_agg.parquet",
    "data/feature_store/item_popularity.parquet"
)

Write-Host "`nChecking outputs..." -ForegroundColor Cyan
foreach ($p in $required) {
    if (-not (Test-Path $p)) {
        Write-Host " Missing output: $p" -ForegroundColor Red
        exit 1
    } else {
        Write-Host " Found: $p" -ForegroundColor Green
    }
}

Write-Host "`n [DONE] All outputs present." -ForegroundColor Green
