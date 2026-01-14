# Run training + evaluation
Set-Location $PSScriptRoot\..
python -m ml.recommender.train `
  --feature_store_dir data/feature_store `
  --registry_dir ml/registry/recommender

python -m ml.recommender.evaluate `
  --feature_store_dir data/feature_store `
  --registry_dir ml/registry/recommender `
  --reports_dir ml/reports/recommender `
  --ks 5,10,20