# Run only training
Set-Location $PSScriptRoot\..
python -m ml.recommender.train `
  --feature_store_dir data/feature_store `
  --registry_dir ml/registry/recommender `
  --embedding_dim 64 `
  --epochs 2 `
  --batch_size 2048 `
  --max_interactions 2000000
