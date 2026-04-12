$ErrorActionPreference = "Stop"

Write-Host "============================================================"
Write-Host "  Step 1/2 -- Caching DINOv2 features"
Write-Host "============================================================"
poetry run python data/cache_dino_features.py --batch-size 8 --skip-existing

Write-Host ""
Write-Host "============================================================"
Write-Host "  Step 2/2 -- Training"
Write-Host "============================================================"
poetry run python training/train.py
