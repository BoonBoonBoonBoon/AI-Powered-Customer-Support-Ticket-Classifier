param(
  [string]$DataPath = "data\enriched_customer_tickets.csv",
  [string]$ModelName = "distilbert-base-uncased",
  [string]$Version = "t1.0.3"
)

$ErrorActionPreference = "Stop"

# Paths
$repoRoot = Split-Path -Parent $PSScriptRoot
$python = Join-Path $repoRoot ".venv\Scripts\python.exe"
$versionDir = Join-Path $repoRoot ("models\transformers\" + $Version)
$logsDir = $versionDir

# Ensure output directories exist
New-Item -ItemType Directory -Force -Path $versionDir | Out-Null

# Training args
$trainArgs = @(
  (Join-Path $repoRoot "scripts\train_transformer.py"),
  "--data", (Join-Path $repoRoot $DataPath),
  "--model-name", $ModelName,
  "--epochs", "3",
  "--batch-size", "16",
  "--max-len", "256",
  "--output-version", $Version,
  "--exclude-pattern", "__type_[a-z0-9_]+",
  "--class-weight-priority", "auto",
  "--class-weight-department", "auto",
  "--label-smoothing", "0.05",
  "--select-metric", "sum"
)

# Train and tee logs
$trainLog = Join-Path $logsDir "train.log"
& $python @trainArgs 2>&1 | Tee-Object -FilePath $trainLog
if ($LASTEXITCODE -ne 0) { Write-Host "Training failed with exit code $LASTEXITCODE"; exit $LASTEXITCODE }

# Emit manifest
$emitLog = Join-Path $logsDir "emit_manifest.log"
& $python (Join-Path $repoRoot "scripts\emit_manifest.py") "--version" $Version 2>&1 | Tee-Object -FilePath $emitLog
if ($LASTEXITCODE -ne 0) { Write-Warning "emit_manifest.py returned non-zero exit code $LASTEXITCODE" }

# Compare models
$compareLog = Join-Path $logsDir "compare.log"
& $python (Join-Path $repoRoot "scripts\compare_models.py") 2>&1 | Tee-Object -FilePath $compareLog
if ($LASTEXITCODE -ne 0) { Write-Warning "compare_models.py returned non-zero exit code $LASTEXITCODE" }

Write-Host "Pipeline complete: logs at $logsDir"