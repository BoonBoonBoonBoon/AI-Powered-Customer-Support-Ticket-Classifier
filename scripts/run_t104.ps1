param(
  [string]$DataPath = "data\enriched_customer_tickets.csv",
  [string]$ModelName = "distilbert-base-uncased",
  [string]$Version = "t1.0.4"
)

$ErrorActionPreference = "Stop"

# Paths
$repoRoot = Split-Path -Parent $PSScriptRoot
$python = Join-Path $repoRoot ".venv\Scripts\python.exe"
$versionDir = Join-Path $repoRoot ("models\transformers\" + $Version)

# Ensure output directories exist
New-Item -ItemType Directory -Force -Path $versionDir | Out-Null

# Training args (tuned)
$trainArgs = @(
  (Join-Path $repoRoot "scripts\train_transformer.py"),
  "--data", (Join-Path $repoRoot $DataPath),
  "--model-name", $ModelName,
  "--epochs", "5",
  "--batch-size", "16",
  "--grad-accum", "2",
  "--lr", "3e-5",
  "--weight-decay", "0.01",
  "--warmup-ratio", "0.1",
  "--max-len", "256",
  "--output-version", $Version,
  # Apply leak-guard patterns explicitly
  "--exclude-pattern", "__department_[a-z0-9_]+",
  "--exclude-pattern", "__dept_[a-z0-9_]+",
  "--exclude-pattern", "__type_[a-z0-9_]+",
  # Loss/selection
  "--class-weight-priority", "auto",
  "--class-weight-department", "auto",
  "--label-smoothing", "0.05",
  "--select-metric", "priority",
  "--loss-weight-priority", "2.0",
  "--loss-weight-department", "0.5"
)

# Train and capture logs reliably
$trainLog = Join-Path $versionDir "train.log"
$trainErrLog = Join-Path $versionDir "train.err.log"
$quotedArgs = $trainArgs | ForEach-Object { if ($_ -match '\s') { '"' + $_ + '"' } else { $_ } }
$argString = [string]::Join(' ', $quotedArgs)
$p = Start-Process -FilePath $python -ArgumentList $argString -NoNewWindow -PassThru -Wait -RedirectStandardOutput $trainLog -RedirectStandardError $trainErrLog
if ($p.ExitCode -ne 0) { Write-Host "Training failed with exit code $($p.ExitCode)"; exit $p.ExitCode }

# Emit manifest
$emitLog = Join-Path $versionDir "emit_manifest.log"
& $python (Join-Path $repoRoot "scripts\emit_manifest.py") "--version" $Version 2>&1 | Tee-Object -FilePath $emitLog
if ($LASTEXITCODE -ne 0) { Write-Warning "emit_manifest.py returned non-zero exit code $LASTEXITCODE" }

# Compare models and save log
$compareLog = Join-Path $versionDir "compare.log"
& $python (Join-Path $repoRoot "scripts\compare_models.py") 2>&1 | Tee-Object -FilePath $compareLog
if ($LASTEXITCODE -ne 0) { Write-Warning "compare_models.py returned non-zero exit code $LASTEXITCODE" }

Write-Host "t1.0.4 pipeline complete: logs at $versionDir"