# Run the test suite with the source tree on the path.
# Run from the repo root:  scripts\test.ps1
$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root

$env:PYTHONPATH = Join-Path $root "src"
python -m pytest tests @args
