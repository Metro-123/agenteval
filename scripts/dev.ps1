# Build the dashboard and serve it, all from one command.
# Run from the repo root:  scripts\dev.ps1
$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root

$env:PYTHONPATH = Join-Path $root "src"

Write-Host "==> building web bundle" -ForegroundColor Cyan
if (-not (Test-Path (Join-Path $root "web\node_modules"))) {
    npm.cmd --prefix web install
}
npm.cmd --prefix web run build

Write-Host "==> starting API + dashboard on http://127.0.0.1:8000" -ForegroundColor Cyan
Write-Host "    (Ctrl+C to stop)" -ForegroundColor DarkGray
python -m uvicorn agenteval.api:app --host 127.0.0.1 --port 8000
