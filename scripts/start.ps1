# Start the dashboard in a detached process that survives this shell.
#   scripts\start.ps1          -> background, writes scripts\server.log
#   scripts\start.ps1 -Stop    -> stop it
param([switch]$Stop)

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
$log = Join-Path $PSScriptRoot "server.log"
$url = "http://127.0.0.1:8000"

function Get-Server {
    Get-CimInstance Win32_Process -Filter "Name='python.exe'" -ErrorAction SilentlyContinue |
        Where-Object { $_.CommandLine -like "*uvicorn*agenteval.api*" }
}

if ($Stop) {
    $p = Get-Server
    if ($p) { $p | ForEach-Object { Stop-Process -Id $_.ProcessId -Force; "stopped pid $($_.ProcessId)" } }
    else { "not running" }
    exit 0
}

if (Get-Server) { "already running at $url"; exit 0 }

$env:PYTHONPATH = Join-Path $root "src"
Start-Process -FilePath "python" `
    -ArgumentList "-m", "uvicorn", "agenteval.api:app",
                  "--host", "127.0.0.1", "--port", "8000", "--log-level", "info" `
    -WorkingDirectory $root -WindowStyle Hidden -RedirectStandardError $log

for ($i = 0; $i -lt 40; $i++) {
    Start-Sleep -Milliseconds 500
    try {
        $r = Invoke-WebRequest "$url/api/health" -UseBasicParsing -TimeoutSec 2
        if ($r.StatusCode -eq 200) { "up at $url"; exit 0 }
    } catch { }
}
"failed to start; see $log"
Get-Content $log -Tail 20
exit 1
