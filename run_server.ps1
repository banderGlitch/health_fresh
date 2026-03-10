# AI-Analyzer - Start server from project root
# Run: .\run_server.ps1

$projectRoot = $PSScriptRoot
Set-Location $projectRoot

Write-Host "Starting server from: $projectRoot" -ForegroundColor Green
Write-Host "Health check: http://127.0.0.1:8002/health" -ForegroundColor Cyan
Write-Host ""

# Kill any process already using port 8002
$pids = (Get-NetTCPConnection -LocalPort 8002 -ErrorAction SilentlyContinue).OwningProcess | Sort-Object -Unique
foreach ($p in $pids) { Stop-Process -Id $p -Force -ErrorAction SilentlyContinue }
Start-Sleep -Seconds 2

$env:PYTHONUNBUFFERED = "1"
python -m uvicorn api.main:app --host 127.0.0.1 --port 8002 --log-level info
