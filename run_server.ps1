# AI-Analyzer - Start server from project root
# Run: .\run_server.ps1

$projectRoot = $PSScriptRoot
Set-Location $projectRoot

Write-Host "Starting server from: $projectRoot" -ForegroundColor Green
Write-Host "Health check: http://127.0.0.1:8001/health" -ForegroundColor Cyan
Write-Host ""

python -m uvicorn api.main:app --reload --host 127.0.0.1 --port 8001 --log-level info
