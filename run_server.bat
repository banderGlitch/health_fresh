@echo off
cd /d "%~dp0"
echo Starting AI-Analyzer server from: %CD%
echo.
python -m uvicorn api.main:app --reload --host 127.0.0.1 --port 8001 --log-level info
pause
