@echo off
cd /d "%~dp0"
echo Starting AI-Analyzer server from: %CD%
echo.
rem Kill any process already using port 8002
for /f "tokens=5" %%a in ('netstat -ano ^| findstr :8002 ^| findstr LISTENING') do taskkill /F /PID %%a 2>nul
timeout /t 2 /nobreak >nul
set PYTHONUNBUFFERED=1
python -m uvicorn api.main:app --host 127.0.0.1 --port 8002 --log-level info
pause
