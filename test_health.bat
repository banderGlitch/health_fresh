@echo off
cd /d "%~dp0"
echo Testing health endpoint (bypasses browser cache)...
python -c "import urllib.request; print(urllib.request.urlopen('http://127.0.0.1:8000/health').read().decode())"
echo.
pause
