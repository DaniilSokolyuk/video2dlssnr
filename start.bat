@echo off
setlocal
cd /d "%~dp0"

where python >nul 2>nul
if errorlevel 1 (
    echo Python 3 not found on PATH. Install it from python.org and retry.
    pause
    exit /b 1
)

set "VENV_PY=.venv\Scripts\python.exe"
if not exist "%VENV_PY%" (
    echo Creating virtual environment .venv ...
    python -m venv .venv
)
if not exist "%VENV_PY%" (
    echo Failed to create the virtual environment.
    pause
    exit /b 1
)

"%VENV_PY%" -m pip install --quiet --upgrade pip
"%VENV_PY%" -m pip install --quiet -r requirements.txt
if errorlevel 1 (
    echo Failed to install dependencies.
    pause
    exit /b 1
)

echo Starting UI - a browser tab will open at http://127.0.0.1:7860
"%VENV_PY%" app.py
pause
