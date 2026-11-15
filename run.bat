@echo off
setlocal enabledelayedexpansion

title Telecom Churn Intelligence - Unified Launcher
color 0A

echo =====================================================================
echo           TELECOM CUSTOMER CHURN INTELLIGENCE SYSTEM
echo           XGBoost Production Pipeline - Unified Launcher
echo =====================================================================
echo.

cd /d "%~dp0"

REM 1. Check or Create Virtual Environment
if not exist ".venv\Scripts\python.exe" (
    echo [*] Virtual environment not found. Initializing .venv...
    where python >nul 2>nul
    if %ERRORLEVEL% equ 0 (
        python -m venv .venv
    ) else (
        where py >nul 2>nul
        if !ERRORLEVEL! equ 0 (
            py -m venv .venv
        ) else (
            color 0C
            echo [ERROR] Python not found in system PATH. Please install Python 3.10+
            pause
            exit /b 1
        )
    )
    echo [*] .venv created successfully.
) else (
    echo [*] Existing .venv detected.
)

REM 2. Activate Virtual Environment
call ".venv\Scripts\activate.bat"

REM 3. Verify / Install Dependencies
echo [*] Checking dependencies from app\backend\requirements.txt...
python -m pip install -r app\backend\requirements.txt --quiet --disable-pip-version-check
if %ERRORLEVEL% neq 0 (
    echo [WARNING] Dependency check encountered a warning, continuing launch...
)

REM 4. Launch Browser in Background
echo [*] Launching dashboard in default browser at http://localhost:8000 ...
start "" cmd /c "timeout /t 2 /nobreak >nul & start http://localhost:8000"

REM 5. Start Uvicorn Server
echo.
echo =====================================================================
echo  FastAPI Server running on: http://localhost:8000
echo  Web Dashboard URL:        http://localhost:8000
echo  Interactive API Docs:     http://localhost:8000/docs
echo  Press Ctrl+C to stop the server at any time.
echo =====================================================================
echo.

cd app\backend
python -m uvicorn main:app --host 127.0.0.1 --port 8000

pause
