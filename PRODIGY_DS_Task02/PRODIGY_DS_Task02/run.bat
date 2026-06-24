@echo off
title PRODIGY DS Task-02 - Titanic EDA Dashboard
color 0A

echo.
echo  ============================================================
echo   Titanic EDA Dashboard - PRODIGY InfoTech Data Science
echo  ============================================================
echo.

:: Check Python
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo  [ERROR] Python not found. Please install Python 3.10+ from python.org
    pause
    exit /b
)

echo  [1/2] Installing dependencies...
pip install -r requirements.txt --quiet

echo  [2/2] Starting Flask server...
echo.
echo  ============================================================
echo   Open your browser at:  http://localhost:5000
echo   Press Ctrl+C to stop the server
echo  ============================================================
echo.

python app.py

pause
