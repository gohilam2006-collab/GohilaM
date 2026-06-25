@echo off
title Social Media Sentiment Analysis Dashboard
color 0A

echo.
echo =====================================================
echo   Social Media Sentiment Analysis Dashboard
echo =====================================================
echo.

:: Check Python
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] Python is not installed or not in PATH.
    echo Please install Python 3.8+ from https://python.org
    echo Make sure to check "Add Python to PATH" during install.
    pause
    exit /b 1
)

echo [1/3] Installing dependencies...
pip install -r requirements.txt --quiet
if %errorlevel% neq 0 (
    echo [WARN] Some packages may have failed. Trying continue...
)

echo [2/3] Running initial analysis pipeline...
python main.py
if %errorlevel% neq 0 (
    echo [WARN] Pipeline had issues, starting server anyway...
)

echo [3/3] Starting web server...
echo.
echo  ===========================================
echo   Open your browser: http://localhost:5000
echo  ===========================================
echo.
python app.py

pause
