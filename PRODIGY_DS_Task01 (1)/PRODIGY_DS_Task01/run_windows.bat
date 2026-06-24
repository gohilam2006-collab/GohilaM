@echo off
title PRODIGY DS Task 01 - Distribution Visualizer
color 0B

echo.
echo  ============================================
echo   PRODIGY_DS_Task01  ^|  Distribution Visualizer
echo  ============================================
echo.

:: Check Python
python --version >nul 2>&1
if errorlevel 1 (
    echo  [ERROR] Python not found. Please install Python 3.10+
    echo  Download: https://www.python.org/downloads/
    pause
    exit /b 1
)

echo  [1/3] Python found
echo  [2/3] Installing dependencies...
pip install -r requirements.txt --quiet

echo  [3/3] Starting server...
echo.
echo  Open your browser:  http://127.0.0.1:5000
echo.
echo  Press Ctrl+C to stop.
echo  ============================================
echo.

python main.py

pause
