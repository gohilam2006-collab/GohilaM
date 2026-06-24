@echo off
title Bank Marketing Decision Tree Classifier

echo ================================================
echo   Bank Marketing Decision Tree Classifier
echo ================================================
echo.

:: Check Python
python --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python not found. Please install Python 3.8+ from https://python.org
    pause
    exit /b 1
)

:: Install dependencies if needed
echo [1/3] Installing dependencies...
pip install -r requirements.txt --quiet

:: Run pipeline to train model and generate visuals
echo [2/3] Training model and generating visuals...
python main.py

:: Start Flask server
echo [3/3] Starting web server at http://localhost:5000
echo.
echo  Open your browser and go to: http://localhost:5000
echo  Press Ctrl+C to stop the server.
echo.
start "" http://localhost:5000
python app.py

pause
