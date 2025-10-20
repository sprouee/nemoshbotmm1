@echo off
echo ========================================
echo MM2 Bot - Installation
echo ========================================
echo.

echo [1/3] Checking Python...
python --version
if %errorlevel% neq 0 (
    echo ERROR: Python not found!
    echo Please install Python 3.11 from https://www.python.org/downloads/
    pause
    exit /b 1
)

echo.
echo [2/3] Installing dependencies...
python -m pip install --upgrade pip
python -m pip install ultralytics opencv-python pywin32

echo.
echo [3/3] Checking installation...
python -c "import ultralytics; import cv2; import win32api; print('OK')"
if %errorlevel% neq 0 (
    echo ERROR: Installation failed!
    pause
    exit /b 1
)

echo.
echo ========================================
echo Installation complete!
echo ========================================
echo.
echo Next steps:
echo 1. Make sure you have a model in weights/ folder
echo 2. Run MM2_Bot_Launcher.exe
echo.
pause

