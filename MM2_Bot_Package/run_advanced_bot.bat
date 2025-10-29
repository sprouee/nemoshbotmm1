@echo off
chcp 65001 >nul
title Продвинутый MM2 Бот - Ultra Edition

echo ========================================
echo   ПРОДВИНУТЫЙ MM2 БОТ - ULTRA EDITION
echo ========================================
echo.

python run_advanced_bot.py --weights weights/candies_v10.pt --conf 0.25 --show --mode balanced

if errorlevel 1 (
    echo.
    echo Ошибка запуска бота!
    echo Убедитесь, что установлены все зависимости:
    echo pip install ultralytics opencv-python torch
    pause
)
