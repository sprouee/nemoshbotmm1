@echo off
chcp 65001 > nul
setlocal

echo ========================================
echo    Enhanced Ball Hunter Bot v2.0
echo ========================================
echo.
echo Управление остановкой:
echo - Ctrl+C - мягкая остановка (рекомендуется)
echo - Ctrl+Break - принудительная остановка
echo - Закрытие окна - остановка
echo - stop_bot.bat - экстренная остановка
echo.

:: Проверяем, есть ли аргументы
if "%~1"=="" (
    echo Использование: %~nx0 [путь_к_весам] [--conf N] [--no-adaptive] [--show] [--save-screenshots]
    echo.
    echo Примеры:
    echo   %~nx0 weights/candies_v10.pt --conf 0.22
    echo   %~nx0 weights/ball_rtdetr.pt --conf 0.25 --show
    echo   %~nx0 weights/ball_v10.pt --save-screenshots
    echo.
    goto :eof
)

:: Переходим в директорию скрипта
cd /d "%~dp0"

:: Запускаем Python скрипт с переданными аргументами
echo Запуск бота...
python run_enhanced_bot.py %*

echo.
echo Бот завершил работу
pause
