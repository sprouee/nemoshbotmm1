@echo off
echo ========================================
echo    Улучшенный бот для поиска мячиков
echo           Версия 2.0 Enhanced
echo ========================================
echo.

REM Проверяем наличие Python
python --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python не найден! Установите Python 3.8+
    pause
    exit /b 1
)

REM Проверяем наличие файла бота
if not exist "run_enhanced_bot.py" (
    echo [ERROR] Файл run_enhanced_bot.py не найден!
    pause
    exit /b 1
)

REM Проверяем наличие весов
if not exist "weights\ball_rtdetr.pt" (
    echo [WARNING] Веса ball_rtdetr.pt не найдены!
    echo [INFO] Проверяем другие веса...
    if exist "weights\ball_v10.pt" (
        echo [INFO] Найден ball_v10.pt, используем его
        set WEIGHTS=weights\ball_v10.pt
    ) else (
        echo [ERROR] Никакие веса не найдены!
        echo [INFO] Поместите файл .pt в папку weights\
        pause
        exit /b 1
    )
) else (
    set WEIGHTS=weights\ball_rtdetr.pt
)

echo [INFO] Используем веса: %WEIGHTS%
echo [INFO] Запуск улучшенного бота...
echo.
echo [TIP] Нажмите Ctrl+C для остановки
echo [TIP] Убедитесь, что Roblox запущен и в фокусе
echo.

REM Запускаем бота
python run_enhanced_bot.py --weights %WEIGHTS% --conf 0.25

echo.
echo [EXIT] Бот остановлен
pause
