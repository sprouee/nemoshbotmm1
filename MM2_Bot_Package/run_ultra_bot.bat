@echo off
chcp 65001 >nul
title ULTRA BOT v6.0 - Maximum Performance & Intelligence

echo.
echo ╔══════════════════════════════════════════════════════════════╗
echo ║                    🤖 ULTRA BOT v6.0                        ║
echo ║              Maximum Performance & Intelligence             ║
echo ╚══════════════════════════════════════════════════════════════╝
echo.
echo 🚀 Запуск ультра-бота...
echo.

REM Проверка Python
python --version >nul 2>&1
if errorlevel 1 (
    echo ❌ Python не найден! Установите Python 3.8+
    pause
    exit /b 1
)

REM Проверка зависимостей
echo 📦 Проверка зависимостей...
python -c "import ultralytics, torch, cv2, numpy" >nul 2>&1
if errorlevel 1 (
    echo ⚠️  Зависимости не установлены. Установка...
    pip install -r requirements.txt
    if errorlevel 1 (
        echo ❌ Ошибка установки зависимостей!
        pause
        exit /b 1
    )
)

REM Выбор режима
echo.
echo 🎮 Выберите режим:
echo [1] Стандартный режим (оптимизирован для вашей системы)
echo [2] Режим Хищника (с обнаружением игроков)
echo [3] Режим с визуализацией
echo [4] Тестирование производительности
echo [5] Режим разработчика (все опции)
echo.

set /p choice="Ваш выбор (1-5): "

if "%choice%"=="1" (
    echo.
    echo 🎯 Запуск в стандартном режиме...
    python run_ultra_bot.py
) else if "%choice%"=="2" (
    echo.
    echo 🦹 Запуск в режиме Хищника...
    python run_ultra_bot.py --predator
) else if "%choice%"=="3" (
    echo.
    echo 👁️  Запуск с визуализацией...
    python run_ultra_bot.py --show
) else if "%choice%"=="4" (
    echo.
    echo 🧪 Запуск тестирования производительности...
    python benchmark_ultra.py
) else if "%choice%"=="5" (
    echo.
    echo 🔧 Режим разработчика
    echo.
    echo Доступные опции:
    echo --predator          : Режим Хищника
    echo --show              : Визуализация
    echo --save-screenshots  : Сохранение скриншотов
    echo --no-ai             : Отключить ИИ
    echo --weights MODEL     : Выбор модели
    echo --conf VALUE        : Порог уверенности
    echo.
    set /p dev_args="Введите аргументы: "
    python run_ultra_bot.py %dev_args%
) else (
    echo ❌ Неверный выбор!
    pause
    exit /b 1
)

echo.
echo ╔══════════════════════════════════════════════════════════════╗
echo ║                     🎮 СЕССИЯ ЗАВЕРШЕНА                     ║
echo ╚══════════════════════════════════════════════════════════════╝
echo.
echo 📊 Статистика сохранена в логах
echo 🧠 Прогресс обучения сохранен
echo.

pause