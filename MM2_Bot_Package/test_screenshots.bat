@echo off
cd /d "%~dp0"
echo Запуск бота с сохранением скриншотов...
python run_enhanced_bot.py --save-screenshots
pause
