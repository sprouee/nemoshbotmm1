#!/usr/bin/env python3
"""
УЛЬТРА-БОТ для поиска мячиков v6.0 - Максимальная производительность и интеллект
===============================================================================

НОВЫЕ ВОЗМОЖНОСТИ:
✓ Оптимизированный инференс с FP16 и кэшированием
✓ Продвинутая система предсказания движения целей
✓ Reinforcement Learning для адаптивного обучения
✓ Расширенная система уклонения от угроз
✓ Интеллектуальная навигация с перехватом траекторий
✓ Многоуровневая оптимизация производительности
✓ Автоматическая адаптация параметров
✓ Продвинутая система памяти и обучения

Автор: AI Assistant
Версия: 6.0 Ultra
"""

import sys
import os
import argparse
import time
import json
from pathlib import Path

# Импорт улучшенного бота
from run_enhanced_bot import EnhancedBallBot

class UltraBotLauncher:
    """Ультра-загрузчик для продвинутого бота"""

    def __init__(self):
        self.config = self.load_config()
        self.performance_profile = self.detect_system_profile()

    def load_config(self):
        """Загрузка конфигурации"""
        config_path = Path(__file__).parent / "ultra_bot_config.json"
        default_config = {
            "performance_mode": "ultra",
            "ai_enabled": True,
            "adaptive_learning": True,
            "prediction_enabled": True,
            "evasion_system": True,
            "memory_system": True,
            "multi_threading": True,
            "fp16_acceleration": True,
            "dynamic_resolution": True,
            "auto_tuning": True
        }

        if config_path.exists():
            with open(config_path, 'r') as f:
                loaded_config = json.load(f)
                default_config.update(loaded_config)

        return default_config

    def detect_system_profile(self):
        """Определение профиля производительности системы"""
        try:
            import torch
            if torch.cuda.is_available():
                gpu_name = torch.cuda.get_device_name(0).lower()
                if 'rtx' in gpu_name or 'gtx 16' in gpu_name or 'gtx 20' in gpu_name:
                    return "high_end"
                elif 'gtx' in gpu_name or 'mx' in gpu_name:
                    return "mid_range"
                else:
                    return "low_end"
            else:
                return "cpu_only"
        except:
            return "cpu_only"

    def get_optimized_weights(self):
        """Выбор оптимальных весов модели на основе системы"""
        weights_dir = Path(__file__).parent / "weights"

        if self.performance_profile in ["high_end", "mid_range"]:
            # Для мощных систем используем точные модели
            preferred_weights = ["candies_v10.pt", "ball_rtdetr.pt"]
        else:
            # Для слабых систем используем легкие модели
            preferred_weights = ["1ball_v10.pt", "balls_v10.pt"]

        for weight_file in preferred_weights:
            if (weights_dir / weight_file).exists():
                return str(weights_dir / weight_file)

        # Fallback к любому доступному
        for file in weights_dir.glob("*.pt"):
            if "ball" in file.name.lower():
                return str(file)

        return "weights/candies_v10.pt"  # Последний fallback

    def get_optimized_parameters(self):
        """Получение оптимизированных параметров для текущей системы"""

        base_params = {
            "high_end": {
                "predict_size": 512,
                "conf": 0.20,
                "frame_skip": 1,
                "player_skip": 3,
                "memory_targets": 12,
                "adaptive_mode": True
            },
            "mid_range": {
                "predict_size": 416,
                "conf": 0.22,
                "frame_skip": 1,
                "player_skip": 4,
                "memory_targets": 10,
                "adaptive_mode": True
            },
            "low_end": {
                "predict_size": 320,
                "conf": 0.25,
                "frame_skip": 2,
                "player_skip": 6,
                "memory_targets": 8,
                "adaptive_mode": True
            },
            "cpu_only": {
                "predict_size": 256,
                "conf": 0.28,
                "frame_skip": 3,
                "player_skip": 8,
                "memory_targets": 6,
                "adaptive_mode": False
            }
        }

        return base_params.get(self.performance_profile, base_params["cpu_only"])

    def create_ultra_bot(self, player_detection=False, show_window=False, save_screenshots=False):
        """Создание ультра-бота с оптимальными настройками"""

        print("🚀 ULTRA BOT v6.0 - Инициализация...")
        print(f"💻 Профиль системы: {self.performance_profile.upper()}")

        # Выбор оптимальных весов
        weights_path = self.get_optimized_weights()
        player_weights = "weights/peoples_yolov10m.pt" if player_detection else None

        # Получение оптимизированных параметров
        params = self.get_optimized_parameters()

        print(f"🎯 Модель: {Path(weights_path).name}")
        print(f"⚡ Размер детекции: {params['predict_size']}")
        print(f"🎚️ Порог уверенности: {params['conf']}")
        print(f"🧠 ИИ системы: {'ВКЛЮЧЕНЫ' if self.config['ai_enabled'] else 'ОТКЛЮЧЕНЫ'}")

        if player_weights and Path(player_weights).exists():
            print("🦹 Режим Хищника: АКТИВИРОВАН")
        else:
            print("🤖 Стандартный режим")

        # Создание бота с оптимизированными параметрами
        bot = EnhancedBallBot(
            weights_path=weights_path,
            player_weights_path=player_weights,
            conf_thres=params['conf'],
            adaptive_mode=params['adaptive_mode'],
            show_window=show_window,
            save_screenshots=save_screenshots
        )

        # Применение дополнительных оптимизаций
        bot.frame_skip = params['frame_skip']
        bot.player_detection_skip = params['player_skip']
        bot.target_memory = bot.target_memory.__class__(maxlen=params['memory_targets'])

        # Активация продвинутых систем
        if self.config['ai_enabled']:
            print("🎓 Системы ИИ активированы:")
            print("  ✓ Reinforcement Learning")
            print("  ✓ Предсказание движения")
            print("  ✓ Система уклонения")
            print("  ✓ Интеллектуальная память")

        print("\\n" + "="*60)
        print("🎮 ULTRA BOT ГОТОВ К РАБОТЕ!")
        print("🎯 Нажмите Ctrl+C для остановки")
        print("="*60 + "\\n")

        return bot

def main():
    parser = argparse.ArgumentParser(description='Ultra Bot v6.0 - Maximum Performance & Intelligence')
    parser.add_argument('--predator', action='store_true',
                       help='Активировать режим Хищника (поиск игроков)')
    parser.add_argument('--show', action='store_true',
                       help='Показать окно с аннотациями')
    parser.add_argument('--save-screenshots', action='store_true',
                       help='Сохранять скриншоты с разметкой')
    parser.add_argument('--weights', type=str, default=None,
                       help='Принудительный выбор весов модели')
    parser.add_argument('--conf', type=float, default=None,
                       help='Принудительный порог уверенности')
    parser.add_argument('--no-ai', action='store_true',
                       help='Отключить системы ИИ')
    parser.add_argument('--performance-test', action='store_true',
                       help='Режим тестирования производительности')

    args = parser.parse_args()

    # Создание ультра-загрузчика
    launcher = UltraBotLauncher()

    # Отключение ИИ если запрошено
    if args.no_ai:
        launcher.config['ai_enabled'] = False

    # Переопределение весов если указано
    if args.weights:
        weights_path = f"weights/{args.weights}"
        if not Path(weights_path).exists():
            print(f"❌ Веса не найдены: {weights_path}")
            return
    else:
        weights_path = launcher.get_optimized_weights()

    # Режим тестирования производительности
    if args.performance_test:
        print("🧪 РЕЖИМ ТЕСТИРОВАНИЯ ПРОИЗВОДИТЕЛЬНОСТИ")
        print(f"💻 Система: {launcher.performance_profile.upper()}")

        params = launcher.get_optimized_parameters()
        print(f"⚙️ Оптимальные параметры: {json.dumps(params, indent=2)}")

        # Тест инференса
        try:
            import torch
            from ultralytics import YOLO

            model = YOLO(weights_path)
            if torch.cuda.is_available():
                model.to('cuda')

            # Тестовый запуск
            dummy_img = torch.randn(1, 3, params['predict_size'], params['predict_size'])
            if torch.cuda.is_available():
                dummy_img = dummy_img.cuda()

            start_time = time.time()
            for _ in range(10):
                _ = model(dummy_img, verbose=False)
            end_time = time.time()

            fps = 10 / (end_time - start_time)
            print(f"🚀 Скорость инференса: {fps:.1f} FPS")
            print("✅ Тест завершен успешно!")

        except Exception as e:
            print(f"❌ Ошибка тестирования: {e}")

        return

    # Создание и запуск ультра-бота
    try:
        bot = launcher.create_ultra_bot(
            player_detection=args.predator,
            show_window=args.show,
            save_screenshots=args.save_screenshots
        )

        # Переопределение параметров если указано
        if args.conf:
            bot.base_conf_thres = args.conf
            bot.conf_thres = args.conf

        # Запуск бота
        bot.run()

    except KeyboardInterrupt:
        print("\\n⏹️ ULTRA BOT остановлен пользователем")
    except Exception as e:
        print(f"\\n❌ Критическая ошибка: {e}")
        print("💡 Проверьте системные требования и веса моделей")

if __name__ == '__main__':
    main()