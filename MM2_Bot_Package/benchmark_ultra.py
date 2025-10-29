#!/usr/bin/env python3
"""
Бенчмарк ULTRA BOT - Тестирование производительности и точности
===============================================================

Тестирует все новые системы оптимизации и ИИ для сравнения производительности.
"""

import time
import torch
import numpy as np
import cv2
import psutil
import GPUtil
from ultralytics import YOLO
import json
from pathlib import Path
import matplotlib.pyplot as plt
from datetime import datetime

class UltraBotBenchmark:
    """Комплексное тестирование производительности ULTRA BOT"""

    def __init__(self):
        self.results = {}
        self.system_info = self.get_system_info()

    def get_system_info(self):
        """Сбор информации о системе"""
        info = {
            'cpu': psutil.cpu_count(),
            'cpu_logical': psutil.cpu_count(logical=True),
            'ram_total': psutil.virtual_memory().total / (1024**3),  # GB
            'ram_available': psutil.virtual_memory().available / (1024**3),  # GB
        }

        try:
            gpus = GPUtil.getGPUs()
            if gpus:
                gpu = gpus[0]
                info.update({
                    'gpu_name': gpu.name,
                    'gpu_memory_total': gpu.memoryTotal,
                    'gpu_memory_free': gpu.memoryFree,
                    'gpu_driver': gpu.driver,
                    'gpu_cuda': torch.cuda.is_available()
                })
        except:
            info['gpu_name'] = 'N/A'
            info['gpu_cuda'] = False

        return info

    def benchmark_inference_speed(self, model_path, sizes=[256, 320, 416, 512, 640], iterations=50):
        """Тестирование скорости инференса"""
        print(f"🧪 Тестирование инференса: {Path(model_path).name}")

        results = {}

        try:
            model = YOLO(model_path)
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            model.to(device)

            for size in sizes:
                print(f"  📏 Тестирование размера {size}px...")

                # Создание тестового батча
                batch_size = 1
                dummy_input = torch.randn(batch_size, 3, size, size).to(device)

                # Разогрев
                for _ in range(5):
                    _ = model(dummy_input, verbose=False)

                # Основной тест
                torch.cuda.synchronize() if device == 'cuda' else None
                start_time = time.time()

                for _ in range(iterations):
                    _ = model(dummy_input, verbose=False)

                torch.cuda.synchronize() if device == 'cuda' else None
                end_time = time.time()

                total_time = end_time - start_time
                fps = iterations / total_time
                latency = (total_time / iterations) * 1000  # ms

                results[size] = {
                    'fps': round(fps, 2),
                    'latency_ms': round(latency, 2),
                    'total_time': round(total_time, 3)
                }

                print(f"    ✅ {fps:.1f} FPS, {latency:.1f}ms latency")

        except Exception as e:
            print(f"    ❌ Ошибка: {e}")
            results = {'error': str(e)}

        return results

    def benchmark_accuracy(self, model_path, test_images=None):
        """Тестирование точности детекции"""
        print(f"🎯 Тестирование точности: {Path(model_path).name}")

        if not test_images:
            # Создание синтетических тестовых изображений
            test_images = self.generate_synthetic_test_images()

        results = {
            'total_images': len(test_images),
            'detections': 0,
            'avg_confidence': 0,
            'false_positives': 0,
            'false_negatives': 0
        }

        try:
            model = YOLO(model_path)
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            model.to(device)

            total_conf = 0

            for img_path in test_images:
                results_img = model(img_path, conf=0.1, verbose=False)

                if results_img and len(results_img) > 0:
                    boxes = results_img[0].boxes
                    if boxes is not None:
                        confs = boxes.conf.cpu().numpy()
                        results['detections'] += len(confs)
                        total_conf += np.sum(confs)

            if results['detections'] > 0:
                results['avg_confidence'] = total_conf / results['detections']

        except Exception as e:
            print(f"    ❌ Ошибка точности: {e}")
            results['error'] = str(e)

        return results

    def generate_synthetic_test_images(self, count=10):
        """Генерация синтетических тестовых изображений"""
        images = []

        for i in range(count):
            # Создание изображения с случайными кругами (имитация мячиков)
            img = np.zeros((416, 416, 3), dtype=np.uint8)
            img.fill(100)  # Серый фон

            # Добавление случайных кругов
            for _ in range(np.random.randint(1, 5)):
                center = (np.random.randint(50, 366), np.random.randint(50, 366))
                radius = np.random.randint(10, 30)
                color = (np.random.randint(200, 255), np.random.randint(100, 200), np.random.randint(50, 150))
                cv2.circle(img, center, radius, color, -1)

            # Сохранение
            img_path = f"temp_test_{i}.png"
            cv2.imwrite(img_path, img)
            images.append(img_path)

        return images

    def benchmark_memory_usage(self, model_path):
        """Тестирование использования памяти"""
        print(f"💾 Тестирование памяти: {Path(model_path).name}")

        results = {}

        try:
            # Замер базового использования
            process = psutil.Process()
            base_memory = process.memory_info().rss / (1024**2)  # MB

            # Загрузка модели
            model = YOLO(model_path)
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            model.to(device)

            # Замер после загрузки
            load_memory = process.memory_info().rss / (1024**2)  # MB

            # Тестовый инференс
            dummy_input = torch.randn(1, 3, 416, 416).to(device)
            _ = model(dummy_input, verbose=False)

            # Замер во время работы
            work_memory = process.memory_info().rss / (1024**2)  # MB

            # GPU память
            gpu_memory = 0
            if torch.cuda.is_available():
                gpu_memory = torch.cuda.memory_allocated() / (1024**2)  # MB

            results = {
                'base_memory_mb': round(base_memory, 1),
                'load_memory_mb': round(load_memory, 1),
                'work_memory_mb': round(work_memory, 1),
                'model_memory_delta_mb': round(load_memory - base_memory, 1),
                'gpu_memory_mb': round(gpu_memory, 1)
            }

            print(f"    📊 RAM: {results['model_memory_delta_mb']}MB, GPU: {results['gpu_memory_mb']}MB")

        except Exception as e:
            print(f"    ❌ Ошибка памяти: {e}")
            results = {'error': str(e)}

        return results

    def run_full_benchmark(self, models=None, output_file="benchmark_results.json"):
        """Запуск полного бенчмарка"""
        print("🚀 ULTRA BOT BENCHMARK v6.0")
        print("=" * 50)
        print(f"💻 Система: {self.system_info}")
        print()

        if not models:
            # Автоопределение доступных моделей
            weights_dir = Path("weights")
            models = []
            if weights_dir.exists():
                for pt_file in weights_dir.glob("*.pt"):
                    if "ball" in pt_file.name.lower() or "cand" in pt_file.name.lower():
                        models.append(str(pt_file))

        if not models:
            print("❌ Модели не найдены в папке weights/")
            return

        print(f"🎯 Найдено моделей: {len(models)}")
        for model in models:
            print(f"  - {Path(model).name}")
        print()

        all_results = {
            'timestamp': datetime.now().isoformat(),
            'system_info': self.system_info,
            'models': {}
        }

        for model_path in models:
            model_name = Path(model_path).name
            print(f"🔬 Тестирование модели: {model_name}")
            print("-" * 40)

            model_results = {}

            # 1. Скорость инференса
            model_results['inference_speed'] = self.benchmark_inference_speed(model_path)

            # 2. Точность
            model_results['accuracy'] = self.benchmark_accuracy(model_path)

            # 3. Использование памяти
            model_results['memory'] = self.benchmark_memory_usage(model_path)

            all_results['models'][model_name] = model_results
            print()

        # Сохранение результатов
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(all_results, f, indent=2, ensure_ascii=False)

        print(f"💾 Результаты сохранены в: {output_file}")

        # Вывод сводки
        self.print_summary(all_results)

        return all_results

    def print_summary(self, results):
        """Вывод сводки результатов"""
        print("\\n📊 СВОДКА РЕЗУЛЬТАТОВ")
        print("=" * 50)

        for model_name, model_data in results['models'].items():
            print(f"\\n🎯 {model_name}:")

            if 'inference_speed' in model_data and 'error' not in model_data['inference_speed']:
                speeds = model_data['inference_speed']
                best_size = max(speeds.keys(), key=lambda x: speeds[x]['fps'] if isinstance(speeds[x], dict) else 0)
                best_fps = speeds[best_size]['fps'] if isinstance(speeds[best_size], dict) else 0
                print(f"  🚀 Лучшая скорость: {best_fps} FPS при {best_size}px")

            if 'accuracy' in model_data and 'error' not in model_data['accuracy']:
                acc = model_data['accuracy']
                print(f"  🎯 Детекций: {acc['detections']}, Средняя уверенность: {acc.get('avg_confidence', 0):.3f}")

            if 'memory' in model_data and 'error' not in model_data['memory']:
                mem = model_data['memory']
                print(f"  💾 Память: +{mem['model_memory_delta_mb']}MB RAM")

        # Рекомендация
        print("\\n🎖️ РЕКОМЕНДАЦИЯ:")
        best_model = None
        best_score = 0

        for model_name, model_data in results['models'].items():
            score = 0
            if 'inference_speed' in model_data:
                speeds = model_data['inference_speed']
                if not isinstance(speeds, dict) or 'error' in speeds:
                    continue
                max_fps = max(s['fps'] for s in speeds.values() if isinstance(s, dict))
                score += max_fps * 0.6

            if 'accuracy' in model_data:
                acc = model_data['accuracy']
                if isinstance(acc, dict) and 'detections' in acc:
                    score += acc['detections'] * 0.4

            if score > best_score:
                best_score = score
                best_model = model_name

        if best_model:
            print(f"  🏆 Рекомендуемая модель: {best_model}")
        else:
            print("  ⚠️ Не удалось определить лучшую модель")

def main():
    import argparse

    parser = argparse.ArgumentParser(description='ULTRA BOT Benchmark Suite')
    parser.add_argument('--models', nargs='+', help='Список моделей для тестирования')
    parser.add_argument('--output', default='benchmark_results.json', help='Файл для сохранения результатов')
    parser.add_argument('--quick', action='store_true', help='Быстрый тест (меньше итераций)')

    args = parser.parse_args()

    benchmark = UltraBotBenchmark()

    if args.quick:
        # Быстрый тест для предварительной оценки
        print("⚡ Быстрый режим тестирования")
        iterations = 10
        sizes = [320, 416]
    else:
        iterations = 50
        sizes = [256, 320, 416, 512, 640]

    # Переопределение параметров в классе для быстрого режима
    original_benchmark = benchmark.benchmark_inference_speed
    def quick_benchmark(model_path, sizes=sizes, iterations=iterations):
        return original_benchmark(model_path, sizes, iterations)
    benchmark.benchmark_inference_speed = quick_benchmark

    try:
        results = benchmark.run_full_benchmark(args.models, args.output)
        print("\\n✅ Бенчмарк завершен успешно!")

    except KeyboardInterrupt:
        print("\\n⏹️ Бенчмарк прерван пользователем")
    except Exception as e:
        print(f"\\n❌ Ошибка бенчмарка: {e}")

if __name__ == "__main__":
    main()