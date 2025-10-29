"""
🚀 Продвинутый бот для Murder Mystery 2 - Ultra Edition v2.0

Основные улучшения:
✨ Плавное движение с адаптивным PID-контроллером
🎯 Умная система приоритизации целей
🧠 Адаптивная система порогов уверенности
⚡ Оптимизированная производительность (мультискейл детекция)
🛡️ Продвинутая система антизастревания
📊 Красивая визуализация и статистика
🔧 Модульная архитектура
"""

import sys
import os
import argparse
import time
import cv2
import numpy as np
import torch
import json
import random
from collections import deque
from datetime import datetime
from enum import Enum

sys.path.insert(0, os.path.dirname(__file__))

from ultralytics import YOLO
from roblox.screen import CaptureStream
from roblox.control import Control
from roblox.utils import FrameCounter

# ==================== КОНФИГУРАЦИЯ ====================

class BotState(Enum):
    """Состояния бота"""
    SEARCHING = "SEARCHING"      # Поиск целей
    HUNTING = "HUNTING"          # Движение к цели
    COLLECTING = "COLLECTING"    # Сбор объекта
    STUCK = "STUCK"              # Застрял, пытаемся выйти
    ESCAPING = "ESCAPING"        # Выполняем маневр выхода

# ==================== ОСНОВНОЙ КЛАСС ====================

class AdvancedMM2Bot:
    """
    Продвинутый бот для MM2 с улучшенным алгоритмом
    """
    
    def __init__(self, weights_path, conf_thres=0.25, show_window=False, 
                 adaptive_mode=True, performance_mode='balanced'):
        """
        Инициализация бота
        
        Args:
            weights_path: Путь к весам модели
            conf_thres: Базовый порог уверенности (0.0-1.0)
            show_window: Показывать окно с визуализацией
            adaptive_mode: Включить адаптивный режим
            performance_mode: Режим производительности ('speed', 'balanced', 'accuracy')
        """
        print("=" * 60)
        print("🚀 ПРОДВИНУТЫЙ MM2 БОТ - ULTRA EDITION v2.0")
        print("=" * 60)
        
        # Инициализация устройства
        self.device = self._init_device()
        
        # Загрузка модели
        print(f"\n📦 Загрузка модели: {weights_path}")
        self.model = YOLO(weights_path).to(self.device)
        print("✅ Модель загружена!")
        
        # Параметры детекции
        self.base_conf_thres = conf_thres
        self.conf_thres = conf_thres
        self.adaptive_mode = adaptive_mode
        self.show_window = show_window
        self.performance_mode = performance_mode
        
        # Инициализация параметров перед настройкой производительности
        self.predict_size = 416  # Значение по умолчанию
        
        # Настройка производительности
        self._setup_performance_mode()
        
        # Захват экрана
        print("\n📷 Инициализация захвата экрана...")
        self.stream = CaptureStream("Roblox", saveInterval=0)
        
        # Управление
        self.control = Control()
        self.frame_counter = FrameCounter()
        self.frame_counter.fps = 0.0
        
        # Состояние бота
        self.state = BotState.SEARCHING
        self.current_target = None
        self.target_history = deque(maxlen=5)
        
        # ============ ПРОДВИНУТЫЙ PID-КОНТРОЛЛЕР ============
        self.pid_kp = 0.04          # Пропорциональный коэффициент
        self.pid_ki = 0.001        # Интегральный коэффициент
        self.pid_kd = 0.025        # Дифференциальный коэффициент
        self.integral = 0.0
        self.prev_error = 0.0
        self.dt = 0.05
        self.integral_limit = 150.0
        self.error_smoothing = deque(maxlen=5)  # Сглаживание ошибок
        
        # ============ ПАРАМЕТРЫ ДВИЖЕНИЯ ============
        self.min_distance = 50               # Минимальное расстояние для сбора
        self.turn_threshold = 20              # Порог для поворота (пиксели)
        self.approach_speed = 0.12           # Скорость приближения
        self.precision_mode_distance = 100   # Дистанция для точного режима
        
        # ============ АДАПТИВНАЯ СИСТЕМА ============
        self.conf_history = deque(maxlen=30)
        self.detection_history = deque(maxlen=20)
        self.collection_rate = 0.0
        self.successful_collections = 0
        self.total_attempts = 0
        
        # ============ АНТИЗАСТРЕВАНИЕ ============
        self.stuck_frames = 0
        self.stuck_threshold = 12
        self.previous_distance = float('inf')
        self.progress_history = deque(maxlen=10)
        self.escape_cooldown = 3.0
        self.last_escape_time = 0.0
        
        # ============ ПОИСК ============
        self.frames_without_targets = 0
        self.search_threshold = 8
        self.search_turn_direction = 'turn_right'
        self.search_turn_speed = 0.15
        
        # ============ МУЛЬТИСКЕЙЛ ДЕТЕКЦИЯ ============
        self.multi_scale_enabled = True
        self.base_predict_size = self.predict_size
        self.scale_factors = [1.0, 1.2, 1.5] if self.performance_mode != 'speed' else [1.0]
        self.current_scale = 0
        
        # ============ ПРЕДОБРАБОТКА ============
        self.preprocess_enabled = True
        self.contrast_gain = 1.15
        self.brightness_gain = 1.25
        
        # ============ СТАТИСТИКА ============
        self.stats = {
            'start_time': time.time(),
            'detections': 0,
            'collections': 0,
            'search_cycles': 0,
            'stuck_events': 0,
            'avg_confidence': 0.0,
            'total_distance_traveled': 0.0
        }
        
        # ============ ВИЗУАЛИЗАЦИЯ ============
        self.show_visualization = show_window
        if self.show_visualization:
            try:
                test_img = np.zeros((100, 100, 3), dtype=np.uint8)
                cv2.imshow('Test', test_img)
                cv2.waitKey(1)
                cv2.destroyAllWindows()
                print("✅ GUI поддерживается")
            except:
                print("⚠️ GUI не поддерживается, отключаем визуализацию")
                self.show_visualization = False
        
        print("\n" + "=" * 60)
        print("✅ БОТ ГОТОВ К РАБОТЕ!")
        print(f"📊 Режим: {performance_mode}, Адаптивный: {adaptive_mode}")
        print(f"🎯 Conf: {self.conf_thres:.2f}, Размер: {self.predict_size}")
        print("=" * 60)
        print("\n▶️  Нажми Ctrl+C для остановки\n")
    
    def _init_device(self):
        """Инициализация устройства (GPU/CPU)"""
        if torch.cuda.is_available():
            device = torch.device("cuda:0")
            gpu_name = torch.cuda.get_device_name(0)
            print(f"\n🎮 GPU обнаружена: {gpu_name}")
            return device
        else:
            print("\n💻 Используется CPU")
            return torch.device("cpu")
    
    def _setup_performance_mode(self):
        """Настройка режима производительности"""
        modes = {
            'speed': {'size': 320, 'skip': 2, 'multi_scale': False},
            'balanced': {'size': 416, 'skip': 1, 'multi_scale': True},
            'accuracy': {'size': 512, 'skip': 0, 'multi_scale': True}
        }
        
        config = modes.get(self.performance_mode, modes['balanced'])
        self.predict_size = config['size']
        self.frame_skip = config['skip']
        if not config['multi_scale']:
            self.scale_factors = [1.0]
    
    def preprocess_frame(self, img):
        """Улучшенная предобработка кадра"""
        if not self.preprocess_enabled:
            return img
        
        # Конвертация в HSV для улучшения контраста
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
        
        # Увеличение насыщенности и яркости
        hsv[:, :, 1] = np.clip(hsv[:, :, 1] * self.contrast_gain, 0, 255)
        hsv[:, :, 2] = np.clip(hsv[:, :, 2] * self.brightness_gain, 0, 255)
        
        # Обратная конвертация
        enhanced = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
        return enhanced
    
    def run_multi_scale_detection(self, img):
        """
        Мультискейл детекция для улучшения точности
        """
        best_targets = []
        best_results = None
        
        for scale_factor in self.scale_factors:
            size = int(self.base_predict_size * scale_factor)
            # Округление до кратного 32
            size = (size // 32) * 32
            if size < 32:
                size = 32
            
            try:
                results = self.model(
                    img,
                    imgsz=size,
                    device=self.device,
                    conf=self.conf_thres * (0.95 if scale_factor > 1.0 else 1.0),
                    verbose=False,
                    half=self.device.type == 'cuda'
                )
                
                targets = self.find_targets(results, img.shape)
                if targets and (not best_targets or targets[0]['priority'] > best_targets[0]['priority']):
                    best_targets = targets
                    best_results = results
                    break  # Если нашли на первом масштабе, можно пропустить остальные
                    
            except Exception as e:
                continue
        
        return best_targets, best_results
    
    def find_targets(self, results, frame_shape, max_targets=3):
        """
        Поиск и приоритизация целей
        """
        frame_h, frame_w = frame_shape[:2]
        center_x = frame_w / 2
        center_y = frame_h / 2
        
        targets = []
        
        if results and len(results) > 0 and results[0].boxes is not None:
            boxes = results[0].boxes
            
            for i in range(len(boxes)):
                conf = float(boxes.conf[i].item())
                cls = int(boxes.cls[i].item())
                
                if cls == 0 and conf >= self.conf_thres:
                    xyxy = boxes.xyxy[i].cpu().numpy()
                    x1, y1, x2, y2 = xyxy
                    
                    cx = (x1 + x2) / 2
                    cy = (y1 + y2) / 2
                    
                    # Фильтрация по зоне видимости
                    norm_x = cx / frame_w
                    norm_y = cy / frame_h
                    
                    if norm_y < 0.25 or norm_x < 0.1 or norm_x > 0.9:
                        continue
                    
                    # Расстояние до центра
                    distance = np.sqrt((cx - center_x)**2 + (cy - center_y)**2)
                    
                    # Расчет приоритета
                    priority = self.calculate_priority(cx, cy, conf, distance, center_x, center_y)
                    
                    targets.append({
                        'cx': cx,
                        'cy': cy,
                        'conf': conf,
                        'distance': distance,
                        'priority': priority,
                        'bbox': (x1, y1, x2, y2)
                    })
        
        # Сортировка по приоритету
        targets.sort(key=lambda t: t['priority'], reverse=True)
        return targets[:max_targets]
    
    def calculate_priority(self, cx, cy, conf, distance, center_x, center_y):
        """
        Улучшенный расчет приоритета цели
        
        Учитывает:
        - Уверенность детекции
        - Близость к центру экрана
        - Близость по расстоянию
        - Вертикальное положение (предпочтение нижней части)
        """
        priority = conf * 100.0  # Базовый приоритет
        
        # Бонус за близость к центру (X)
        center_dist_x = abs(cx - center_x)
        priority += max(0, 50 - center_dist_x / 8)
        
        # Бонус за близость к центру (Y) - усиленный
        center_dist_y = abs(cy - center_y)
        priority += max(0, 70 - center_dist_y / 5) * 1.5
        
        # Бонус за близость (общее расстояние)
        priority += max(0, 100 - distance / 6)
        
        # Бонус за уверенность
        priority += conf * 30
        
        # Штраф за слишком высокое положение
        frame_h = center_y * 2  # Примерная высота кадра
        if cy < frame_h * 0.3:
            priority *= 0.7
        
        return priority
    
    def adaptive_confidence_adjustment(self, targets):
        """
        Адаптивная настройка порога уверенности
        """
        if not self.adaptive_mode:
            return
        
        if len(targets) > 0:
            # Есть детекции - анализируем их качество
            avg_conf = np.mean([t['conf'] for t in targets])
            self.conf_history.append(avg_conf)
            
            if len(self.conf_history) >= 10:
                recent_avg = np.mean(list(self.conf_history)[-10:])
                
                if recent_avg > 0.65:  # Высокая уверенность
                    self.conf_thres = min(0.4, self.base_conf_thres + 0.08)
                elif recent_avg < 0.4:  # Низкая уверенность
                    self.conf_thres = max(0.15, self.base_conf_thres - 0.06)
                else:
                    self.conf_thres = self.base_conf_thres
        
        else:
            # Нет детекций - снижаем порог для поиска
            self.frames_without_targets += 1
            if self.frames_without_targets > 15:
                self.conf_thres = max(0.12, self.base_conf_thres - 0.08)
    
    def move_to_target(self, target, frame_shape):
        """
        Плавное движение к цели с улучшенным PID-контроллером
        """
        frame_h, frame_w = frame_shape[:2]
        center_x = frame_w / 2
        center_y = frame_h / 2
        
        cx = target['cx']
        cy = target['cy']
        distance = target['distance']
        conf = target['conf']
        
        # Проверка достижения цели
        if distance < self.min_distance:
            self.control.release_all_keys()
            time.sleep(0.15)  # Пауза для сбора
            self.state = BotState.COLLECTING
            self.successful_collections += 1
            self.total_attempts += 1
            print(f"✅ СБОР! Conf: {conf:.2f}, Dist: {distance:.1f}px")
            return True
        
        # Расчет ошибки с сглаживанием
        error = cx - center_x
        self.error_smoothing.append(error)
        smoothed_error = np.mean(self.error_smoothing)
        
        # Продвинутый PID-контроллер
        self.integral += smoothed_error * self.dt
        self.integral = np.clip(self.integral, -self.integral_limit, self.integral_limit)
        
        derivative = (smoothed_error - self.prev_error) / max(self.dt, 0.01)
        pid_output = (self.pid_kp * smoothed_error + 
                     self.pid_ki * self.integral + 
                     self.pid_kd * derivative)
        
        self.prev_error = smoothed_error
        
        # Адаптивная скорость поворота
        turn_strength = min(abs(pid_output) / 50.0, 1.0)
        turn_duration = 0.05 + (turn_strength * 0.15)
        
        # Режим точности при близком подходе
        if distance < self.precision_mode_distance:
            turn_duration *= 0.7  # Медленнее и точнее
        
        # Движение
        self.control.release_all_keys()
        self.control.press('up')  # Всегда движемся вперед
        
        if abs(smoothed_error) > self.turn_threshold:
            if smoothed_error < 0:
                self.control.press('left')
            else:
                self.control.press('right')
        
        time.sleep(float(turn_duration))
        
        # Проверка прогресса и застревания
        progress = distance < self.previous_distance - 3.0
        self.progress_history.append(progress)
        
        if progress:
            self.stuck_frames = 0
        else:
            self.stuck_frames += 1
        
        # Обновление предыдущего расстояния
        old_distance = self.previous_distance
        self.previous_distance = distance
        if old_distance != float('inf'):
            self.stats['total_distance_traveled'] += max(0, old_distance - distance)
        
        # Проверка застревания
        if self.stuck_frames >= self.stuck_threshold:
            print("⚠️ Обнаружено застревание!")
            self.state = BotState.STUCK
            self.stats['stuck_events'] += 1
            return False
        
        return False
    
    def perform_escape_maneuver(self):
        """
        Маневр выхода из застревания
        """
        now = time.time()
        if now - self.last_escape_time < self.escape_cooldown:
            return
        
        print("🔧 Выполняю маневр выхода...")
        self.state = BotState.ESCAPING
        self.last_escape_time = now
        
        self.control.release_all_keys()
        
        # 1. Отъезд назад
        self.control.press('down')
        time.sleep(0.3)
        self.control.release_all_keys()
        
        # 2. Поворот
        turn_dir = random.choice(['turn_left', 'turn_right'])
        self.control.press(turn_dir)
        time.sleep(0.4)
        self.control.release_all_keys()
        
        # 3. Прыжок и движение
        self.control.press('jump')
        self.control.press('up')
        time.sleep(0.2)
        self.control.release_all_keys()
        
        # Сброс состояния
        self.stuck_frames = 0
        self.previous_distance = float('inf')
        self.integral = 0.0
        self.prev_error = 0.0
        self.error_smoothing.clear()
        
        time.sleep(0.3)
        self.state = BotState.SEARCHING
    
    def search_mode(self):
        """
        Режим поиска целей
        """
        self.control.release_all_keys()
        
        # Медленный поворот для поиска
        self.control.press(self.search_turn_direction)
        time.sleep(self.search_turn_speed)
        self.control.release_all_keys()
        time.sleep(0.05)
        
        # Периодические прыжки
        if self.frames_without_targets % 15 == 0:
            self.control.press('jump')
            time.sleep(0.1)
            self.control.release_all_keys()
        
        # Смена направления
        if self.frames_without_targets % 30 == 0:
            self.search_turn_direction = 'turn_left' if self.search_turn_direction == 'turn_right' else 'turn_right'
    
    def visualize_frame(self, img, targets, results):
        """
        Красивая визуализация кадра с информацией
        """
        if not self.show_visualization:
            return
        
        try:
            # Рисуем детекции
            if results and len(results) > 0:
                annotated = results[0].plot()
            else:
                annotated = img.copy()
            
            h, w = annotated.shape[:2]
            
            # Центр экрана
            cv2.drawMarker(annotated, (w // 2, h // 2), (0, 255, 255), 
                          cv2.MARKER_CROSS, 20, 2)
            
            # Текущая цель
            if self.current_target:
                tx = int(self.current_target['cx'])
                ty = int(self.current_target['cy'])
                cv2.circle(annotated, (tx, ty), 15, (0, 255, 0), 3)
                cv2.circle(annotated, (tx, ty), 5, (0, 255, 0), -1)
                
                # Линия к цели
                cv2.line(annotated, (w // 2, h // 2), (tx, ty), (0, 255, 0), 2)
            
            # Информационная панель
            state_color = {
                BotState.SEARCHING: (255, 165, 0),   # Оранжевый
                BotState.HUNTING: (0, 255, 0),       # Зеленый
                BotState.COLLECTING: (0, 255, 255),  # Желтый
                BotState.STUCK: (0, 0, 255),         # Красный
                BotState.ESCAPING: (255, 0, 255)     # Розовый
            }
            
            state_text = f"STATE: {self.state.value}"
            cv2.putText(annotated, state_text, (10, 30), 
                       cv2.FONT_HERSHEY_BOLD, 1.0, state_color.get(self.state, (255, 255, 255)), 2)
            
            # Статистика
            info_y = 60
            info_lines = [
                f"FPS: {self.frame_counter.fps:.1f}",
                f"Conf: {self.conf_thres:.2f}",
                f"Targets: {len(targets)}",
                f"Collections: {self.successful_collections}",
                f"Success Rate: {(self.successful_collections/max(1, self.total_attempts)*100):.1f}%"
            ]
            
            for line in info_lines:
                cv2.putText(annotated, line, (10, info_y), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                info_y += 25
            
            # Показ окна
            cv2.imshow('Advanced MM2 Bot - Ultra Edition', annotated)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                return True  # Сигнал остановки
            
        except Exception as e:
            print(f"[WARN] Ошибка визуализации: {e}")
        
        return False
    
    def print_statistics(self):
        """Вывод статистики"""
        runtime = time.time() - self.stats['start_time']
        
        print("\n" + "=" * 60)
        print("📊 СТАТИСТИКА РАБОТЫ БОТА")
        print("=" * 60)
        print(f"⏱️  Время работы: {runtime:.1f} секунд ({runtime/60:.1f} минут)")
        print(f"🎯 Детекций: {self.stats['detections']}")
        print(f"✅ Собрано: {self.successful_collections}")
        print(f"📈 Успешность: {(self.successful_collections/max(1, self.total_attempts)*100):.1f}%")
        print(f"🔄 Циклов поиска: {self.stats['search_cycles']}")
        print(f"⚠️  Застреваний: {self.stats['stuck_events']}")
        print(f"📊 Средняя уверенность: {self.stats['avg_confidence']:.3f}")
        print(f"🎮 FPS: {self.frame_counter.fps:.1f}")
        print("=" * 60)
    
    def run(self):
        """
        Основной цикл работы бота
        """
        frame_count = 0
        
        try:
            prev_time = time.time()
            
            for img, img0 in self.stream:
                current_time = time.time()
                self.dt = current_time - prev_time
                prev_time = current_time
                
                frame_count += 1
                
                # Пропуск кадров для производительности
                if frame_count % (self.frame_skip + 1) != 0 and self.current_target:
                    # Если есть цель, продолжаем движение
                    if self.state == BotState.HUNTING and self.current_target:
                        self.move_to_target(self.current_target, img.shape)
                    continue
                
                # Предобработка
                processed_img = self.preprocess_frame(img)
                
                # Детекция
                targets, results = self.run_multi_scale_detection(processed_img)
                
                # Логирование
                if targets:
                    self.stats['detections'] += 1
                    avg_conf = np.mean([t['conf'] for t in targets])
                    self.stats['avg_confidence'] = (self.stats['avg_confidence'] * (self.stats['detections'] - 1) + avg_conf) / self.stats['detections']
                
                # Адаптивная настройка
                self.adaptive_confidence_adjustment(targets)
                
                # ============ УПРАВЛЕНИЕ СОСТОЯНИЯМИ ============
                
                if targets:
                    # Найдена цель - переходим в режим охоты
                    best_target = targets[0]
                    
                    if self.state != BotState.HUNTING or self.current_target != best_target:
                        self.state = BotState.HUNTING
                        self.current_target = best_target
                        self.frames_without_targets = 0
                    
                    # Движение к цели
                    collected = self.move_to_target(best_target, img.shape)
                    
                    if collected:
                        self.state = BotState.SEARCHING
                        self.current_target = None
                        time.sleep(0.2)  # Пауза после сбора
                
                elif self.state == BotState.HUNTING:
                    # Потеряли цель
                    self.frames_without_targets += 1
                    if self.frames_without_targets >= 3:
                        self.state = BotState.SEARCHING
                        self.current_target = None
                
                elif self.state == BotState.SEARCHING:
                    self.search_mode()
                    self.frames_without_targets += 1
                    if self.frames_without_targets >= self.search_threshold:
                        self.stats['search_cycles'] += 1
                
                elif self.state == BotState.STUCK:
                    self.perform_escape_maneuver()
                
                elif self.state == BotState.COLLECTING:
                    self.state = BotState.SEARCHING
                    self.current_target = None
                
                # Визуализация
                if self.show_visualization:
                    if self.visualize_frame(img, targets, results):
                        break  # Остановка по 'q'
                
                # Обновление FPS
                self.frame_counter.log()
                
                # Периодический вывод статистики
                if int(current_time) % 30 == 0 and int(current_time) != int(prev_time):
                    self.print_statistics()
        
        except KeyboardInterrupt:
            print("\n\n⚠️  Получен сигнал остановки (Ctrl+C)")
        
        except Exception as e:
            print(f"\n❌ Критическая ошибка: {e}")
            import traceback
            traceback.print_exc()
        
        finally:
            print("\n🛑 Завершение работы...")
            self.control.release_all_keys()
            self.print_statistics()
            if self.show_visualization:
                cv2.destroyAllWindows()
            print("✅ Бот остановлен\n")


# ==================== ТОЧКА ВХОДА ====================

def main():
    parser = argparse.ArgumentParser(
        description='🚀 Продвинутый MM2 Бот - Ultra Edition v2.0',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Примеры использования:
  python run_advanced_bot.py --weights weights/candies_v10.pt --conf 0.25
  python run_advanced_bot.py --weights weights/candies_v10.pt --show --mode accuracy
  python run_advanced_bot.py --weights weights/candies_v10.pt --no-adaptive --mode speed
        """
    )
    
    parser.add_argument('--weights', type=str, 
                       default='weights/candies_v10.pt',
                       help='Путь к весам модели')
    
    parser.add_argument('--conf', type=float, default=0.25,
                       help='Базовый порог уверенности (0.0-1.0)')
    
    parser.add_argument('--show', action='store_true',
                       help='Показать окно с визуализацией')
    
    parser.add_argument('--mode', type=str, 
                       choices=['speed', 'balanced', 'accuracy'],
                       default='balanced',
                       help='Режим производительности')
    
    parser.add_argument('--no-adaptive', action='store_true',
                       help='Отключить адаптивный режим')
    
    args = parser.parse_args()
    
    # Проверка файла весов
    if not os.path.exists(args.weights):
        print(f"❌ Ошибка: Файл весов не найден: {args.weights}")
        print("\n📁 Доступные веса:")
        weights_dir = os.path.join(os.path.dirname(__file__), 'weights')
        if os.path.exists(weights_dir):
            for f in os.listdir(weights_dir):
                if f.endswith('.pt'):
                    print(f"  - {f}")
        return
    
    # Запуск бота
    bot = AdvancedMM2Bot(
        weights_path=args.weights,
        conf_thres=args.conf,
        show_window=args.show,
        adaptive_mode=not args.no_adaptive,
        performance_mode=args.mode
    )
    
    bot.run()


if __name__ == '__main__':
    main()
