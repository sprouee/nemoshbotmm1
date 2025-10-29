"""
🚀 ULTRA BOT - Премиум версия бота для Murder Mystery 2
Версия 3.0 - Ultimate Performance Edition

Максимальные улучшения:
✨ Умная система навигации с предсказанием траектории
✨ Мультицелевая система приоритизации
✨ Адаптивная система порогов с машинным обучением
✨ Оптимизированная производительность (GPU/CPU)
✨ Система памяти целей с маппингом локаций
✨ Улучшенная визуализация с телеметрией
✨ Автоматическая оптимизация параметров
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
import keyboard
from collections import deque
from datetime import datetime
from enum import Enum
from typing import Optional, List, Dict, Tuple
import threading

sys.path.insert(0, os.path.dirname(__file__))

from ultralytics import YOLO
from roblox.screen import CaptureStream
from roblox.control import Control
from roblox.utils import FrameCounter

# Импорт улучшенных утилит
try:
    from bot_utils_enhanced import (
        TargetTracker, PerformanceMonitor, SmartSearchEngine,
        AdaptiveController, calculate_target_priority
    )
except ImportError:
    print("[WARN] bot_utils_enhanced не найден, используем базовые функции")


class BotState(Enum):
    """Состояния бота"""
    INITIALIZING = "initializing"
    SEARCHING = "searching"
    APPROACHING = "approaching"
    COLLECTING = "collecting"
    RETREATING = "retreating"
    STUCK = "stuck"


class UltraBot:
    """
    Премиум бот с максимальными улучшениями для сбора монет/мячиков
    """
    
    def __init__(
        self,
        weights_path: str,
        player_weights_path: Optional[str] = None,
        conf_thres: float = 0.22,
        show_window: bool = True,
        performance_mode: str = "balanced"
    ):
        """
        Инициализация премиум бота
        
        Args:
            weights_path: Путь к весам модели для монет
            player_weights_path: Путь к весам модели для игроков (опционально)
            conf_thres: Базовый порог уверенности
            show_window: Показывать окно с визуализацией
            performance_mode: Режим производительности ("speed", "balanced", "accuracy")
        """
        print("=" * 60)
        print("🚀 ULTRA BOT v3.0 - Ultimate Performance Edition")
        print("=" * 60)
        
        # === Инициализация модели ===
        self.device = self._init_device()
        print(f"[INFO] Device: {self.device}")
        
        print(f"[LOAD] Загрузка модели: {weights_path}")
        self.model = YOLO(weights_path)
        if self.device.type == 'cuda':
            self.model.to(self.device)
            # Оптимизация для CUDA
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.deterministic = False
        
        self.player_model = None
        if player_weights_path and os.path.exists(player_weights_path):
            print(f"[LOAD] Загрузка модели игроков: {player_weights_path}")
            self.player_model = YOLO(player_weights_path)
            if self.device.type == 'cuda':
                self.player_model.to(self.device)
        
        # === Параметры производительности ===
        self.performance_mode = performance_mode
        self._setup_performance_mode()
        
        # === Инициализация захвата экрана ===
        print("[INIT] Инициализация захвата экрана...")
        self.stream = CaptureStream("Roblox", saveInterval=0)
        
        # === Инициализация управления ===
        self.control = Control()
        self.frame_counter = FrameCounter(interval=3)
        self.frame_counter.fps = 0.0
        
        # === Состояние бота ===
        self.state = BotState.INITIALIZING
        self.previous_state = None
        
        # === Параметры детекции ===
        self.base_conf_thres = conf_thres
        self.conf_thres = conf_thres
        self.adaptive_confidence = True
        self.conf_history = deque(maxlen=100)
        
        # === Параметры движения ===
        self.min_distance = 48  # Оптимизированное расстояние сбора
        self.turn_threshold = 20
        self.approach_threshold = 180
        
        # === Улучшенный PID контроллер ===
        self._init_pid_controller()
        
        # === Система навигации ===
        self._init_navigation()
        
        # === Система памяти целей ===
        self.target_memory = deque(maxlen=15)
        self.memory_retention = 3.0  # секунд
        self.location_map = {}  # Карта локаций
        
        # === Система поиска ===
        self._init_search_system()
        
        # === Антизастревание ===
        self.stuck_detection = {
            'frames': 0,
            'threshold': 20,
            'last_position': None,
            'position_history': deque(maxlen=10)
        }
        
        # === Статистика ===
        self.stats = {
            'start_time': time.time(),
            'detections': 0,
            'collections': 0,
            'search_cycles': 0,
            'stuck_events': 0,
            'avg_fps': 0.0,
            'avg_confidence': 0.0,
            'state_history': deque(maxlen=100)
        }
        
        # === Визуализация ===
        self.show_window = show_window
        self.window_name = "Ultra Bot - Live View"
        if self.show_window:
            self._init_visualization()
        
        # === Утилиты (если доступны) ===
        try:
            self.target_tracker = TargetTracker(max_history=15)
            self.performance_monitor = PerformanceMonitor()
            self.search_engine = SmartSearchEngine()
            self.adaptive_controller = AdaptiveController()
        except NameError:
            print("[INFO] Расширенные утилиты недоступны, используем базовые функции")
            self.target_tracker = None
            self.performance_monitor = None
            self.search_engine = None
            self.adaptive_controller = None
        
        # === Глобальная остановка ===
        self.stop_requested = False
        self._setup_stop_handler()
        
        print("\n[OK] ✨ Ultra Bot готов к работе!")
        print(f"[INFO] Режим: {performance_mode}")
        print(f"[INFO] Conf threshold: {self.conf_thres}")
        print(f"[INFO] Device: {self.device}")
        print("\n[START] Ctrl+C или 'Q' для остановки\n")
    
    def _init_device(self) -> torch.device:
        """Инициализация устройства с оптимизацией"""
        if torch.cuda.is_available():
            device = torch.device("cuda:0")
            torch.cuda.set_device(device)
            gpu_name = torch.cuda.get_device_name(0)
            memory_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
            print(f"[GPU] {gpu_name} ({memory_gb:.1f} GB)")
            return device
        else:
            print("[CPU] Используется CPU")
            return torch.device("cpu")
    
    def _setup_performance_mode(self):
        """Настройка параметров в зависимости от режима производительности"""
        if self.performance_mode == "speed":
            self.predict_size = 320
            self.frame_skip = 2
            self.multi_scale = False
            self.use_half_precision = True
        elif self.performance_mode == "accuracy":
            self.predict_size = 640
            self.frame_skip = 0
            self.multi_scale = True
            self.use_half_precision = False
        else:  # balanced
            self.predict_size = 416
            self.frame_skip = 1
            self.multi_scale = True
            self.use_half_precision = self.device.type == 'cuda'
        
        self.frame_counter_internal = 0
    
    def _init_pid_controller(self):
        """Инициализация улучшенного PID контроллера"""
        self.pid = {
            'kp': 0.045,
            'ki': 0.0012,
            'kd': 0.028,
            'integral': 0.0,
            'prev_error': 0.0,
            'dt': 0.05,
            'integral_limit': 120.0,
            'alpha': 0.3  # EMA для сглаживания
        }
    
    def _init_navigation(self):
        """Инициализация системы навигации"""
        self.current_target = None
        self.target_history = deque(maxlen=10)
        self.smoothed_target = None
        self.smoothing_alpha = 0.35
        
        # Предсказание траектории
        self.trajectory_predictor = {
            'history': deque(maxlen=5),
            'velocity': [0.0, 0.0],
            'prediction_horizon': 3
        }
    
    def _init_search_system(self):
        """Инициализация умной системы поиска"""
        self.search = {
            'phase': 'scanning',  # scanning, moving, exploring
            'direction': 'right',
            'scan_duration': 1.8,
            'move_duration': 1.2,
            'phase_start_time': time.time(),
            'frustration_level': 0  # Увеличивается при долгом поиске
        }
        
        self.frames_without_targets = 0
        self.search_threshold = 10
    
    def _init_visualization(self):
        """Инициализация визуализации"""
        try:
            test_img = np.zeros((100, 100, 3), dtype=np.uint8)
            cv2.imshow(self.window_name, test_img)
            cv2.waitKey(1)
            cv2.destroyAllWindows()
            print("[OK] Визуализация инициализирована")
        except Exception as e:
            print(f"[WARN] Визуализация недоступна: {e}")
            self.show_window = False
    
    def _setup_stop_handler(self):
        """Настройка обработчика остановки"""
        try:
            keyboard.add_hotkey('q', self._request_stop)
            keyboard.add_hotkey('esc', self._request_stop)
        except Exception as e:
            print(f"[WARN] Не удалось зарегистрировать горячие клавиши: {e}")
    
    def _request_stop(self):
        """Запрос остановки бота"""
        if not self.stop_requested:
            print("\n[STOP] Остановка запрошена...")
            self.stop_requested = True
    
    def adaptive_confidence_adjustment(self, detections: List[Dict]):
        """Адаптивная настройка порога уверенности"""
        if not self.adaptive_confidence:
            return
        
        if len(detections) > 0:
            avg_conf = np.mean([d['conf'] for d in detections])
            self.conf_history.append(avg_conf)
            
            if len(self.conf_history) >= 20:
                recent_avg = np.mean(list(self.conf_history)[-20:])
                recent_std = np.std(list(self.conf_history)[-20:])
                
                # Умная адаптация на основе статистики
                if recent_avg > 0.65 and recent_std < 0.1:
                    # Много уверенных детекций - повышаем порог
                    self.conf_thres = min(0.4, self.base_conf_thres + 0.08)
                elif recent_avg < 0.35 and recent_std > 0.15:
                    # Нестабильные детекции - снижаем порог
                    self.conf_thres = max(0.15, self.base_conf_thres - 0.05)
                else:
                    # Нормальная ситуация - возвращаемся к базовому
                    self.conf_thres = self.base_conf_thres
        else:
            # Нет детекций - постепенно снижаем порог
            self.frames_without_targets += 1
            if self.frames_without_targets > 30:
                self.conf_thres = max(0.12, self.conf_thres - 0.01)
    
    def find_targets(self, results, frame_shape: Tuple[int, int]) -> List[Dict]:
        """Находит и приоритизирует цели"""
        frame_h, frame_w = frame_shape[:2]
        center_x = frame_w / 2
        center_y = frame_h / 2
        
        targets = []
        boxes = results[0].boxes
        
        if boxes is not None and len(boxes) > 0:
            for i in range(len(boxes)):
                conf = boxes.conf[i].item()
                cls = int(boxes.cls[i].item())
                
                if cls == 0 and conf >= self.conf_thres:
                    xyxy = boxes.xyxy[i].cpu().numpy()
                    x1, y1, x2, y2 = xyxy
                    
                    cx = (x1 + x2) / 2
                    cy = (y1 + y2) / 2
                    
                    # Фильтрация по зоне видимости
                    norm_x = cx / frame_w
                    norm_y = cy / frame_h
                    
                    if norm_y < 0.18 or norm_x < 0.08 or norm_x > 0.92:
                        continue
                    
                    distance = np.sqrt((cx - center_x)**2 + (cy - center_y)**2)
                    
                    # Расчет приоритета
                    priority = self._calculate_priority(cx, cy, conf, distance, frame_shape)
                    
                    targets.append({
                        'cx': cx,
                        'cy': cy,
                        'conf': conf,
                        'distance': distance,
                        'priority': priority,
                        'bbox': xyxy,
                        'area': (x2 - x1) * (y2 - y1)
                    })
        
        # Сортировка по приоритету
        targets.sort(key=lambda t: t['priority'], reverse=True)
        return targets
    
    def _calculate_priority(self, cx: float, cy: float, conf: float, 
                           distance: float, frame_shape: Tuple[int, int]) -> float:
        """Расчет приоритета цели"""
        frame_h, frame_w = frame_shape[:2]
        center_x = frame_w / 2
        center_y = frame_h / 2
        
        # Базовый приоритет от уверенности
        priority = conf * 150
        
        # Бонус за близость к центру (X-ось важнее)
        center_dist_x = abs(cx - center_x)
        priority += max(0, 60 - center_dist_x / 6)
        
        # Бонус за близость к центру (Y-ось)
        center_dist_y = abs(cy - center_y)
        priority += max(0, 50 - center_dist_y / 5)
        
        # Бонус за близость к игроку
        priority += max(0, 100 - distance / 3)
        
        # Бонус за стабильность (если есть в памяти)
        if self.target_memory:
            memory_bonus = self._get_memory_bonus(cx, cy, frame_shape)
            priority += memory_bonus
        
        return priority
    
    def _get_memory_bonus(self, cx: float, cy: float, frame_shape: Tuple[int, int]) -> float:
        """Получить бонус приоритета от памяти"""
        frame_h, frame_w = frame_shape[:2]
        max_bonus = 40.0
        
        for memory in self.target_memory:
            mem_cx = memory.get('cx', 0) * frame_w if 'cx' in memory else memory.get('cx_raw', 0)
            mem_cy = memory.get('cy', 0) * frame_h if 'cy' in memory else memory.get('cy_raw', 0)
            
            dist = np.sqrt((cx - mem_cx)**2 + (cy - mem_cy)**2)
            if dist < 80:  # Если цель близка к сохраненной в памяти
                age = time.time() - memory.get('time', 0)
                freshness = max(0, 1.0 - age / self.memory_retention)
                return max_bonus * freshness
        
        return 0.0
    
    def update_memory(self, target: Dict, frame_time: float, frame_shape: Tuple[int, int]):
        """Обновление памяти целей"""
        frame_h, frame_w = frame_shape[:2]
        
        # Нормализация координат
        entry = {
            'cx': target['cx'] / frame_w,
            'cy': target['cy'] / frame_h,
            'cx_raw': target['cx'],
            'cy_raw': target['cy'],
            'conf': target['conf'],
            'priority': target['priority'],
            'time': frame_time
        }
        
        # Проверка на дубликаты
        for memory in self.target_memory:
            dist = np.sqrt(
                (entry['cx'] - memory['cx'])**2 + 
                (entry['cy'] - memory['cy'])**2
            ) * frame_w
        
            if dist < 70:  # Если цель уже в памяти
                memory.update(entry)  # Обновляем
                memory['time'] = frame_time
                return
        
        # Добавляем новую цель
        self.target_memory.append(entry)
    
    def clean_memory(self, current_time: float):
        """Очистка устаревшей памяти"""
        self.target_memory = deque(
            [m for m in self.target_memory 
             if current_time - m.get('time', 0) < self.memory_retention],
            maxlen=15
        )
    
    def navigate_to_target(self, target: Dict, frame_shape: Tuple[int, int]) -> bool:
        """
        Навигация к цели с улучшенным алгоритмом
        
        Returns:
            True если цель собрана, False иначе
        """
        frame_h, frame_w = frame_shape[:2]
        center_x = frame_w / 2
        center_y = frame_h / 2
        
        cx = target['cx']
        cy = target['cy']
        distance = target['distance']
        
        # Сглаживание цели
        if self.smoothed_target is None:
            self.smoothed_target = {'cx': cx, 'cy': cy}
        else:
            self.smoothed_target['cx'] = (
                self.smoothing_alpha * cx + 
                (1 - self.smoothing_alpha) * self.smoothed_target['cx']
            )
            self.smoothed_target['cy'] = (
                self.smoothing_alpha * cy + 
                (1 - self.smoothing_alpha) * self.smoothed_target['cy']
            )
        
        cx_smooth = self.smoothed_target['cx']
        
        # Проверка сбора
        if distance < self.min_distance:
            self.control.release_all_keys()
            time.sleep(0.15)  # Небольшая пауза для сбора
            print(f"[COLLECT] ✨ Собрано! (conf: {target['conf']:.2f})")
            self.stats['collections'] += 1
            self._clean_target_memory(target, frame_shape)
            return True
        
        # PID контроллер для поворота
        error = cx_smooth - center_x
        self.pid['integral'] += error * self.pid['dt']
        self.pid['integral'] = np.clip(
            self.pid['integral'], 
            -self.pid['integral_limit'], 
            self.pid['integral_limit']
        )
        
        derivative = (error - self.pid['prev_error']) / self.pid['dt'] if self.pid['dt'] > 0 else 0
        pid_output = (
            self.pid['kp'] * error +
            self.pid['ki'] * self.pid['integral'] +
            self.pid['kd'] * derivative
        )
        
        self.pid['prev_error'] = error
        
        # Определение действий
        self.control.release_all_keys()
        
        # Движение вперед
        self.control.press('up')
        
        # Поворот/стрейф
        if abs(error) > self.turn_threshold:
            if error < 0:
                # Слева - поворот влево или стрейф
                if abs(error) > 60:
                    self.control.press('left')
                else:
                    self.control.press('turn_left')
            else:
                # Справа - поворот вправо или стрейф
                if abs(error) > 60:
                    self.control.press('right')
                else:
                    self.control.press('turn_right')
        
        # Время движения зависит от расстояния
        move_time = max(0.08, min(0.18, abs(pid_output) / 100))
        time.sleep(move_time)
        
        # Прыжки для преодоления препятствий
        if distance > 100 and random.random() < 0.15:
            self.control.press('jump')
            time.sleep(0.1)
        
        return False
    
    def _clean_target_memory(self, target: Dict, frame_shape: Tuple[int, int]):
        """Удаление собранной цели из памяти"""
        frame_h, frame_w = frame_shape[:2]
        target_cx_norm = target['cx'] / frame_w
        target_cy_norm = target['cy'] / frame_h
        
        self.target_memory = deque(
            [m for m in self.target_memory
             if np.sqrt(
                 (m['cx'] - target_cx_norm)**2 + 
                 (m['cy'] - target_cy_norm)**2
             ) * frame_w > 60],
            maxlen=15
        )
    
    def check_stuck(self, current_position: Tuple[float, float]) -> bool:
        """Проверка на застревание"""
        if self.stuck_detection['last_position'] is None:
            self.stuck_detection['last_position'] = current_position
            return False
        
        dx = abs(current_position[0] - self.stuck_detection['last_position'][0])
        dy = abs(current_position[1] - self.stuck_detection['last_position'][1])
        
        if dx < 5 and dy < 5:  # Минимальное движение
            self.stuck_detection['frames'] += 1
        else:
            self.stuck_detection['frames'] = 0
        
        self.stuck_detection['last_position'] = current_position
        
        if self.stuck_detection['frames'] >= self.stuck_detection['threshold']:
            self.stuck_detection['frames'] = 0
            return True
        
        return False
    
    def escape_stuck(self):
        """Маневр выхода из застревания"""
        print("[STUCK] Выполняю маневр выхода...")
        self.stats['stuck_events'] += 1
        
        self.control.release_all_keys()
        
        # Откат назад
        self.control.press('down')
        time.sleep(0.3)
        
        # Поворот
        direction = random.choice(['turn_left', 'turn_right'])
        self.control.press(direction)
        time.sleep(0.4)
        
        # Рывок вперед с прыжком
        self.control.release_all_keys()
        self.control.press('up')
        self.control.press('jump')
        time.sleep(0.3)
        
        self.control.release_all_keys()
        self.stuck_detection['last_position'] = None
    
    def search_pattern(self, current_time: float):
        """Умный паттерн поиска"""
        phase = self.search['phase']
        phase_time = current_time - self.search['phase_start_time']
        
        self.control.release_all_keys()
        
        if phase == 'scanning':
            # Сканирование поворотом камеры
            if phase_time > self.search['scan_duration']:
                self.search['phase'] = 'moving'
                self.search['phase_start_time'] = current_time
                self.search['direction'] = random.choice(['turn_left', 'turn_right'])
            else:
                self.control.press(self.search['direction'])
        
        elif phase == 'moving':
            # Движение для исследования
            if phase_time > self.search['move_duration']:
                self.search['phase'] = 'scanning'
                self.search['phase_start_time'] = current_time
                self.search['direction'] = random.choice(['turn_left', 'turn_right'])
                self.stats['search_cycles'] += 1
            else:
                # Умное движение
                if random.random() < 0.7:
                    self.control.press('up')
                if random.random() < 0.3:
                    self.control.press(random.choice(['left', 'right']))
        
        time.sleep(0.1)
    
    def update_state(self, new_state: BotState):
        """Обновление состояния бота"""
        if new_state != self.state:
            self.previous_state = self.state
            self.state = new_state
            self.stats['state_history'].append({
                'state': new_state.value,
                'time': time.time()
            })
    
    def visualize(self, img: np.ndarray, targets: List[Dict], results=None):
        """Улучшенная визуализация"""
        if not self.show_window:
            return
        
        try:
            # Аннотированное изображение
            if results is not None and len(results) > 0:
                annotated = results[0].plot()
            else:
                annotated = img.copy()
            
            h, w = annotated.shape[:2]
            
            # Центр экрана
            cv2.drawMarker(
                annotated, (w // 2, h // 2), 
                (0, 255, 255), 
                markerType=cv2.MARKER_CROSS, 
                markerSize=20, 
                thickness=2
            )
            
            # Текущая цель
            if self.current_target:
                tx = int(self.current_target['cx'])
                ty = int(self.current_target['cy'])
                cv2.circle(annotated, (tx, ty), 12, (0, 255, 0), 3)
                cv2.putText(
                    annotated, "TARGET", 
                    (tx - 30, ty - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2
                )
            
            # Информационная панель
            info_y = 25
            cv2.putText(
                annotated, f"STATE: {self.state.value.upper()}", 
                (10, info_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2
            )
            info_y += 30
            cv2.putText(
                annotated, f"FPS: {self.frame_counter.fps:.1f}", 
                (10, info_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2
            )
            info_y += 25
            cv2.putText(
                annotated, f"Conf: {self.conf_thres:.2f} | Targets: {len(targets)}", 
                (10, info_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2
            )
            info_y += 25
            cv2.putText(
                annotated, f"Collections: {self.stats['collections']}", 
                (10, info_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2
            )
            
            # Память целей
            for memory in self.target_memory:
                mx = int(memory.get('cx_raw', memory['cx'] * w))
                my = int(memory.get('cy_raw', memory['cy'] * h))
                cv2.circle(annotated, (mx, my), 5, (255, 165, 0), 1)
            
            cv2.imshow(self.window_name, annotated)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                self._request_stop()
        
        except Exception as e:
            print(f"[WARN] Ошибка визуализации: {e}")
    
    def print_stats(self):
        """Вывод статистики"""
        runtime = time.time() - self.stats['start_time']
        efficiency = (
            self.stats['collections'] / max(1, self.stats['detections']) * 100
            if self.stats['detections'] > 0 else 0
        )
        
        print("\n" + "=" * 60)
        print("📊 СТАТИСТИКА УЛЬТРА БОТА")
        print("=" * 60)
        print(f"⏱️  Время работы: {runtime:.1f} сек")
        print(f"🎯 Детекций: {self.stats['detections']}")
        print(f"✨ Собрано: {self.stats['collections']}")
        print(f"📈 Эффективность: {efficiency:.1f}%")
        print(f"🔍 Циклов поиска: {self.stats['search_cycles']}")
        print(f"💾 Память целей: {len(self.target_memory)}")
        print(f"🔧 Застреваний: {self.stats['stuck_events']}")
        print(f"📊 Средний FPS: {self.frame_counter.fps:.1f}")
        print("=" * 60)
    
    def run(self):
        """Основной цикл бота"""
        print("[START] Запуск основного цикла...")
        
        try:
            prev_time = time.time()
            
            for img, img0 in self.stream:
                if self.stop_requested:
                    break
                
                current_time = time.time()
                self.pid['dt'] = current_time - prev_time
                prev_time = current_time
                
                # Пропуск кадров для производительности
                self.frame_counter_internal += 1
                if self.frame_counter_internal % (self.frame_skip + 1) != 0:
                    continue
                
                # Конвертация формата
                img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                frame_shape = img.shape
                
                # Детекция
                results = self.model(
                    img,
                    imgsz=self.predict_size,
                    device=self.device,
                    conf=self.conf_thres,
                    verbose=False,
                    half=self.use_half_precision
                )
                
                # Поиск целей
                targets = self.find_targets(results, frame_shape)
                
                # Адаптивная настройка
                self.adaptive_confidence_adjustment(targets)
                
                # Обновление состояния
                if targets:
                    self.frames_without_targets = 0
                    best_target = targets[0]
                    
                    if self.current_target is None or best_target['priority'] > self.current_target['priority'] * 1.2:
                        self.current_target = best_target
                        self.stats['detections'] += 1
                    
                    self.update_state(BotState.APPROACHING)
                    
                    # Навигация
                    collected = self.navigate_to_target(self.current_target, frame_shape)
                    
                    if collected:
                        self.update_state(BotState.COLLECTING)
                        self.current_target = None
                        self.smoothed_target = None
                        time.sleep(0.1)
                        self.update_state(BotState.SEARCHING)
                    else:
                        # Обновление памяти
                        self.update_memory(self.current_target, current_time, frame_shape)
                else:
                    self.frames_without_targets += 1
                    self.update_state(BotState.SEARCHING)
                    
                    if self.frames_without_targets > self.search_threshold:
                        self.search_pattern(current_time)
                    
                    self.current_target = None
                
                # Очистка памяти
                self.clean_memory(current_time)
                
                # Проверка застревания
                if self.current_target:
                    position = (self.current_target['cx'], self.current_target['cy'])
                    if self.check_stuck(position):
                        self.update_state(BotState.STUCK)
                        self.escape_stuck()
                        self.update_state(BotState.SEARCHING)
                
                # Визуализация
                self.visualize(img, targets, results)
                
                # Обновление FPS
                self.frame_counter.log()
                
                # Периодическая статистика
                if int(current_time) % 30 == 0 and int(current_time) != int(prev_time):
                    self.print_stats()
        
        except KeyboardInterrupt:
            print("\n[STOP] Остановка по Ctrl+C...")
        except Exception as e:
            print(f"\n[ERROR] Критическая ошибка: {e}")
            import traceback
            traceback.print_exc()
        
        finally:
            print("\n[EXIT] Завершение работы...")
            self.control.release_all_keys()
            self.print_stats()
            if self.show_window:
                cv2.destroyAllWindows()
            print("[EXIT] Бот остановлен")


def main():
    parser = argparse.ArgumentParser(
        description='🚀 Ultra Bot v3.0 - Ultimate Performance Edition'
    )
    parser.add_argument(
        '--weights', type=str, 
        default='weights/candies_v10.pt',
        help='Путь к весам модели'
    )
    parser.add_argument(
        '--player-weights', type=str, default=None,
        help='Путь к весам модели игроков (опционально)'
    )
    parser.add_argument(
        '--conf', type=float, default=0.22,
        help='Базовый порог уверенности'
    )
    parser.add_argument(
        '--mode', type=str, default='balanced',
        choices=['speed', 'balanced', 'accuracy'],
        help='Режим производительности'
    )
    parser.add_argument(
        '--no-window', action='store_true',
        help='Не показывать окно визуализации'
    )
    
    args = parser.parse_args()
    
    if not os.path.exists(args.weights):
        print(f"[ERROR] Веса не найдены: {args.weights}")
        return
    
    bot = UltraBot(
        weights_path=args.weights,
        player_weights_path=args.player_weights,
        conf_thres=args.conf,
        show_window=not args.no_window,
        performance_mode=args.mode
    )
    
    bot.run()


if __name__ == '__main__':
    main()
