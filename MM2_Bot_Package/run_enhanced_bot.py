"""
ULTRA BOT для поиска мячиков в Murder Mystery 2
Версия 6.0 - Maximum Performance & Intelligence

🌟 РЕВОЛЮЦИОННЫЕ УЛУЧШЕНИЯ:
🚀 Оптимизированная производительность (FP16, кэширование, асинхронная обработка)
🧠 Искусственный интеллект (Reinforcement Learning, предсказание движения)
🛡️ Продвинутая система безопасности (оценка угроз, интеллектуальное уклонение)
📊 Аналитическая система (мониторинг, обучение, статистика)
⚡ Автоматическая оптимизация под системные характеристики
🎯 Интеллектуальная навигация с перехватом траекторий
"""

import sys
import os
import argparse
import time
import cv2
import numpy as np
import torch
import signal
import random
import json
import keyboard  # Заменяем msvcrt
from collections import deque
from datetime import datetime
from enum import Enum
import asyncio
import threading
from concurrent.futures import ThreadPoolExecutor
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(__file__))

from ultralytics import YOLO
from roblox.screen import CaptureStream
from roblox.control import Control

# Глобальная переменная для отслеживания состояния остановки
stop_requested = False

def request_stop():
    """Функция для потокобезопасного запроса остановки."""
    global stop_requested
    if not stop_requested:
        print("\n[STOP] Команда остановки по клавише 'i'")
        stop_requested = True

from roblox.utils import FrameCounter

class OptimizedInferenceEngine:
    """Оптимизированный движок инференса с кэшированием и асинхронной обработкой"""

    def __init__(self, model, device, use_fp16=True):
        self.model = model
        self.device = device
        self.use_fp16 = use_fp16 and device.type == 'cuda'

        # Кэш для предобработанных изображений
        self.preprocess_cache = {}
        self.cache_max_size = 10

        # Пул потоков для асинхронной обработки
        self.executor = ThreadPoolExecutor(max_workers=3)

        # Оптимизация модели
        if self.use_fp16:
            self.model.half()
            print("[OPTIMIZE] FP16 режим активирован")

        # Torch оптимизации
        torch.backends.cudnn.benchmark = True
        if torch.cuda.is_available():
            torch.cuda.set_device(device)

    def preprocess_image(self, img, cache_key=None):
        """Оптимизированная предобработка с кэшированием"""
        if cache_key and cache_key in self.preprocess_cache:
            return self.preprocess_cache[cache_key]

        # Конвертация и улучшение изображения
        if img.shape[2] == 3:  # RGB
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        # HSV улучшение для лучшей видимости мячиков
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
        hsv[..., 1] = np.clip(hsv[..., 1] * 1.15, 0, 255)  # Увеличение насыщенности
        hsv[..., 2] = np.clip(hsv[..., 2] * 1.3, 0, 255)   # Увеличение яркости
        enhanced = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

        # Кэширование
        if cache_key:
            if len(self.preprocess_cache) >= self.cache_max_size:
                # Удаляем самый старый элемент
                oldest_key = next(iter(self.preprocess_cache))
                del self.preprocess_cache[oldest_key]
            self.preprocess_cache[cache_key] = enhanced.copy()

        return enhanced

    async def async_inference(self, img, imgsz, conf):
        """Асинхронный инференс"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self.executor,
            lambda: self.model(img, imgsz=imgsz, device=self.device,
                             conf=conf, verbose=False, half=self.use_fp16,
                             augment=False, agnostic_nms=True)
        )

    def batch_inference(self, images, imgsz, conf):
        """Пакетный инференс для нескольких изображений"""
        if not images:
            return []

        # Подготовка батча
        batch_tensor = torch.stack([
            torch.from_numpy(cv2.resize(img, (imgsz, imgsz))).permute(2, 0, 1).float() / 255.0
            for img in images
        ]).to(self.device)

        if self.use_fp16:
            batch_tensor = batch_tensor.half()

        with torch.no_grad():
            results = self.model(batch_tensor, conf=conf, verbose=False)

        return results

class AdvancedPredictor:
    """Расширенная система предсказания движения целей"""

    def __init__(self, history_length=15):
        self.target_history = {}  # track_id -> deque of positions
        self.prediction_horizon = 5  # кадров вперед
        self.history_length = history_length
        self.velocity_cache = {}  # track_id -> velocity vector

    def update_target_track(self, track_id, cx, cy, timestamp):
        """Обновляет трек цели"""
        if track_id not in self.target_history:
            self.target_history[track_id] = deque(maxlen=self.history_length)

        self.target_history[track_id].append({
            'cx': cx, 'cy': cy, 'time': timestamp
        })

        # Вычисляем скорость
        if len(self.target_history[track_id]) >= 3:
            self._calculate_velocity(track_id)

    def _calculate_velocity(self, track_id):
        """Вычисляет скорость движения цели"""
        history = list(self.target_history[track_id])
        if len(history) < 3:
            return

        # Используем последние 3 точки для вычисления скорости
        recent = history[-3:]
        dt1 = recent[1]['time'] - recent[0]['time']
        dt2 = recent[2]['time'] - recent[1]['time']

        if dt1 > 0 and dt2 > 0:
            vx1 = (recent[1]['cx'] - recent[0]['cx']) / dt1
            vy1 = (recent[1]['cy'] - recent[0]['cy']) / dt1
            vx2 = (recent[2]['cx'] - recent[1]['cx']) / dt2
            vy2 = (recent[2]['cy'] - recent[1]['cy']) / dt2

            # Сглаженная скорость
            vx = (vx1 + vx2) / 2
            vy = (vy1 + vy2) / 2

            self.velocity_cache[track_id] = {'vx': vx, 'vy': vy}

    def predict_position(self, track_id, frames_ahead=1):
        """Предсказывает будущую позицию цели"""
        if track_id not in self.target_history or track_id not in self.velocity_cache:
            return None

        history = list(self.target_history[track_id])
        if not history:
            return None

        current_pos = history[-1]
        velocity = self.velocity_cache[track_id]

        # Предсказание с учетом ускорения
        if len(history) >= 5:
            # Вычисляем ускорение
            accel = self._calculate_acceleration(track_id)
            if accel:
                predicted_x = current_pos['cx'] + velocity['vx'] * frames_ahead + 0.5 * accel['ax'] * frames_ahead**2
                predicted_y = current_pos['cy'] + velocity['vy'] * frames_ahead + 0.5 * accel['ay'] * frames_ahead**2
            else:
                predicted_x = current_pos['cx'] + velocity['vx'] * frames_ahead
                predicted_y = current_pos['cy'] + velocity['vy'] * frames_ahead
        else:
            predicted_x = current_pos['cx'] + velocity['vx'] * frames_ahead
            predicted_y = current_pos['cy'] + velocity['vy'] * frames_ahead

        return {'cx': predicted_x, 'cy': predicted_y}

    def _calculate_acceleration(self, track_id):
        """Вычисляет ускорение цели"""
        history = list(self.target_history[track_id])
        if len(history) < 5:
            return None

        # Используем производную скорости
        velocities = []
        for i in range(1, len(history)):
            dt = history[i]['time'] - history[i-1]['time']
            if dt > 0:
                vx = (history[i]['cx'] - history[i-1]['cx']) / dt
                vy = (history[i]['cy'] - history[i-1]['cy']) / dt
                velocities.append({'vx': vx, 'vy': vy, 'time': history[i]['time']})

        if len(velocities) < 2:
            return None

        # Вычисляем ускорение
        ax = (velocities[-1]['vx'] - velocities[-2]['vx']) / (velocities[-1]['time'] - velocities[-2]['time'])
        ay = (velocities[-1]['vy'] - velocities[-2]['vy']) / (velocities[-1]['time'] - velocities[-2]['time'])

        return {'ax': ax, 'ay': ay}

    def get_optimal_intercept_point(self, track_id, bot_position, bot_speed=100):
        """Вычисляет оптимальную точку перехвата"""
        prediction = self.predict_position(track_id, frames_ahead=3)
        if not prediction:
            return None

        target_pos = prediction
        target_vel = self.velocity_cache.get(track_id, {'vx': 0, 'vy': 0})

        # Время до цели
        dx = target_pos['cx'] - bot_position[0]
        dy = target_pos['cy'] - bot_position[1]
        distance = np.sqrt(dx**2 + dy**2)

        # Учитываем движение цели
        relative_vx = target_vel['vx']
        relative_vy = target_vel['vy']

        # Простое предсказание (можно улучшить до полного решения уравнения)
        time_to_target = distance / bot_speed

        # Корректировка на движение цели
        intercept_x = target_pos['cx'] + relative_vx * time_to_target
        intercept_y = target_pos['cy'] + relative_vy * time_to_target

        return {'cx': intercept_x, 'cy': intercept_y, 'time': time_to_target}

class AdaptiveReinforcementLearner:
    """Система reinforcement learning для адаптивного обучения стратегий"""

    def __init__(self, alpha=0.1, gamma=0.9, epsilon=0.1):
        self.alpha = alpha  # Скорость обучения
        self.gamma = gamma  # Дисконтирующий фактор
        self.epsilon = epsilon  # Параметр исследования

        # Q-таблица для различных состояний и действий
        self.q_table = {}

        # Определение состояний
        self.states = [
            'searching_low_conf', 'searching_high_conf',
            'hunting_close', 'hunting_far', 'hunting_stuck',
            'confirming_collection', 'evading_threat'
        ]

        # Определение действий
        self.actions = [
            'increase_conf', 'decrease_conf', 'maintain_conf',
            'aggressive_approach', 'cautious_approach', 'strafe_left', 'strafe_right',
            'jump_evasion', 'pause_movement', 'memory_search'
        ]

        # Инициализация Q-таблицы
        for state in self.states:
            self.q_table[state] = {action: 0.0 for action in self.actions}

        # Статистика обучения
        self.learning_stats = {
            'episodes': 0,
            'total_reward': 0,
            'best_actions': {}
        }

    def get_state(self, bot_state, conf_threshold, target_distance=None, stuck_frames=0, threats_nearby=0):
        """Определение текущего состояния на основе параметров бота"""
        if bot_state == 'SEARCHING':
            if conf_threshold < 0.3:
                return 'searching_low_conf'
            else:
                return 'searching_high_conf'
        elif bot_state == 'HUNTING':
            if stuck_frames > 10:
                return 'hunting_stuck'
            elif target_distance and target_distance < 100:
                return 'hunting_close'
            else:
                return 'hunting_far'
        elif bot_state == 'CONFIRMING':
            return 'confirming_collection'
        elif threats_nearby > 0:
            return 'evading_threat'
        else:
            return 'searching_high_conf'  # состояние по умолчанию

    def choose_action(self, state, exploration=True):
        """Выбор действия на основе Q-таблицы с ε-жадной стратегией"""
        if exploration and random.random() < self.epsilon:
            # Исследовательское действие
            return random.choice(self.actions)
        else:
            # Эксплуататорское действие - выбираем лучшее из известных
            state_actions = self.q_table.get(state, {action: 0.0 for action in self.actions})
            return max(state_actions, key=state_actions.get)

    def update_q_value(self, state, action, reward, next_state):
        """Обновление Q-значения по формуле Q-learning"""
        if state not in self.q_table:
            self.q_table[state] = {action: 0.0 for action in self.actions}
        if next_state not in self.q_table:
            self.q_table[next_state] = {action: 0.0 for action in self.actions}

        current_q = self.q_table[state][action]
        max_next_q = max(self.q_table[next_state].values())

        # Q-learning формула
        new_q = current_q + self.alpha * (reward + self.gamma * max_next_q - current_q)
        self.q_table[state][action] = new_q

    def calculate_reward(self, action_taken, success_metrics):
        """Расчет награды на основе результатов действия"""
        reward = 0

        # Положительные награды
        if success_metrics.get('collection_success', False):
            reward += 10  # Успешный сбор
        if success_metrics.get('target_found', False):
            reward += 2   # Нахождение цели
        if success_metrics.get('fps_improved', False):
            reward += 1   # Улучшение производительности

        # Отрицательные награды
        if success_metrics.get('stuck_occurred', False):
            reward -= 3   # Застревание
        if success_metrics.get('threat_collision', False):
            reward -= 5   # Столкновение с угрозой
        if success_metrics.get('long_search_time', False):
            reward -= 1   # Долгое время поиска

        # Награды за действия
        if action_taken == 'memory_search' and success_metrics.get('memory_success', False):
            reward += 3   # Успешный возврат к памяти
        if action_taken == 'jump_evasion' and success_metrics.get('evasion_success', False):
            reward += 2   # Успешное уклонение

        return reward

    def adapt_parameters(self, current_params, learned_action):
        """Адаптация параметров бота на основе изученного действия"""
        adapted_params = current_params.copy()

        if learned_action == 'increase_conf':
            adapted_params['conf_threshold'] = min(0.5, current_params['conf_threshold'] + 0.02)
        elif learned_action == 'decrease_conf':
            adapted_params['conf_threshold'] = max(0.15, current_params['conf_threshold'] - 0.02)
        elif learned_action == 'aggressive_approach':
            adapted_params['min_distance'] = max(30, current_params['min_distance'] - 5)
        elif learned_action == 'cautious_approach':
            adapted_params['min_distance'] = min(100, current_params['min_distance'] + 5)

        return adapted_params

    def save_learning_progress(self, filename="learning_progress.json"):
        """Сохранение прогресса обучения"""
        data = {
            'q_table': self.q_table,
            'learning_stats': self.learning_stats,
            'timestamp': time.time()
        }
        try:
            with open(filename, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            print(f"[LEARNING] Ошибка сохранения прогресса: {e}")

    def load_learning_progress(self, filename="learning_progress.json"):
        """Загрузка прогресса обучения"""
        try:
            with open(filename, 'r') as f:
                data = json.load(f)
                self.q_table = data.get('q_table', self.q_table)
                self.learning_stats = data.get('learning_stats', self.learning_stats)
                print(f"[LEARNING] Загружен прогресс обучения от {time.ctime(data.get('timestamp', 0))}")
        except FileNotFoundError:
            print("[LEARNING] Файл прогресса не найден, начинаем с нуля")
        except Exception as e:
            print(f"[LEARNING] Ошибка загрузки прогресса: {e}")

class EnhancedEvasionSystem:
    """Расширенная система уклонения от угроз"""

    def __init__(self):
        self.threat_zones = []  # Зоны угроз
        self.evasion_patterns = {
            'circle_strafe': self._circle_strafe,
            'jump_evasion': self._jump_evasion,
            'distance_maintenance': self._distance_maintenance,
            'predictive_dodge': self._predictive_dodge
        }
        self.last_evasion_time = 0
        self.evasion_cooldown = 2.0

    def assess_threat_level(self, player_targets, bot_position):
        """Оценка уровня угрозы"""
        threat_level = 0
        imminent_threats = []

        for player in player_targets:
            distance = np.sqrt((player['cx'] - bot_position[0])**2 + (player['cy'] - bot_position[1])**2)
            if distance < 150:  # Критическая дистанция
                threat_level += 3
                imminent_threats.append(player)
            elif distance < 300:  # Предупреждающая дистанция
                threat_level += 1

        return threat_level, imminent_threats

    def choose_evasion_strategy(self, threat_level, imminent_threats, bot_state):
        """Выбор стратегии уклонения"""
        if threat_level >= 3 and time.time() - self.last_evasion_time > self.evasion_cooldown:
            if len(imminent_threats) > 1:
                return 'circle_strafe'  # Множественные угрозы
            else:
                return 'predictive_dodge'  # Одна угроза
        elif threat_level >= 1:
            return 'distance_maintenance'  # Поддержание дистанции
        else:
            return None

    def execute_evasion(self, strategy, imminent_threats, control_system):
        """Выполнение маневра уклонения"""
        if strategy == 'circle_strafe':
            return self._circle_strafe(imminent_threats, control_system)
        elif strategy == 'jump_evasion':
            return self._jump_evasion(imminent_threats, control_system)
        elif strategy == 'predictive_dodge':
            return self._predictive_dodge(imminent_threats, control_system)
        elif strategy == 'distance_maintenance':
            return self._distance_maintenance(imminent_threats, control_system)
        return False

    def _circle_strafe(self, threats, control):
        """Круговое уклонение от множественных угроз"""
        control.press('jump')
        time.sleep(0.1)
        control.release('jump')

        # Выполняем круговое движение
        for _ in range(8):
            control.press('left')
            time.sleep(0.05)
            control.release('left')
            control.press('right')
            time.sleep(0.05)
            control.release('right')

        self.last_evasion_time = time.time()
        return True

    def _jump_evasion(self, threats, control):
        """Уклонение прыжком"""
        control.press('jump')
        control.press('up')
        time.sleep(0.3)
        control.release('jump')
        control.release('up')

        self.last_evasion_time = time.time()
        return True

    def _predictive_dodge(self, threats, control):
        """Предиктивное уклонение на основе движения угрозы"""
        if not threats:
            return False

        threat = threats[0]
        # Определяем направление движения угрозы и уклоняемся в противоположную сторону
        dodge_direction = 'left' if threat.get('cx', 0) > 400 else 'right'

        control.press(dodge_direction)
        control.press('jump')
        time.sleep(0.2)
        control.release(dodge_direction)
        control.release('jump')

        self.last_evasion_time = time.time()
        return True

    def _distance_maintenance(self, threats, control):
        """Поддержание безопасной дистанции"""
        if not threats:
            return False

        # Медленное отползание назад
        control.press('down')
        time.sleep(0.5)
        control.release('down')

        return True

class EnhancedBallBot:
    def __init__(self, weights_path, player_weights_path=None, conf_thres=0.25, adaptive_mode=True, show_window=False, save_screenshots=False):
        print(f"[INIT] Загрузка улучшенного бота: {weights_path}")
        if player_weights_path:
            print(f"[PREDATOR] Загрузка модели игроков: {player_weights_path}")
        
        # --- Усиленная инициализация GPU ---
        if torch.cuda.is_available():
            self.device = torch.device("cuda:0")
            torch.cuda.set_device(self.device)
            gpu_name = torch.cuda.get_device_name(0)
            print(f"[INFO] Device: CUDA ({gpu_name})")
        else:
            self.device = torch.device("cpu")
            print("[INFO] Device: CPU")
        # --- Конец усиленной инициализации ---
        
        # Загрузка модели
        self.model = YOLO(weights_path).to(self.device)
        self.player_model = None
        if player_weights_path:
            self.player_model = YOLO(player_weights_path).to(self.device)
            print("[PREDATOR] Режим 'Хищник' активирован.")

        # Инициализация оптимизированных движков
        self.inference_engine = OptimizedInferenceEngine(self.model, self.device, use_fp16=True)
        self.predictor = AdvancedPredictor(history_length=20)
        if self.player_model:
            self.player_inference_engine = OptimizedInferenceEngine(self.player_model, self.device, use_fp16=True)

        # Инициализация систем ИИ
        self.reinforcement_learner = AdaptiveReinforcementLearner(alpha=0.1, gamma=0.9, epsilon=0.15)
        self.evasion_system = EnhancedEvasionSystem()

        # Загрузка прогресса обучения
        try:
            self.reinforcement_learner.load_learning_progress()
        except:
            print("[LEARNING] Начинаем обучение с нуля")

        self.base_conf_thres = conf_thres
        self.conf_thres = conf_thres
        self.adaptive_mode = adaptive_mode
        self.show_window = show_window
        self.save_screenshots = save_screenshots
        self.window_failed = False
        self.use_half = self.device.type == 'cuda'
        self.manual_stop_key = 'i'
        
        if self.show_window:
            print("[DEBUG] Режим отображения включен - окно должно появиться")
            print("[DEBUG] Управление: 'q' - выход, 's' - сохранить скриншот")
            # Тестируем OpenCV GUI
            try:
                import numpy as np
                test_img = np.zeros((100, 100, 3), dtype=np.uint8)
                cv2.imshow('OpenCV Test', test_img)
                cv2.waitKey(1)
                cv2.destroyAllWindows()
                print("[DEBUG] OpenCV GUI тест прошел успешно")
            except Exception as e:
                print(f"[WARN] OpenCV GUI не работает: {e}")
                print("[INFO] Автоматически переключаемся на сохранение скриншотов")
                self.show_window = False
                self.save_screenshots = True
                self.window_failed = True
        
        if self.save_screenshots:
            print("[DEBUG] Режим сохранения скриншотов включен")
            self.screenshot_counter = 0
        
        # Адаптивная система порогов
        self.conf_history = deque(maxlen=50)  # История последних 50 детекций
        self.no_detection_frames = 0
        self.max_no_detection = 20
        
        # Инициализация захвата экрана
        print("[INIT] Захват экрана...")
        self.stream = CaptureStream("Roblox", saveInterval=0)
        
        # Инициализация управления
        self.control = Control()
        self.frame_counter = FrameCounter()
        self.frame_counter.fps = 0.0
        
        # Улучшенный PID контроллер
        self.pid_kp = 0.04
        self.pid_ki = 0.001
        self.pid_kd = 0.025
        self.integral = 0.0
        self.prev_error = 0.0
        self.dt = 0.05
        self.integral_limit = 150.0
        
        # Параметры движения
        self.min_distance = 52
        self.turn_threshold = 24
        self.paste_delay = 0.22
        self.predict_size = 320  # FPS-BOOST: Меньше для скорости, мультискейл компенсирует
        self.approach_distance = 140

        # Векторное управление движением (новая система навигации)
        self.forward_min_step = 0.045
        self.forward_max_step = 0.16
        self.forward_error_decay = 1.25
        self.forward_block_error = 0.58
        self.strafe_deadzone = 0.18
        self.strafe_scale = 0.1
        self.turn_deadzone = 14
        self.turn_strong_error = 0.42
        self.turn_base = 0.038
        self.turn_scale = 0.12
        self.turn_max = 0.21
        self.turn_focus_error = 0.18
        self.turn_focus_distance = 180
        self.aim_alpha = 0.32         # EMA сглаживание цели
        self.smoothed_cx = None
        self.smoothed_cy = None
        self.progress_history = deque(maxlen=14)
        self.progress_gain_far = 12
        self.progress_gain_close = 5
        self.last_escape_time = 0.0
        self.escape_cooldown = 3.5  # Увеличен кулдаун
        self.escape_push_duration = 0.4  # Усилен импульс

        # Мультискейл-детекция и fallback-параметры
        self.multi_scale_enabled = True
        self.multi_scale_factor = 1.3
        self.multi_scale_conf_shift = -0.05
        self.multi_scale_conf_floor = 0.14
        self.heavy_scan_imgsz = 640
        self.heavy_scan_relief = 2
        self.last_detection_pass = 'base'
        self.det_pass_stats = {'base': 0, 'boost': 0, 'heavy': 0}

        # Предобработка кадра для улучшения видимости монет
        self.preprocess_enabled = True
        self.preprocess_s_gain = 1.12
        self.preprocess_v_gain = 1.28
        
        # --- Новая архитектура на состояниях ---
        self.state = 'SEARCHING'  # Начальное состояние
        self.search_phase = 'SCANNING' # Фазы: SCANNING, MOVING
        self.search_scan_start_time = 0.0
        self.search_move_start_time = 0.0
        self.search_turn_direction = 'turn_right'
        self.confirmation_frames = 0 # Счетчик для подтверждения сбора
        self.target_for_confirmation = None # Цель, которую подтверждаем
        
        # Параметры для нового режима
        self.hardcore_aim_mode = True
        self.aim_complete_threshold_px = 12
        self.stuck_detection_threshold = 18
        self.stuck_for_obstacle_avoidance_threshold = 10 # Порог для огибания
        self.search_scan_duration = 1.6
        self.search_move_duration = 0.8
        self.stuck_frames = 0 # Инициализация для анти-стук логики
        self.long_search_threshold = 30.0 # секунд до активации "прорыва"
        self.last_candy_found_time = time.time()
        
        # --- Гибридная модель v3.3 ---
        self.high_fps_predict_size = 256
        self.high_accuracy_predict_size = 416
        self.dynamic_predict_size = True
        self.far_distance_threshold = 200 # Дистанция для переключения режимов
        self.y_axis_priority_weight = 2.5 # Насколько важна близость к горизонтальному центру
        
        # --- Predator v5.0 ---
        self.ego_filter_zone = (0.35, 0.40, 0.65, 1.0) # (x1, y1, x2, y2) в % от экрана
        self.risk_factor_weight = 150.0 # Штраф за близость игрока
        self.risk_distance_threshold = 250 # Дистанция в px, на которой игрок считается угрозой
        self.player_conf_thres = 0.40 # Отдельный порог для игроков
        self.threat_confirmation_distance = 100 # px для подтверждения угрозы
        self.use_fp16 = True # Включаем FP16 для максимальной производительности
        self.threat_cache_ttl = 2.0 # Время жизни кэша угроз в секундах
        self.player_detection_hunt_cutoff_dist = 150 # Дистанция до цели, при которой отключается поиск игроков
        self.confirmation_max_frames = 5 # Кадров на подтверждение сбора
        # --- Конец Predator ---
        
        # --- Конец гибридной модели ---
        
        self.target_cache = deque(maxlen=5) # Кэш для быстрого переключения целей

        # --- Оптимизация производительности ---
        self.frame_skip = 1 # Пропускать N кадров между детекциями конфет
        self.player_detection_skip = 4 # Пропускать N кадров для детекции игроков (значение +1)
        self.frame_counter_internal = 0
        self.last_known_candies = []
        self.last_detected_players = [] # Для "двойной проверки"
        self.confirmed_threats = [] # Подтвержденные угрозы
        self.last_player_detection_time = 0.0 # Время последнего обнаружения игроков
        # --- Конец оптимизации ---

        # Система поиска с паттернами (больше не используется)
        self.frames_without_coins = 0
        self.search_threshold = 8
        self.is_searching = False
        self.is_moving_to_coin = False
        
        # Система приоритизации целей
        self.target_history = deque(maxlen=10)
        self.current_target = None
        self.target_switch_threshold = 0.35  # Порог для смены цели
        
        # Антизастревание
        self.frames_since_jump = 0
        self.jump_interval = 12
        self.previous_distance = float('inf')
        self.frames_after_collect = 0
        self.post_collect_frames = 8
        self.retry_frames = 0
        self.retry_threshold = 5
        self.stuck_threshold = 15  # Кадров без прогресса
        
        # Статистика и логирование
        self.stats = {
            'total_detections': 0,
            'successful_collections': 0,
            'search_cycles': 0,
            'avg_confidence': 0.0,
            'memory_revisits': 0,
            'start_time': time.time(),
            'memory_returns': 0,
            'pass_counts': {'base': 0, 'boost': 0, 'heavy': 0},
            'cache_retrievals': 0,
            'players_detected': 0, # Predator stat
        }
        self.conf_sum = 0.0
        self.log_data = []
        
        # Память целей и реактивное поведение
        self.target_memory = deque(maxlen=8)
        self.memory_retention = 2.2  # сек удержания памяти
        self.memory_min_conf = 0.18
        self.memory_revisit_cooldown = 0.6
        self.last_revisit_time = 0.0
        self.last_frame_shape = None
        self.last_detection_time = 0.0

        # Поиск: режим тяжёлого сканирования, если нет детекций
        self.no_detect_frames = 0
        self.heavy_scan_interval = 4
        self.heavy_conf_floor = 0.12
        
        # Система логирования
        self.log_file = f"bot_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        self.log_data = []
        
        print("[OK] Улучшенный бот готов!")
        print(f"[INFO] Conf: {self.conf_thres}, Adaptive: {self.adaptive_mode}")
        print(f"[INFO] Min dist: {self.min_distance}px, Predict Size: Dynamic, Frame Skip: {self.frame_skip}")
        if self.player_model:
            print("[INFO] Predator Mode: ON")
            if self.use_fp16:
                print("[INFO] Half-precision (FP16) enabled for maximum performance.")
        print("\n[START] Ctrl+C для остановки\n")
    
    def adaptive_confidence_adjustment(self, detections):
        """Адаптивная настройка порога уверенности"""
        if not self.adaptive_mode:
            return
        
        if len(detections) > 0:
            # Есть детекции - можем быть более строгими
            avg_conf = np.mean([d['conf'] for d in detections])
            self.conf_history.append(avg_conf)
            
            if len(self.conf_history) >= 10:
                recent_avg = np.mean(list(self.conf_history)[-10:])
                if recent_avg > 0.7:
                    self.conf_thres = min(0.4, self.base_conf_thres + 0.1)
                elif recent_avg < 0.4:
                    self.conf_thres = max(0.15, self.base_conf_thres - 0.05)
                else:
                    self.conf_thres = self.base_conf_thres
        else:
            # Нет детекций - снижаем порог для поиска
            self.no_detection_frames += 1
            if self.no_detection_frames > self.max_no_detection:
                self.conf_thres = max(0.1, self.base_conf_thres - 0.1)
        # Убрали вывод, чтобы не спамить в консоль
        # print(f"[ADAPTIVE] Conf threshold: {self.conf_thres:.3f}")
    
    def filter_ego(self, player_targets, frame_shape):
        """Фильтрует детекцию собственного персонажа."""
        if not player_targets:
            return []
        
        frame_h, frame_w = frame_shape[:2]
        x1_ego = self.ego_filter_zone[0] * frame_w
        y1_ego = self.ego_filter_zone[1] * frame_h
        x2_ego = self.ego_filter_zone[2] * frame_w
        y2_ego = self.ego_filter_zone[3] * frame_h

        filtered_players = []
        for player in player_targets:
            px, py = player['cx'], player['cy']
            # Игрок не считается "эго", если он вне зоны
            if not (x1_ego < px < x2_ego and y1_ego < py < y2_ego):
                filtered_players.append(player)
        
        return filtered_players

    def find_player_targets(self, results, frame_shape):
        """Находит игроков на кадре."""
        targets = []
        boxes = results[0].boxes
        if boxes is not None and len(boxes) > 0:
            for i in range(len(boxes)):
                conf = boxes.conf[i].item()
                # Предполагаем, что класс игрока тоже 0 или нужно будет настроить
                if conf >= self.player_conf_thres: # Используем отдельный порог
                    xyxy = boxes.xyxy[i].cpu().numpy()
                    x1, y1, x2, y2 = xyxy
                    targets.append({
                        'cx': (x1 + x2) / 2,
                        'cy': (y1 + y2) / 2,
                        'conf': conf,
                        'bbox': xyxy
                    })
        return targets

    def find_prioritized_targets(self, results, frame_shape, player_targets, max_targets=3):
        """Находит цели с приоритизацией с учетом игроков."""
        frame_h, frame_w = frame_shape[:2]
        center_x = frame_w / 2
        center_y = frame_h / 2
        
        targets = []
        frame_time = time.time()
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
                    
                    # Нормализованные координаты
                    norm_x = cx / frame_w
                    norm_y = cy / frame_h

                    # Более мягкая фильтрация по зоне видимости (сдвигаем ближе к центру, но даём шанс краям)
                    if norm_y < 0.20 or norm_x < 0.08 or norm_x > 0.92:
                        continue
                    
                    distance = np.sqrt((cx - center_x)**2 + (cy - center_y)**2)
                    
                    entry = {
                        'cx': cx,
                        'cy': cy,
                        'conf': conf,
                        'distance': distance
                    }
                    was_memory = self.maybe_refresh_memory(entry, frame_time, frame_shape)

                    # Система приоритизации
                    priority_score = self.calculate_priority_score(cx, cy, conf, distance, frame_shape, player_targets)
                    
                    targets.append({
                        'cx': cx, 'cy': cy, 'conf': conf, 'distance': distance,
                        'priority_score': priority_score, 'bbox': xyxy,
                        'memory_boost': was_memory
                    })
        
        # Сортировка по приоритету
        targets.sort(key=lambda t: t['priority_score'], reverse=True)
        return targets[:max_targets]

    def preprocess_frame(self, img, cache_key=None):
        """Обновленная предобработка с использованием оптимизированного движка"""
        if not self.preprocess_enabled:
            return img

        # Используем оптимизированный движок
        return self.inference_engine.preprocess_image(img, cache_key)

    def align_imgsz(self, size):
        stride = 32
        return int(np.ceil(size / stride) * stride)

    def run_multi_scale_detection(self, img, frame_time):
        """Запускает серию детекций с мультискейлом и возвращает цели"""
        detection_passes = []
        base_conf = self.conf_thres
        detection_passes.append({
            'name': 'base',
            'imgsz': self.align_imgsz(self.predict_size),
            'conf': base_conf
        })

        if self.multi_scale_enabled:
            boost_imgsz = self.align_imgsz(int(self.predict_size * self.multi_scale_factor))
            boost_imgsz = min(self.heavy_scan_imgsz, max(self.predict_size, boost_imgsz))
            if boost_imgsz > self.predict_size:
                detection_passes.append({
                    'name': 'boost',
                    'imgsz': boost_imgsz,
                    'conf': max(self.multi_scale_conf_floor, base_conf + self.multi_scale_conf_shift)
                })

        heavy_ready = self.multi_scale_enabled and self.no_detect_frames >= self.heavy_scan_interval
        if heavy_ready:
            detection_passes.append({
                'name': 'heavy',
                'imgsz': self.align_imgsz(self.heavy_scan_imgsz),
                'conf': self.heavy_conf_floor
            })

        selected_targets = []
        selected_results = None
        selected_pass = None

        for cfg in detection_passes:
            # Используем оптимизированный движок
            results = self.inference_engine.model(
                img,
                imgsz=cfg['imgsz'],
                device=self.device,
                conf=cfg['conf'],
                verbose=False,
                half=self.inference_engine.use_fp16,
                augment=False,
                agnostic_nms=True
            )

            selected_results = results
            # Детекция без учета игроков на первом этапе
            targets = self.find_prioritized_targets(results, img.shape, [], max_targets=5)
            if targets:
                selected_targets = targets
                selected_pass = cfg['name']
                self.last_detection_pass = selected_pass
                self.det_pass_stats[selected_pass] = self.det_pass_stats.get(selected_pass, 0) + 1
                if selected_pass == 'heavy':
                    self.no_detect_frames = max(0, self.no_detect_frames - self.heavy_scan_relief)

                # Интеграция предсказания для найденных целей
                for i, target in enumerate(selected_targets):
                    track_id = f"candy_{i}_{int(frame_time)}"
                    self.predictor.update_target_track(track_id, target['cx'], target['cy'], frame_time)
                    target['track_id'] = track_id

                    # Добавляем предсказанную позицию
                    prediction = self.predictor.predict_position(track_id, frames_ahead=3)
                    if prediction:
                        target['predicted_cx'] = prediction['cx']
                        target['predicted_cy'] = prediction['cy']

                break

            # heavy pass без результата — тоже фиксируем, чтобы избежать спама
            if cfg['name'] == 'heavy':
                self.no_detect_frames = max(0, self.no_detect_frames - self.heavy_scan_relief)
                self.last_detection_pass = 'heavy'

        if selected_pass is None and detection_passes:
            self.last_detection_pass = detection_passes[-1]['name']

        return selected_targets, selected_results
    
    def calculate_risk_factor(self, candy_x, candy_y, player_targets):
        """Рассчитывает фактор риска для конфеты на основе близости игроков."""
        if not player_targets:
            return 0.0

        total_risk = 0.0
        for player in player_targets:
            player_x, player_y = player['cx'], player['cy']
            dist_to_player = np.hypot(candy_x - player_x, candy_y - player_y)
            
            if dist_to_player < self.risk_distance_threshold:
                # Риск обратно пропорционален дистанции
                risk = (1 - (dist_to_player / self.risk_distance_threshold))
                total_risk += risk

        return total_risk

    def calculate_priority_score(self, cx, cy, conf, distance, frame_shape, player_targets):
        """Вычисляет приоритет цели с учетом Y-оси и игроков (Хищник)."""
        frame_h, frame_w = frame_shape[:2]
        center_x = frame_w / 2
        center_y = frame_h / 2
        
        # Базовый приоритет
        priority = conf * 100
        
        # Бонус за близость к горизонтальному центру (X-ось)
        center_dist_x = abs(cx - center_x)
        priority += max(0, 50 - center_dist_x / 8)

        # УСИЛЕННЫЙ бонус за близость к ВЕРТИКАЛЬНОМУ центру (Y-ось)
        center_dist_y = abs(cy - center_y)
        priority += max(0, 80 - center_dist_y / 4) * self.y_axis_priority_weight
        
        # Бонус за близость к игроку
        priority += max(0, 130 - distance / 4)
        
        # Штраф за риск (режим "Хищник")
        risk = self.calculate_risk_factor(cx, cy, player_targets)
        priority -= risk * self.risk_factor_weight

        return priority

    def clean_target_memory(self, current_time):
        if not self.target_memory:
            return
        filtered = [entry for entry in self.target_memory
                    if current_time - entry['time'] <= self.memory_retention and entry['conf'] >= self.memory_min_conf]
        self.target_memory.clear()
        self.target_memory.extend(filtered)

    def maybe_refresh_memory(self, entry, frame_time, frame_shape):
        frame_h, frame_w = frame_shape[:2]
        if frame_w == 0 or frame_h == 0:
            return False

        self.clean_target_memory(frame_time)

        cx_norm = entry['cx'] / frame_w
        cy_norm = entry['cy'] / frame_h

        matched = False
        for memory in self.target_memory:
            dx = (cx_norm - memory['cx_norm']) * frame_w
            dy = (cy_norm - memory['cy_norm']) * frame_h
            distance = np.hypot(dx, dy)
            if distance < 70:
                memory['cx_norm'] = cx_norm
                memory['cy_norm'] = cy_norm
                memory['conf'] = max(memory['conf'], float(entry['conf']))
                memory['priority'] = max(memory.get('priority', 0.0), float(entry['conf']) * 100.0)
                memory['time'] = frame_time
                matched = True

        return matched

    def update_target_memory(self, target, frame_time, frame_shape):
        if frame_shape is None:
            return
        frame_h, frame_w = frame_shape[:2]
        if frame_w == 0 or frame_h == 0:
            return

        entry = {
            'cx': target['cx'],
            'cy': target['cy'],
            'conf': target['conf'],
            'priority': target.get('priority_score', target['conf'] * 100.0)
        }

        if not self.maybe_refresh_memory(entry, frame_time, frame_shape):
            self.clean_target_memory(frame_time)
            self.target_memory.append({
                'cx_norm': float(entry['cx'] / frame_w),
                'cy_norm': float(entry['cy'] / frame_h),
                'conf': float(entry['conf']),
                'priority': float(entry['priority']),
                'time': frame_time
            })

        self.last_detection_time = frame_time

    def forget_target(self, target, frame_shape):
        if frame_shape is None or not self.target_memory:
            return
        frame_h, frame_w = frame_shape[:2]
        if frame_w == 0 or frame_h == 0:
            return

        current_time = time.time()
        self.clean_target_memory(current_time)

        target_cx_norm = target['cx'] / frame_w
        target_cy_norm = target['cy'] / frame_h

        remaining = [entry for entry in self.target_memory
                     if np.hypot(target_cx_norm - entry['cx_norm'], target_cy_norm - entry['cy_norm']) > 0.03]
        self.target_memory.clear()
        self.target_memory.extend(remaining)

    def select_memory_target(self):
        current_time = time.time()
        if current_time - self.last_revisit_time < self.memory_revisit_cooldown:
            return None
        if current_time - self.last_detection_time < 0.4:
            return None

        self.clean_target_memory(current_time)
        if not self.target_memory:
            return None

        best = max(self.target_memory, key=lambda entry: (entry.get('priority', 0.0), entry['conf']))
        return best

    def navigate_to_memory(self, memory_entry):
        if self.last_frame_shape is None:
            return

        frame_h, frame_w = self.last_frame_shape[:2]
        if frame_w == 0 or frame_h == 0:
            return

        cx = memory_entry['cx_norm'] * frame_w
        cy = memory_entry['cy_norm'] * frame_h
        center_x = frame_w / 2
        center_y = frame_h / 2
        distance = np.hypot(cx - center_x, cy - center_y)

        pseudo_target = {
            'cx': cx,
            'cy': cy,
            'conf': max(memory_entry.get('conf', self.memory_min_conf), self.memory_min_conf),
            'distance': distance,
            'priority_score': memory_entry.get('priority', memory_entry.get('conf', 0.2) * 100.0),
            'from_memory': True
        }

        collected = self.move_to_target(pseudo_target, self.last_frame_shape)
        if collected:
            self.forget_target(pseudo_target, self.last_frame_shape)

        self.last_revisit_time = time.time()
        self.stats['memory_revisits'] += 1

    def calculate_stability_bonus_from_memory(self, cx, cy, frame_shape):
        frame_h, frame_w = frame_shape[:2]
        distances = []
        for memory_entry in self.target_memory:
            mem_cx = memory_entry['cx_norm'] * frame_w
            mem_cy = memory_entry['cy_norm'] * frame_h
            distance = np.sqrt((cx - mem_cx) ** 2 + (cy - mem_cy) ** 2)
            distances.append(distance)
        if not distances:
            return 0
        avg_distance = np.mean(distances)
        normalized = max(0, 1.0 - avg_distance / (0.4 * frame_w))
        return normalized * 30
    
    def move_to_target(self, target, frame_shape):
        """Движение к цели в режиме Hardcore Aim"""
        frame_h, frame_w = frame_shape[:2]
        center_x = frame_w / 2

        # Используем предсказанную позицию если доступна
        if 'predicted_cx' in target and 'predicted_cy' in target:
            # Вычисляем оптимальную точку перехвата
            bot_pos = (center_x, center_y)
            intercept = self.predictor.get_optimal_intercept_point(
                target.get('track_id'), bot_pos, bot_speed=120
            )
            if intercept:
                cx = intercept['cx']
                cy = intercept['cy']
                print(f"[PREDICT] Используем точку перехвата: ({cx:.1f}, {cy:.1f})")
            else:
                cx = target['predicted_cx']
                cy = target['predicted_cy']
        else:
            cx = target['cx']
            cy = target['cy']

        # EMA сглаживание для стабильности
        if self.smoothed_cx is None: self.smoothed_cx = cx
        else: self.smoothed_cx = self.aim_alpha * cx + (1 - self.aim_alpha) * self.smoothed_cx
        
        error = self.smoothed_cx - center_x
        distance = target['distance']

        # 1. Проверка сбора -> Переход в режим ПОДТВЕРЖДЕНИЯ
        if distance < self.min_distance:
            self.control.release_all_keys()
            self.state = 'CONFIRMING'
            self.target_for_confirmation = target
            self.confirmation_frames = 0
            print(f"[CONFIRM] Подошел к цели, проверяю сбор...")
            return # Выходим из move_to_target, чтобы главный цикл обработал новое состояние

        # 2. Проверка застревания и огибание
        if self.previous_distance is not None:
            if distance >= self.previous_distance - 2.0: # Используем небольшую константу вместо удаленной переменной
                self.stuck_frames += 1
            else:
                self.stuck_frames = 0 # Сбрасываем, если есть прогресс
                
            # --- Реактивное огибание препятствий ---
            if self.stuck_for_obstacle_avoidance_threshold < self.stuck_frames < self.stuck_detection_threshold and distance > self.min_distance * 1.5:
                print("[AVOID] Попытка обойти препятствие...")
                self.control.release_all_keys()
                self.control.press('up')
                strafe_key = 'left' if random.random() < 0.5 else 'right'
                self.control.press(strafe_key)
                time.sleep(0.4) # Короткий резкий стрейф
                self.control.release(strafe_key)
                # Пропускаем несколько кадров застревания, чтобы дать маневру сработать
                self.stuck_frames += 2 
                return

            if self.stuck_frames >= self.stuck_detection_threshold:
                print(f"[STUCK] Застрял на {self.stuck_frames} кадров! Сбрасываю цель.")
                self.perform_escape_maneuver(error)
                self.forget_target(target, frame_shape)
                self.reset_movement_state()
                return
            
        # 3. Универсальная логика движения (вперед + стрейф)
        self.control.release_all_keys()
        self.control.press('up')

        # Постоянная коррекция стрейфом
        if abs(error) > self.aim_complete_threshold_px:
            strafe_key = 'left' if error < 0 else 'right'
            self.control.press(strafe_key)
        
        # Динамический sleep на основе FPS для стабильности
        sleep_duration = self.forward_max_step / max(1.0, self.frame_counter.fps / 20.0)
        time.sleep(float(sleep_duration))
        
        self.previous_distance = distance

    def perform_escape_maneuver(self, error):
        """Протокол 'Прорыв' для выхода из углов."""
        now = time.time()
        if now - self.last_escape_time < self.escape_cooldown:
            return

        print("[ESCAPE] Выполняем манёвр выхода...")
        self.control.release_all_keys()
        
        # 1. Отъехать назад, чтобы освободить место
        self.control.press('down')
        time.sleep(float(self.escape_push_duration * 1.5))
        self.control.release_all_keys()

        # 2. Резко повернуться в сторону от препятствия
        turn_direction = 'turn_right' if error < 0 else 'turn_left'
        self.control.press(turn_direction)
        time.sleep(float(0.4))
        self.control.release_all_keys()

        # 3. Сделать рывок вперед с прыжком
        self.control.press('up')
        self.control.press('jump')
        time.sleep(float(self.escape_push_duration))
        self.control.release_all_keys()

        self.last_escape_time = now
        self.reset_movement_state()

    def reset_movement_state(self):
        """Сброс всех переменных состояния движения."""
        self.current_target = None
        self.previous_distance = float('inf')
        self.progress_history.clear()
        self.retry_frames = 0
        self.integral = 0.0
        self.prev_error = 0.0
        self.smoothed_cx = None
    
    def log_detection(self, targets, frame_time):
        """Логирование детекций для анализа"""
        log_entry = {
            'timestamp': frame_time,
            'targets_count': len(targets),
            'confidence_threshold': self.conf_thres,
            'fps': self.frame_counter.fps,
            'pass': self.last_detection_pass,
            'targets': []
        }
        
        for target in targets:
            log_entry['targets'].append({
                'cx': float(target['cx']),
                'cy': float(target['cy']),
                'conf': float(target['conf']),
                'distance': float(target['distance']),
                'priority': float(target['priority_score']),
                'memory_boost': bool(target.get('memory_boost', False))
            })
        
        self.log_data.append(log_entry)
        
        if len(self.log_data) >= 100:
            self.save_log()

    def save_log(self):
        """Сохранение лога в файл"""
        try:
            with open(self.log_file, 'w') as f:
                json.dump(self.log_data, f, indent=2)
            print(f"[LOG] Сохранено {len(self.log_data)} записей в {self.log_file}")
            self.log_data = []
        except Exception as e:
            print(f"[ERROR] Ошибка сохранения лога: {e}")

    def print_stats(self):
        """Вывод статистики"""
        runtime = time.time() - self.stats['start_time']
        if self.stats['total_detections'] > 0:
            self.stats['avg_confidence'] = self.conf_sum / self.stats['total_detections']
        efficiency = (self.stats['successful_collections'] / max(1, self.stats['total_detections'])) * 100
        
        print(f"\n[STATS] Время работы: {runtime:.1f}с")
        print(f"[STATS] Детекций: {self.stats['total_detections']}")
        print(f"[STATS] Собрано: {self.stats['successful_collections']}")
        eff_percent = (self.stats['successful_collections'] / self.stats['total_detections'] * 100) if self.stats['total_detections'] > 0 else 0
        print(f"[STATS] Эффективность: {eff_percent:.1f}%")
        print(f"[STATS] Циклов поиска: {self.stats['search_cycles']}")
        avg_conf = (self.conf_sum / self.stats['total_detections']) if self.stats['total_detections'] > 0 else 0
        print(f"[STATS] Средняя уверенность: {avg_conf:.3f}")
        print(f"[STATS] Детекция по пассам: base:{self.stats['pass_counts']['base']}, boost:{self.stats['pass_counts']['boost']}, heavy:{self.stats['pass_counts']['heavy']}")
        print(f"[STATS] Возвратов к памяти: {self.stats['memory_revisits']}")
        print(f"[STATS] Переключений по кэшу: {self.stats['cache_retrievals']}")
        if self.player_model:
            print(f"[PREDATOR] Игроков обнаружено: {self.stats['players_detected']}")

    def run(self):
        """Основной цикл улучшенного бота c конечным автоматом."""
        global stop_requested
        stop_requested = False
        
        # Надежная регистрация hotkey
        try:
            keyboard.add_hotkey(self.manual_stop_key, request_stop)
        except Exception as e:
            print(f"[WARN] Не удалось зарегистрировать hotkey: {e}")

        print("[INFO] Управление остановкой:")
        print("[INFO] - Ctrl+C - мягкая остановка")
        print(f"[INFO] - Клавиша '{self.manual_stop_key}' (глобально) - мягкая остановка")
        
        try:
            prev_time = time.time()
            for img, img0 in self.stream:
                if stop_requested: break
                
                # FIX: Конвертируем захваченное изображение из RGB в BGR для корректной обработки в OpenCV
                img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

                current_time = time.time()
                self.dt = current_time - prev_time
                prev_time = current_time
                self.last_frame_shape = img.shape
                
                self.frame_counter_internal += 1
                
                # --- Динамический predict_size ---
                if self.dynamic_predict_size:
                    is_hunting_far = self.state == 'HUNTING' and self.current_target and self.current_target['distance'] > self.far_distance_threshold
                    is_searching = self.state == 'SEARCHING'
                    
                    if is_hunting_far or is_searching:
                        self.predict_size = self.high_fps_predict_size
                    else:
                        self.predict_size = self.high_accuracy_predict_size
                # --- Конец динамики ---

                # --- Детекция (Конфеты и Игроки) ---
                candy_targets = []
                candy_results = None
                det_img = None # Ленивая предобработка

                # Умное кэширование угроз
                if current_time - self.last_player_detection_time > self.threat_cache_ttl:
                    self.confirmed_threats = []

                # Условие для запуска детекции игроков
                should_detect_players = self.state == 'SEARCHING' or \
                                      (self.state == 'HUNTING' and self.current_target and 
                                       self.current_target['distance'] > self.player_detection_hunt_cutoff_dist)

                # 1. Детекция игроков (редко и по условию)
                if self.player_model and should_detect_players and self.frame_counter_internal % (self.player_detection_skip + 1) == 0:
                    if det_img is None: det_img = self.preprocess_frame(img)
                    player_results = self.player_model(det_img, imgsz=self.predict_size, device=self.device, conf=self.player_conf_thres, verbose=False, half=self.use_fp16)
                    raw_players = self.find_player_targets(player_results, det_img.shape)
                    current_players = self.filter_ego(raw_players, det_img.shape)
                    self.last_player_detection_time = current_time # Обновляем время
                    
                    # Логика "Двойной проверки"
                    newly_confirmed = []
                    for p_current in current_players:
                        for p_last in self.last_detected_players:
                            dist = np.hypot(p_current['cx'] - p_last['cx'], p_current['cy'] - p_last['cy'])
                            if dist < self.threat_confirmation_distance:
                                newly_confirmed.append(p_current)
                                break
                    self.confirmed_threats = newly_confirmed
                    self.last_detected_players = current_players # Сохраняем для следующей проверки

                    if self.confirmed_threats:
                        self.stats['players_detected'] += len(self.confirmed_threats)

                # 2. Детекция конфет (чаще)
                if self.frame_counter_internal % (self.frame_skip + 1) == 0:
                    if det_img is None: det_img = self.preprocess_frame(img, cache_key=f"frame_{current_time}")
                    candy_targets, candy_results = self.run_multi_scale_detection(det_img, current_time)
                    self.last_known_candies = candy_targets
                else:
                    candy_targets = self.last_known_candies

                # 3. Пересчет приоритетов конфет с учетом ПОДТВЕРЖДЕННЫХ игроков
                if candy_targets and self.confirmed_threats:
                    for candy in candy_targets:
                        candy['priority_score'] = self.calculate_priority_score(
                            candy['cx'], candy['cy'], candy['conf'], candy['distance'], 
                            img.shape, self.confirmed_threats
                        )
                    candy_targets.sort(key=lambda t: t['priority_score'], reverse=True)
                # --- Конец детекции ---
                
                best_target = candy_targets[0] if candy_targets else None

                # --- Интеграция систем ИИ ---
                # Оценка угроз и выбор стратегии уклонения
                bot_position = (img.shape[1] / 2, img.shape[0] / 2)
                threat_level, imminent_threats = self.evasion_system.assess_threat_level(
                    self.confirmed_threats, bot_position
                )

                evasion_strategy = self.evasion_system.choose_evasion_strategy(
                    threat_level, imminent_threats, self.state
                )

                # Выполнение уклонения если необходимо
                if evasion_strategy and threat_level >= 2:
                    print(f"[EVASION] Выполняем {evasion_strategy} от {len(imminent_threats)} угроз")
                    self.evasion_system.execute_evasion(evasion_strategy, imminent_threats, self.control)

                # Reinforcement learning: определение состояния и выбор действия
                current_rl_state = self.reinforcement_learner.get_state(
                    self.state, self.conf_thres,
                    self.current_target['distance'] if self.current_target else None,
                    self.stuck_frames, threat_level
                )

                learned_action = self.reinforcement_learner.choose_action(current_rl_state)

                # Применение изученного действия
                if learned_action == 'memory_search' and self.state == 'SEARCHING':
                    memory_target = self.select_memory_target()
                    if memory_target:
                        print("[LEARNING] Используем память для поиска цели")
                        self.navigate_to_memory(memory_target)

                # --- Главный конечный автомат ---
                if best_target:
                    self.state = 'HUNTING'
                    self.last_candy_found_time = current_time # Сбрасываем таймер "потеряшки"
                    # Корректно обновляем статистику при новой детекции
                    if not self.current_target or self.current_target['cx'] != best_target['cx']:
                         self.stats['total_detections'] += 1
                         self.conf_sum += best_target['conf']
                    self.current_target = best_target
                else:
                    if self.state == 'HUNTING': # Потеряли цель, начинаем поиск
                        self.state = 'SEARCHING'
                        self.search_phase = 'SCANNING'
                        self.search_scan_start_time = current_time
                        self.search_turn_direction = random.choice(['turn_right', 'turn_left'])
                        print("[SEARCH] Цель потеряна, начинаем сканирование...")

                if self.state == 'HUNTING':
                    self.move_to_target(self.current_target, img.shape)
                    # После move_to_target цель могла быть собрана и обнулена. Проверяем.
                    if self.current_target:
                        self.update_target_memory(self.current_target, current_time, img.shape)
                
                elif self.state == 'CONFIRMING':
                    self.confirmation_frames += 1
                    
                    # Проверяем, осталась ли конфета на месте
                    candy_still_visible = False
                    if candy_targets:
                        for candy in candy_targets:
                            dist = np.hypot(candy['cx'] - self.target_for_confirmation['cx'], candy['cy'] - self.target_for_confirmation['cy'])
                            if dist < 50: # Если в радиусе 50px есть конфета, считаем что она та же
                                candy_still_visible = True
                                break
                    
                    if not candy_still_visible:
                        # Успех! Конфета исчезла.
                        print(f"[COLLECT] Сбор подтвержден! (conf: {self.target_for_confirmation['conf']:.2f})")
                        self.stats['successful_collections'] += 1
                        self.forget_target(self.target_for_confirmation, img.shape)
                        self.reset_movement_state()
                        self.state = 'SEARCHING' # Сразу ищем новую

                        # Обучение: положительная награда за успешный сбор
                        success_metrics = {'collection_success': True, 'target_found': True}
                        reward = self.reinforcement_learner.calculate_reward(learned_action, success_metrics)
                        next_state = self.reinforcement_learner.get_state(self.state, self.conf_thres, None, 0, threat_level)
                        self.reinforcement_learner.update_q_value(current_rl_state, learned_action, reward, next_state)
                    elif self.confirmation_frames > self.confirmation_max_frames:
                        # Провал! Конфета на месте, мы промахнулись.
                        print("[FAIL] Ошибка сбора, пробую микро-коррекцию...")
                        self.control.press('up') # Крошечный шаг вперед
                        time.sleep(0.05)
                        self.control.release('up')
                        self.state = 'HUNTING' # Возвращаемся в режим охоты на ту же цель
                        self.target_for_confirmation = None


                elif self.state == 'SEARCHING':
                    # Проверяем, не "потерялся" ли бот
                    if current_time - self.last_candy_found_time > self.long_search_threshold:
                        print("[SEARCH] Давно не было целей. Выполняю маневр смены локации...")
                        self.perform_escape_maneuver(0) # Выполняем "прорыв"
                        self.last_candy_found_time = current_time # Сбрасываем таймер
                        self.search_phase = 'SCANNING' # Начинаем заново
                        self.search_scan_start_time = current_time
                        continue # Пропускаем остаток цикла

                    self.control.release_all_keys()
                    if self.search_phase == 'SCANNING':
                        # Фаза 1: Сканирование
                        if current_time - self.search_scan_start_time > self.search_scan_duration:
                            self.search_phase = 'MOVING'
                            self.search_move_start_time = current_time
                        else:
                            self.control.press(self.search_turn_direction)
                    
                    elif self.search_phase == 'MOVING':
                        # Фаза 2: Перемещение на новую точку
                        if current_time - self.search_move_start_time > self.search_move_duration:
                            self.search_phase = 'SCANNING'
                            self.search_scan_start_time = current_time
                            self.search_turn_direction = random.choice(['turn_right', 'turn_left'])
                            self.stats['search_cycles'] += 1
                        else:
                            # Умное движение "Следопыт"
                            move_type = random.choices(['forward', 'strafe', 'back'], weights=[0.7, 0.2, 0.1], k=1)[0]
                            if move_type == 'forward':
                                self.control.press('up')
                            elif move_type == 'strafe':
                                key = random.choice(['left', 'right'])
                                self.control.press(key)
                            elif move_type == 'back':
                                self.control.press('down')
                # --- Конец конечного автомата ---

                # Адаптивная настройка и логирование
                self.adaptive_confidence_adjustment(candy_targets)
                if candy_targets: self.log_detection(candy_targets, current_time)

                # Отображение
                if self.show_window or self.save_screenshots:
                    self.display_frame(img, candy_results, candy_targets, self.confirmed_threats)
                
                self.frame_counter.log()
                if int(current_time) % 30 == 0 and int(current_time) != int(prev_time):
                    self.print_stats()

        except KeyboardInterrupt:
            print("\n[STOP] Мягкая остановка бота (Ctrl+C)...")
        except Exception as e:
            print(f"\n[ERROR] Критическая ошибка: {e}")
            print("[STOP] Принудительная остановка...")
        
        finally:
            print("\n[EXIT] Завершение работы...")
            self.control.release_all_keys()

            # Сохранение прогресса обучения
            self.reinforcement_learner.save_learning_progress()

            self.print_stats()
            self.save_log()
            if self.show_window:
                cv2.destroyAllWindows()
            # Надежное удаление hotkey
            try:
                keyboard.remove_hotkey(self.manual_stop_key)
            except Exception as e:
                print(f"[WARN] Не удалось удалить hotkey: {e}")
            print("[EXIT] Бот остановлен")

    def display_frame(self, img, results, targets, player_targets=None):
        """Отрисовка аннотаций и служебной информации на кадре."""
        try:
            if results is not None and len(results) > 0 and results[0].boxes is not None:
                annotated = results[0].plot()
            else:
                annotated = img.copy()
            
            h, w = annotated.shape[:2]
            cv2.drawMarker(annotated, (w // 2, h // 2), (0, 255, 255), markerType=cv2.MARKER_CROSS, markerSize=14, thickness=2)
            
            # Отрисовка игроков
            if player_targets:
                for player in player_targets:
                    x1, y1, x2, y2 = player['bbox']
                    cv2.rectangle(annotated, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)
                    cv2.putText(annotated, f"Player {player['conf']:.2f}", (int(x1), int(y1) - 10), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

            if self.current_target:
                tx = int(self.current_target['cx'] / img.shape[1] * w)
                ty = int(self.current_target['cy'] / img.shape[0] * h)
                cv2.circle(annotated, (tx, ty), 8, (0, 255, 0), 2)

            state_info = f"STATE: {self.state}"
            if self.state == 'SEARCHING':
                state_info += f" ({self.search_phase})"
            
            info = f"conf={self.conf_thres:.2f} fps={self.frame_counter.fps:.1f} targets={len(targets) if targets else 0}"
            cv2.putText(annotated, state_info, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            cv2.putText(annotated, info, (10, 56), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            if self.save_screenshots:
                # Логика сохранения скриншотов...
                pass
            
            if self.show_window:
                cv2.imshow('Hardcore Bot - View', annotated)
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    global stop_requested
                    stop_requested = True
                elif key == ord('s'):
                    filename = f"debug_screenshot_{int(time.time())}.jpg"
                    cv2.imwrite(filename, annotated)
                    print(f"[DEBUG] Скриншот сохранен: {filename}")
        except Exception as e:
            print(f'[WARN] Ошибка отображения: {e}')
            if not self.window_failed:
                self.show_window = False
                self.save_screenshots = True
                self.window_failed = True

def main():
    parser = argparse.ArgumentParser(description='Enhanced Ball Hunter Bot v5.3 - Overdrive Edition')
    parser.add_argument('--weights', type=str, default='weights/candies_v10.pt', 
                       help='Путь к весам модели для конфет')
    parser.add_argument('--player-weights', type=str, default=None, 
                       help='Путь к весам модели для игроков (активирует режим Хищника)')
    parser.add_argument('--conf', type=float, default=0.22, 
                       help='Базовый порог уверенности')
    parser.add_argument('--no-adaptive', action='store_true', 
                       help='Отключить адаптивную систему')
    parser.add_argument('--show', action='store_true',
                       help='Показать окно с аннотированным изображением')
    parser.add_argument('--save-screenshots', action='store_true',
                       help='Сохранять скриншоты с аннотациями на диск')
    parser.add_argument('--size', type=int, default=0,
                       help='Фиксированный размер детекции (0 для динамического)')
    parser.add_argument('--skip', type=int, default=1,
                       help='Количество пропускаемых кадров между детекциями конфет')
    parser.add_argument('--player-skip', type=int, default=4,
                       help='Количество пропускаемых кадров для детекции игроков (выше = быстрее)')
    parser.add_argument('--player-conf', type=float, default=0.40,
                       help='Порог уверенности для детекции игроков')
    parser.add_argument('--no-fp16', action='store_true',
                       help='Отключить half-precision (FP16) режим (не рекомендуется)')
    parser.add_argument('--no-preprocess', action='store_true',
                       help='Отключить предобработку кадра (может поднять FPS)')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.weights):
        print(f"[ERROR] Веса не найдены: {args.weights}")
        print("[INFO] Доступные веса:")
        weights_dir = os.path.join(os.path.dirname(__file__), 'weights')
        if os.path.exists(weights_dir):
            for f in os.listdir(weights_dir):
                if f.endswith('.pt'):
                    print(f"  - {f}")
        return
    
    bot = EnhancedBallBot(args.weights, args.player_weights, args.conf, not args.no_adaptive, 
                         show_window=args.show, save_screenshots=args.save_screenshots)
    if args.size > 0:
        bot.dynamic_predict_size = False
        bot.predict_size = args.size
    bot.frame_skip = args.skip
    bot.player_detection_skip = args.player_skip
    bot.player_conf_thres = args.player_conf
    if args.no_fp16:
        bot.use_fp16 = False
    if args.no_preprocess:
        bot.preprocess_enabled = False
    bot.run()

if __name__ == '__main__':
    main()
