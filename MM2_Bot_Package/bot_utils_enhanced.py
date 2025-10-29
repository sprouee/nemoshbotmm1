"""
Утилиты для улучшенного бота поиска мячиков
Содержит вспомогательные функции и классы
"""

import numpy as np
import cv2
import time
from collections import deque
import json

class TargetTracker:
    """Трекер целей для отслеживания движения объектов"""
    
    def __init__(self, max_history=10):
        self.target_history = deque(maxlen=max_history)
        self.current_targets = {}
        self.target_id_counter = 0
    
    def update_targets(self, detections, frame_time):
        """Обновляет трекинг целей"""
        # Простая система трекинга на основе расстояния
        for detection in detections:
            cx, cy = detection['cx'], detection['cy']
            
            # Ищем ближайший существующий трек
            closest_track_id = None
            min_distance = float('inf')
            
            for track_id, track_data in self.current_targets.items():
                last_pos = track_data['positions'][-1]
                distance = np.sqrt((cx - last_pos['cx'])**2 + (cy - last_pos['cy'])**2)
                
                if distance < min_distance and distance < 50:  # Порог для ассоциации
                    min_distance = distance
                    closest_track_id = track_id
            
            if closest_track_id is not None:
                # Обновляем существующий трек
                self.current_targets[closest_track_id]['positions'].append({
                    'cx': cx, 'cy': cy, 'conf': detection['conf'], 'time': frame_time
                })
                self.current_targets[closest_track_id]['last_seen'] = frame_time
            else:
                # Создаем новый трек
                track_id = self.target_id_counter
                self.target_id_counter += 1
                self.current_targets[track_id] = {
                    'positions': [{'cx': cx, 'cy': cy, 'conf': detection['conf'], 'time': frame_time}],
                    'created': frame_time,
                    'last_seen': frame_time
                }
        
        # Удаляем старые треки
        current_time = time.time()
        tracks_to_remove = []
        for track_id, track_data in self.current_targets.items():
            if current_time - track_data['last_seen'] > 2.0:  # 2 секунды без обновления
                tracks_to_remove.append(track_id)
        
        for track_id in tracks_to_remove:
            del self.current_targets[track_id]
    
    def get_stable_targets(self, min_observations=3):
        """Возвращает стабильные цели (наблюдаемые несколько раз)"""
        stable_targets = []
        for track_id, track_data in self.current_targets.items():
            if len(track_data['positions']) >= min_observations:
                # Вычисляем среднюю позицию и уверенность
                positions = track_data['positions']
                avg_cx = np.mean([p['cx'] for p in positions])
                avg_cy = np.mean([p['cy'] for p in positions])
                avg_conf = np.mean([p['conf'] for p in positions])
                
                stable_targets.append({
                    'track_id': track_id,
                    'cx': avg_cx,
                    'cy': avg_cy,
                    'conf': avg_conf,
                    'observations': len(positions),
                    'stability': len(positions) / 10.0  # Нормализованная стабильность
                })
        
        return stable_targets

class PerformanceMonitor:
    """Монитор производительности бота"""
    
    def __init__(self):
        self.fps_history = deque(maxlen=100)
        self.detection_times = deque(maxlen=50)
        self.movement_times = deque(maxlen=50)
        self.start_time = time.time()
    
    def log_fps(self, fps):
        """Логирует FPS"""
        self.fps_history.append(fps)
    
    def log_detection_time(self, detection_time):
        """Логирует время детекции"""
        self.detection_times.append(detection_time)
    
    def log_movement_time(self, movement_time):
        """Логирует время обработки движения"""
        self.movement_times.append(movement_time)
    
    def get_stats(self):
        """Возвращает статистику производительности"""
        stats = {
            'avg_fps': np.mean(self.fps_history) if self.fps_history else 0,
            'min_fps': np.min(self.fps_history) if self.fps_history else 0,
            'max_fps': np.max(self.fps_history) if self.fps_history else 0,
            'avg_detection_time': np.mean(self.detection_times) if self.detection_times else 0,
            'avg_movement_time': np.mean(self.movement_times) if self.movement_times else 0,
            'runtime': time.time() - self.start_time
        }
        return stats

class SmartSearchEngine:
    """Умный движок поиска с адаптивными паттернами"""
    
    def __init__(self):
        self.search_patterns = [
            # Паттерн 1: Медленный поворот
            {'type': 'slow_turn', 'direction': 'right', 'duration': 0.4, 'pause': 0.2},
            {'type': 'slow_turn', 'direction': 'left', 'duration': 0.3, 'pause': 0.15},
            
            # Паттерн 2: Быстрый сканирование
            {'type': 'quick_scan', 'direction': 'right', 'duration': 0.15, 'pause': 0.05},
            {'type': 'quick_scan', 'direction': 'left', 'duration': 0.15, 'pause': 0.05},
            {'type': 'quick_scan', 'direction': 'right', 'duration': 0.15, 'pause': 0.05},
            
            # Паттерн 3: Движение + поиск
            {'type': 'move_search', 'direction': 'forward', 'duration': 0.3, 'pause': 0.1},
            {'type': 'move_search', 'direction': 'turn_right', 'duration': 0.2, 'pause': 0.1},
            {'type': 'move_search', 'direction': 'forward', 'duration': 0.3, 'pause': 0.1},
            {'type': 'move_search', 'direction': 'turn_left', 'duration': 0.2, 'pause': 0.1},
        ]
        
        self.current_pattern = 0
        self.pattern_success_rate = {}
        self.last_detection_time = 0
    
    def get_next_pattern(self):
        """Возвращает следующий паттерн поиска"""
        pattern = self.search_patterns[self.current_pattern % len(self.search_patterns)]
        self.current_pattern += 1
        return pattern
    
    def update_success_rate(self, pattern_type, success):
        """Обновляет статистику успешности паттернов"""
        if pattern_type not in self.pattern_success_rate:
            self.pattern_success_rate[pattern_type] = {'success': 0, 'total': 0}
        
        self.pattern_success_rate[pattern_type]['total'] += 1
        if success:
            self.pattern_success_rate[pattern_type]['success'] += 1
    
    def get_best_patterns(self):
        """Возвращает наиболее успешные паттерны"""
        if not self.pattern_success_rate:
            return self.search_patterns[:3]  # Возвращаем первые 3 по умолчанию
        
        # Сортируем по успешности
        sorted_patterns = sorted(
            self.pattern_success_rate.items(),
            key=lambda x: x[1]['success'] / max(1, x[1]['total']),
            reverse=True
        )
        
        # Возвращаем топ-3 паттерна
        best_types = [p[0] for p in sorted_patterns[:3]]
        return [p for p in self.search_patterns if p['type'] in best_types]

class AdaptiveController:
    """Адаптивный контроллер для настройки параметров бота"""
    
    def __init__(self):
        self.performance_history = deque(maxlen=20)
        self.detection_history = deque(maxlen=30)
        self.movement_history = deque(maxlen=30)
        
        # Параметры для адаптации
        self.base_params = {
            'conf_threshold': 0.25,
            'min_distance': 50,
            'turn_threshold': 25,
            'search_threshold': 8,
            'jump_interval': 12
        }
        
        self.current_params = self.base_params.copy()
    
    def update_performance(self, fps, detections_count, collections_count):
        """Обновляет данные о производительности"""
        self.performance_history.append({
            'fps': fps,
            'detections': detections_count,
            'collections': collections_count,
            'time': time.time()
        })
    
    def adapt_parameters(self):
        """Адаптирует параметры на основе производительности"""
        if len(self.performance_history) < 10:
            return  # Недостаточно данных для адаптации
        
        recent_performance = list(self.performance_history)[-10:]
        avg_fps = np.mean([p['fps'] for p in recent_performance])
        avg_detections = np.mean([p['detections'] for p in recent_performance])
        avg_collections = np.mean([p['collections'] for p in recent_performance])
        
        # Адаптация порога уверенности
        if avg_fps < 5.0:  # Низкий FPS - повышаем порог
            self.current_params['conf_threshold'] = min(0.4, 
                self.current_params['conf_threshold'] + 0.02)
        elif avg_fps > 8.0 and avg_detections < 2:  # Высокий FPS, мало детекций - снижаем порог
            self.current_params['conf_threshold'] = max(0.15, 
                self.current_params['conf_threshold'] - 0.01)
        
        # Адаптация расстояния сбора
        if avg_collections / max(1, avg_detections) < 0.3:  # Низкая эффективность сбора
            self.current_params['min_distance'] = max(30, 
                self.current_params['min_distance'] - 5)
        elif avg_collections / max(1, avg_detections) > 0.8:  # Высокая эффективность
            self.current_params['min_distance'] = min(80, 
                self.current_params['min_distance'] + 3)
        
        # Адаптация порога поиска
        if avg_detections < 1:  # Мало детекций - начинаем поиск раньше
            self.current_params['search_threshold'] = max(5, 
                self.current_params['search_threshold'] - 1)
        elif avg_detections > 5:  # Много детекций - можем ждать дольше
            self.current_params['search_threshold'] = min(15, 
                self.current_params['search_threshold'] + 1)
    
    def get_current_params(self):
        """Возвращает текущие параметры"""
        return self.current_params.copy()

def calculate_target_priority(cx, cy, conf, distance, frame_shape, target_history=None):
    """Расширенный расчет приоритета цели"""
    frame_h, frame_w = frame_shape[:2]
    center_x = frame_w / 2
    center_y = frame_h / 2
    
    # Базовый приоритет
    priority = conf * 100
    
    # Бонус за близость к центру
    center_distance = np.sqrt((cx - center_x)**2 + (cy - center_y)**2)
    center_bonus = max(0, 60 - center_distance / 8)
    priority += center_bonus
    
    # Бонус за близость к игроку
    distance_bonus = max(0, 120 - distance / 4)
    priority += distance_bonus
    
    # Штрафы за неудобные позиции
    if cy < frame_h * 0.25:  # Слишком высоко
        priority *= 0.6
    elif cy > frame_h * 0.8:  # Слишком низко
        priority *= 0.8
    
    if cx < frame_w * 0.1 or cx > frame_w * 0.9:  # По краям
        priority *= 0.7
    
    # Бонус за стабильность (если есть история)
    if target_history:
        stability_bonus = calculate_stability_bonus(cx, cy, target_history)
        priority += stability_bonus
    
    return priority

def calculate_stability_bonus(cx, cy, target_history):
    """Вычисляет бонус за стабильность цели"""
    if len(target_history) < 2:
        return 0
    
    recent_positions = list(target_history)[-5:]  # Последние 5 позиций
    if len(recent_positions) < 2:
        return 0
    
    # Вычисляем дисперсию позиций
    positions_x = [pos['cx'] for pos in recent_positions]
    positions_y = [pos['cy'] for pos in recent_positions]
    
    variance_x = np.var(positions_x)
    variance_y = np.var(positions_y)
    
    # Чем меньше дисперсия, тем больше бонус
    stability = 1.0 / (1.0 + variance_x + variance_y)
    return stability * 30  # Максимальный бонус 30

def optimize_detection_params(model, test_image, device):
    """Оптимизирует параметры детекции для лучшей производительности"""
    test_sizes = [320, 416, 512, 640]
    best_size = 416
    best_fps = 0
    
    for size in test_sizes:
        start_time = time.time()
        for _ in range(10):  # Тестируем 10 раз
            results = model(test_image, imgsz=size, device=device, verbose=False)
        end_time = time.time()
        
        fps = 10 / (end_time - start_time)
        if fps > best_fps:
            best_fps = fps
            best_size = size
    
    print(f"[OPTIMIZE] Лучший размер: {best_size}, FPS: {best_fps:.1f}")
    return best_size

def save_performance_report(stats, filename="performance_report.json"):
    """Сохраняет отчет о производительности"""
    report = {
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'stats': stats,
        'summary': {
            'total_runtime': stats.get('runtime', 0),
            'avg_fps': stats.get('avg_fps', 0),
            'detection_efficiency': stats.get('avg_detection_time', 0),
            'movement_efficiency': stats.get('avg_movement_time', 0)
        }
    }
    
    try:
        with open(filename, 'w') as f:
            json.dump(report, f, indent=2)
        print(f"[REPORT] Отчет сохранен в {filename}")
    except Exception as e:
        print(f"[ERROR] Ошибка сохранения отчета: {e}")


class TrajectoryPredictor:
    """Предсказатель траектории движения целей"""
    
    def __init__(self, history_size=5):
        self.history = deque(maxlen=history_size)
        self.velocity = [0.0, 0.0]
        self.acceleration = [0.0, 0.0]
    
    def update(self, cx, cy, timestamp):
        """Обновление позиции цели"""
        self.history.append({'cx': cx, 'cy': cy, 'time': timestamp})
        
        if len(self.history) >= 2:
            # Вычисление скорости
            last = self.history[-1]
            prev = self.history[-2]
            dt = last['time'] - prev['time']
            
            if dt > 0:
                self.velocity[0] = (last['cx'] - prev['cx']) / dt
                self.velocity[1] = (last['cy'] - prev['cy']) / dt
                
                if len(self.history) >= 3:
                    # Вычисление ускорения
                    prev_prev = self.history[-2]
                    prev_dt = prev['time'] - prev_prev['time']
                    
                    if prev_dt > 0:
                        prev_velocity_x = (prev['cx'] - prev_prev['cx']) / prev_dt
                        prev_velocity_y = (prev['cy'] - prev_prev['cy']) / prev_dt
                        
                        self.acceleration[0] = (self.velocity[0] - prev_velocity_x) / dt
                        self.acceleration[1] = (self.velocity[1] - prev_velocity_y) / dt
    
    def predict(self, horizon=0.1):
        """Предсказание будущей позиции"""
        if len(self.history) < 2:
            return None
        
        last = self.history[-1]
        predicted_cx = last['cx'] + self.velocity[0] * horizon + 0.5 * self.acceleration[0] * horizon**2
        predicted_cy = last['cy'] + self.velocity[1] * horizon + 0.5 * self.acceleration[1] * horizon**2
        
        return {'cx': predicted_cx, 'cy': predicted_cy}


class PathPlanner:
    """Планировщик пути для оптимальной навигации"""
    
    def __init__(self):
        self.obstacle_history = deque(maxlen=20)
        self.path_cache = {}
    
    def plan_path(self, start, goal, obstacles=None):
        """Планирование пути от начальной точки к цели"""
        # Упрощенный A* алгоритм
        if obstacles is None:
            obstacles = []
        
        # Простое планирование - движение по прямой с обходом препятствий
        path = []
        
        # Разбиваем путь на сегменты
        steps = 10
        for i in range(steps + 1):
            t = i / steps
            x = start[0] * (1 - t) + goal[0] * t
            y = start[1] * (1 - t) + goal[1] * t
            
            # Проверка на препятствия
            collision = False
            for obs in obstacles:
                dist = np.sqrt((x - obs[0])**2 + (y - obs[1])**2)
                if dist < obs[2]:  # Радиус препятствия
                    collision = True
                    break
            
            if not collision:
                path.append((x, y))
        
        return path if path else [goal]


class AdvancedPIDController:
    """Улучшенный PID контроллер с адаптацией"""
    
    def __init__(self, kp=0.04, ki=0.001, kd=0.025):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        
        self.integral = 0.0
        self.prev_error = 0.0
        self.prev_output = 0.0
        
        self.integral_limit = 100.0
        self.alpha = 0.3  # EMA для сглаживания
        
        # Адаптивные коэффициенты
        self.adaptive = True
        self.error_history = deque(maxlen=20)
        
    def compute(self, error, dt):
        """Вычисление PID выхода"""
        # Пропорциональная часть
        p_term = self.kp * error
        
        # Интегральная часть с антивиндэпом
        self.integral += error * dt
        self.integral = np.clip(self.integral, -self.integral_limit, self.integral_limit)
        i_term = self.ki * self.integral
        
        # Дифференциальная часть с фильтрацией
        if dt > 0:
            derivative = (error - self.prev_error) / dt
        else:
            derivative = 0.0
        
        # EMA фильтрация производной
        self.prev_output = self.alpha * derivative + (1 - self.alpha) * self.prev_output
        d_term = self.kd * self.prev_output
        
        # Адаптация (если включена)
        if self.adaptive:
            self.error_history.append(abs(error))
            if len(self.error_history) >= 10:
                avg_error = np.mean(self.error_history)
                if avg_error > 50:  # Большая ошибка - увеличиваем Kp
                    self.kp = min(0.08, self.kp * 1.05)
                elif avg_error < 10:  # Малая ошибка - уменьшаем Kp
                    self.kp = max(0.02, self.kp * 0.95)
        
        output = p_term + i_term + d_term
        
        self.prev_error = error
        return output
    
    def reset(self):
        """Сброс контроллера"""
        self.integral = 0.0
        self.prev_error = 0.0
        self.prev_output = 0.0
        self.error_history.clear()


class ImageEnhancer:
    """Улучшение изображения для лучшей детекции"""
    
    def __init__(self):
        self.enhancement_enabled = True
        
    def enhance(self, img):
        """Улучшение изображения"""
        if not self.enhancement_enabled:
            return img
        
        # Конвертация в HSV
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
        
        # Увеличение насыщенности
        hsv[..., 1] = np.clip(hsv[..., 1] * 1.15, 0, 255)
        
        # Увеличение яркости
        hsv[..., 2] = np.clip(hsv[..., 2] * 1.2, 0, 255)
        
        # Конвертация обратно
        enhanced = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
        
        # Добавление контраста
        enhanced = cv2.convertScaleAbs(enhanced, alpha=1.1, beta=5)
        
        return enhanced


class SmartCache:
    """Умный кэш для оптимизации детекции"""
    
    def __init__(self, ttl=0.5):
        self.cache = {}
        self.ttl = ttl  # Time to live в секундах
        self.access_times = {}
    
    def get(self, key):
        """Получение значения из кэша"""
        if key in self.cache:
            if time.time() - self.access_times[key] < self.ttl:
                return self.cache[key]
            else:
                # Истек срок действия
                del self.cache[key]
                del self.access_times[key]
        return None
    
    def set(self, key, value):
        """Сохранение значения в кэш"""
        self.cache[key] = value
        self.access_times[key] = time.time()
    
    def clear(self):
        """Очистка кэша"""
        self.cache.clear()
        self.access_times.clear()
    
    def cleanup(self):
        """Очистка устаревших записей"""
        current_time = time.time()
        keys_to_remove = [
            key for key, access_time in self.access_times.items()
            if current_time - access_time >= self.ttl
        ]
        
        for key in keys_to_remove:
            if key in self.cache:
                del self.cache[key]
            if key in self.access_times:
                del self.access_times[key]


class PerformanceProfiler:
    """Профилировщик производительности"""
    
    def __init__(self):
        self.timings = {}
        self.call_counts = {}
        
    def start_timer(self, name):
        """Начать таймер"""
        if name not in self.timings:
            self.timings[name] = []
            self.call_counts[name] = 0
        
        self.timings[name].append({'start': time.time()})
        self.call_counts[name] += 1
    
    def end_timer(self, name):
        """Завершить таймер"""
        if name in self.timings and self.timings[name]:
            last = self.timings[name][-1]
            if 'start' in last:
                last['duration'] = time.time() - last['start']
                del last['start']
    
    def get_stats(self):
        """Получить статистику"""
        stats = {}
        for name, timings in self.timings.items():
            durations = [t['duration'] for t in timings if 'duration' in t]
            if durations:
                stats[name] = {
                    'calls': self.call_counts[name],
                    'avg_time': np.mean(durations),
                    'min_time': np.min(durations),
                    'max_time': np.max(durations),
                    'total_time': np.sum(durations)
                }
        return stats
    
    def print_stats(self):
        """Вывести статистику"""
        stats = self.get_stats()
        print("\n" + "=" * 60)
        print("📊 ПРОФИЛЬ ПРОИЗВОДИТЕЛЬНОСТИ")
        print("=" * 60)
        for name, stat in stats.items():
            print(f"{name}:")
            print(f"  Вызовов: {stat['calls']}")
            print(f"  Среднее время: {stat['avg_time']*1000:.2f} мс")
            print(f"  Мин/Макс: {stat['min_time']*1000:.2f} / {stat['max_time']*1000:.2f} мс")
            print(f"  Общее время: {stat['total_time']:.2f} с")
        print("=" * 60)
