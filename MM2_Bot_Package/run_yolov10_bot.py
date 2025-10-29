"""
YOLOv10 Bot для сбора мячиков в Murder Mystery 2

Использует ultralytics YOLOv10 для детекции и сбора объектов.
Лицензия: MIT (САМАЯ СВОБОДНАЯ ЛИЦЕНЗИЯ!)

Установка:
    pip install ultralytics

Запуск:
    python run_yolov10_bot.py --weights weights/ball_v10.pt
"""

import sys
import os
import argparse
import time
import cv2
import numpy as np

# Добавляем путь к модулям проекта
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from ultralytics import YOLO
from roblox.screen import CaptureStream
from roblox.control import Control
from roblox.utils import FrameCounter

class YOLOv10Bot:
    def __init__(self, weights_path, conf_thres=0.25, show_window=True):
        """
        Инициализация бота
        
        Args:
            weights_path: Путь к весам YOLOv10
            conf_thres: Порог уверенности для детекции
            show_window: Показывать окно с детекциями
        """
        print(f"[INIT] Zagruzka modeli YOLOv10: {weights_path}")
        self.model = YOLO(weights_path)
        self.conf_thres = conf_thres
        
        # Проверяем поддержку GUI в OpenCV
        self.gui_supported = self._check_gui_support()
        self.show_window = show_window and self.gui_supported
        
        if show_window and not self.gui_supported:
            print("[WARNING] OpenCV GUI not supported - running without display window")
            print("[INFO] Install opencv-python with GUI support: pip install opencv-python")
        
        # Инициализация захвата экрана
        print("[INIT] Zapusk zakhvata ekrana Roblox...")
        # saveInterval=0 - не сохранять изображения (быстрее работает и не требует прав на запись)
        self.stream = CaptureStream("Roblox", saveInterval=0)
        
        # Инициализация управления
        self.control = Control()
        
        # Счётчик FPS
        self.frame_counter = FrameCounter()
        self.frame_counter.fps = 0.0  # Инициализация
        
        # Параметры движения (адаптивные)
        self.min_distance = 40  # Минимальное расстояние до объекта в пикселях
        self.min_distance_close = 25  # Расстояние для сбора (очень близко)
        self.turn_threshold = 50  # Базовый порог для поворота (пиксели)
        self.turn_threshold_near = 30  # Порог поворота вблизи цели
        
        # Параметры поворота камеры
        self.frames_without_coins = 0  # Счётчик кадров без монет
        self.search_threshold = 5  # После скольких кадров начинать поиск
        self.is_searching = False  # Флаг режима поиска
        self.is_moving_to_coin = False  # Флаг движения к монете
        
        # Антизастревание (улучшенное)
        self.frames_since_jump = 0  # Кадров с последнего прыжка
        self.jump_interval = 15  # Прыгать каждые N кадров (~1 сек)
        self.stuck_frames = 0  # Кадров без прогресса
        self.stuck_threshold = 30  # Порог застревания
        self.last_distance = float('inf')  # Последнее расстояние до цели
        self.no_progress_frames = 0  # Кадров без уменьшения расстояния
        
        # Умный поиск
        self.search_direction = 1  # 1 = вправо, -1 = влево
        self.search_angle = 0  # Текущий угол поиска
        self.last_coin_position = None  # Последняя известная позиция монеты
        self.search_pattern_index = 0  # Индекс паттерна поиска
        
        # История монет для предсказания движения
        self.coin_history = []  # История позиций монет (max 5 кадров)
        self.history_max_len = 5
        
        print("[OK] Bot gotov k rabote!")
        print(f"[INFO] Confidence threshold: {self.conf_thres}")
        print(f"[INFO] Show window: {self.show_window}")
        print(f"[INFO] Camera search enabled - bot will look around!")
        if self.show_window:
            print("\n[START] Nazhmi 'q' v okne dlya ostanovki\n")
        else:
            print("\n[START] Nazhmi Ctrl+C dlya ostanovki\n")
    
    def _check_gui_support(self):
        """
        Проверяет поддержку GUI в OpenCV
        """
        try:
            # Создаем тестовое окно
            test_img = np.zeros((100, 100, 3), dtype=np.uint8)
            cv2.imshow('test', test_img)
            cv2.destroyAllWindows()
            return True
        except cv2.error:
            return False
    
    def find_closest_coin(self, boxes, frame_shape):
        """
        Находит лучшую монету/мячик с умной приоритизацией
        
        Args:
            boxes: Список детекций
            frame_shape: Размеры кадра (height, width)
        
        Returns:
            Информация о лучшей монете или None
        """
        frame_h, frame_w = frame_shape[:2]
        center_x = frame_w / 2
        center_y = frame_h / 2
        
        candidates = []
        
        for box in boxes:
            cls = int(box.cls[0])
            conf = float(box.conf[0])
            
            # Только монеты/мячики (класс 0) с достаточной уверенностью
            if cls == 0 and conf >= self.conf_thres:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                
                # Центр объекта
                cx = (x1 + x2) / 2
                cy = (y1 + y2) / 2
                
                # Размер объекта (площадь)
                area = (x2 - x1) * (y2 - y1)
                size_ratio = area / (frame_w * frame_h)
                
                # Нормализованные координаты
                norm_x = cx / frame_w
                norm_y = cy / frame_h
                
                # Фильтр: игнорируем объекты в верхней части экрана (далеко)
                if norm_y < 0.35:
                    continue
                
                # Мягкий фильтр краёв - объекты по краям менее приоритетны
                edge_penalty = 1.0
                if norm_x < 0.15 or norm_x > 0.85:
                    edge_penalty = 0.7
                elif norm_x < 0.25 or norm_x > 0.75:
                    edge_penalty = 0.85
                
                # Расстояние до центра экрана
                distance = np.sqrt((cx - center_x)**2 + (cy - center_y)**2)
                
                # Угол отклонения (важно для приоритизации поворотов)
                angle_offset = abs(cx - center_x) / frame_w
                
                # СКОР - комбинированный приоритет:
                # 1. Чем ближе - тем лучше
                # 2. Чем выше уверенность - тем лучше
                # 3. Чем больше размер (ближе к камере) - тем лучше
                # 4. Чем меньше угол отклонения - тем лучше
                # 5. Штраф за края экрана
                
                distance_score = 1.0 / (1.0 + distance / 100.0)  # Нормализованное расстояние
                confidence_score = conf
                size_score = min(size_ratio * 100, 1.0)  # Нормализованный размер
                angle_score = 1.0 - min(angle_offset * 2, 0.5)  # Штраф за угол
                
                priority_score = (
                    distance_score * 0.4 +  # Вес: близость
                    confidence_score * 0.3 +  # Вес: уверенность
                    size_score * 0.2 +  # Вес: размер
                    angle_score * 0.1  # Вес: угол
                ) * edge_penalty  # Штраф за края
                
                candidates.append({
                    'cx': cx,
                    'cy': cy,
                    'conf': conf,
                    'distance': distance,
                    'size_ratio': size_ratio,
                    'angle_offset': angle_offset,
                    'priority': priority_score,
                    'area': area
                })
        
        if not candidates:
            return None
        
        # Сортируем по приоритету
        candidates.sort(key=lambda c: c['priority'], reverse=True)
        
        # Возвращаем лучшую монету
        best_coin = candidates[0]
        
        # Обновляем историю для предсказания движения
        self.coin_history.append({
            'cx': best_coin['cx'],
            'cy': best_coin['cy'],
            'frame': len(self.coin_history)
        })
        if len(self.coin_history) > self.history_max_len:
            self.coin_history.pop(0)
        
        return best_coin
    
    def move_to_coin(self, coin, frame_shape):
        """
        Двигается к монете/мячику с адаптивными параметрами
        
        Args:
            coin: Информация о монете (cx, cy, conf, distance, ...)
            frame_shape: Размеры кадра
        """
        frame_h, frame_w = frame_shape[:2]
        center_x = frame_w / 2
        
        cx = coin['cx']
        distance = coin['distance']
        size_ratio = coin.get('size_ratio', 0)
        
        # Если был в режиме поиска - останавливаем поиск
        if self.is_searching:
            self.control.release_all_keys()
            self.is_searching = False
            self.stuck_frames = 0
            self.no_progress_frames = 0
            print("[STOP SEARCH] Moneta naydena!")
        
        # Адаптивный порог поворота в зависимости от расстояния
        adaptive_threshold = self.turn_threshold
        if distance < 150:  # Близко к цели - более точные повороты
            adaptive_threshold = self.turn_threshold_near
        elif distance > 300:  # Далеко - более широкий диапазон
            adaptive_threshold = self.turn_threshold * 1.5
        
        # Проверка на застревание: если расстояние не уменьшается
        is_making_progress = False
        if distance < self.last_distance - 5:  # Прогресс (уменьшение расстояния на 5px+)
            is_making_progress = True
            self.no_progress_frames = 0
        else:
            self.no_progress_frames += 1
        
        self.last_distance = distance
        
        # Если давно нет прогресса - застряли!
        if self.no_progress_frames > self.stuck_threshold:
            self.stuck_frames += 1
            print(f"[STUCK] Zastryanie! Popytka vykhoda ({self.stuck_frames})...")
            
            # Стратегия выхода из застревания
            self.control.release_all_keys()
            time.sleep(0.1)
            
            # Прыжок назад + поворот
            self.control.press('down')
            time.sleep(0.15)
            self.control.press('jump')
            time.sleep(0.2)
            self.control.release_all_keys()
            
            # Случайный поворот для выхода
            if self.stuck_frames % 2 == 0:
                self.control.press('left')
                time.sleep(0.2)
            else:
                self.control.press('right')
                time.sleep(0.2)
            
            self.control.release_all_keys()
            self.no_progress_frames = 0
            self.last_distance = float('inf')  # Сброс для следующей попытки
            return
        
        # Определяем, достаточно ли близко для сбора
        collect_distance = self.min_distance_close if size_ratio > 0.01 else self.min_distance
        
        if distance < collect_distance:
            # Очень близко - пытаемся собрать (пауза для автоподбора)
            self.control.release_all_keys()
            time.sleep(0.3)  # Даём время для автоподбора
            print(f"[COLLECT] Sobiraem monetu! (dist: {distance:.1f}px, conf: {coin['conf']:.2f})")
            
            # После сбора - краткая пауза
            time.sleep(0.1)
            self.is_moving_to_coin = False
            self.last_distance = float('inf')
            return
        
        # Нормальное движение к цели
        current_actions = self.control.current_actions()
        error = cx - center_x
        
        # Плавный поворот с адаптивным порогом
        if abs(error) > adaptive_threshold:
            # Нужен поворот
            if error < 0:
                # Поворачиваем влево
                if 'left' not in current_actions or 'right' in current_actions:
                    self.control.release_all_keys()
                    self.control.press('left')
                    self.control.press('up')
            else:
                # Поворачиваем вправо
                if 'right' not in current_actions or 'left' in current_actions:
                    self.control.release_all_keys()
                    self.control.press('right')
                    self.control.press('up')
        else:
            # Монета по центру - просто идём вперёд
            if 'up' not in current_actions or len(current_actions) > 1:
                self.control.release_all_keys()
                self.control.press('up')
        
        self.is_moving_to_coin = True
        
        # Умные прыжки: чаще когда далеко, реже когда близко
        self.frames_since_jump += 1
        jump_interval_adaptive = self.jump_interval
        if distance > 200:  # Далеко - чаще прыгаем
            jump_interval_adaptive = int(self.jump_interval * 0.75)
        elif distance < 100:  # Близко - реже прыгаем (чтобы не перепрыгнуть)
            jump_interval_adaptive = int(self.jump_interval * 1.5)
        
        if self.frames_since_jump >= jump_interval_adaptive:
            self.control.press('jump')
            time.sleep(0.1)
            self.frames_since_jump = 0
            
            # Возвращаем движение после прыжка
            if abs(error) > adaptive_threshold:
                if error < 0:
                    self.control.press('left')
                else:
                    self.control.press('right')
                self.control.press('up')
            else:
                self.control.press('up')
    
    def turn_camera_to_coin(self, coin, frame_shape):
        """
        Поворачивает камеру к монете, если она сбоку
        
        Args:
            coin: Информация о монете (cx, cy, conf, distance)
            frame_shape: Размеры кадра
        """
        frame_h, frame_w = frame_shape[:2]
        center_x = frame_w / 2
        cx = coin['cx']
        
        # Определяем, насколько далеко монета от центра
        offset = cx - center_x
        camera_turn_threshold = 80  # Порог для поворота камеры (пиксели)
        
        # Поворачиваем камеру, если монета сильно сбоку
        if offset < -camera_turn_threshold:
            # Монета слева - поворачиваем камеру влево
            self.control.press('turn_left')
            time.sleep(0.1)
            self.control.release_all_keys()
            print("[CAMERA] Povorot kamery vlevo k monete")
        elif offset > camera_turn_threshold:
            # Монета справа - поворачиваем камеру вправо
            self.control.press('turn_right')
            time.sleep(0.1)
            self.control.release_all_keys()
            print("[CAMERA] Povorot kamery vpravo k monete")
    
    def start_search(self):
        """
        Умный поиск монет с паттернами и запоминанием направлений
        """
        if not self.is_searching:
            self.is_searching = True
            self.is_moving_to_coin = False
            self.stuck_frames = 0
            print(f"[SEARCH] Umnyy poisk monet (pattern: {self.search_pattern_index % 4})...")
        
        # Останавливаем всё движение
        self.control.release_all_keys()
        
        # Паттерны поиска для лучшего покрытия пространства
        search_patterns = [
            ('turn_right', 0.2),   # Паттерн 0: вправо
            ('turn_left', 0.2),    # Паттерн 1: влево (компенсация)
            ('turn_right', 0.15),   # Паттерн 2: немного вправо
            ('turn_left', 0.15),    # Паттерн 3: немного влево
        ]
        
        pattern_type, pattern_duration = search_patterns[self.search_pattern_index % len(search_patterns)]
        
        # Выполняем поворот по паттерну
        self.control.press(pattern_type)
        time.sleep(pattern_duration)
        self.control.release_all_keys()
        
        # Обновляем индекс паттерна для следующего раза
        self.search_pattern_index += 1
        self.search_angle += pattern_duration * (1 if 'right' in pattern_type else -1) * 30
        
        # Пауза для анализа кадра
        time.sleep(0.08)
        
        # Прыжки во время поиска для предотвращения застревания
        self.frames_since_jump += 1
        if self.frames_since_jump >= self.jump_interval:
            self.control.press('jump')
            time.sleep(0.1)
            self.frames_since_jump = 0
            
        # Периодическое движение вперёд во время поиска (чтобы не стоять на месте)
        if self.frames_without_coins % 10 == 5:  # Каждые 10 кадров немного вперёд
            self.control.press('up')
            time.sleep(0.2)
            self.control.release_all_keys()
    
    def run(self):
        """
        Основной цикл бота
        """
        try:
            for img, img0 in self.stream:
                # Детекция объектов (используем уменьшенное изображение)
                results = self.model(img, conf=self.conf_thres, verbose=False)
                
                # Обработка результатов
                if len(results) > 0 and len(results[0].boxes) > 0:
                    boxes = results[0].boxes
                    
                    # Находим ближайшую монету
                    closest_coin = self.find_closest_coin(boxes, img.shape)
                    
                    if closest_coin:
                        # Монета найдена! Сбрасываем счётчики
                        self.frames_without_coins = 0
                        self.last_coin_position = (closest_coin['cx'], closest_coin['cy'])
                        
                        # Поворачиваем камеру к монете (если она сбоку)
                        self.turn_camera_to_coin(closest_coin, img.shape)
                        
                        # Двигаемся к монете с улучшенной логикой
                        self.move_to_coin(closest_coin, img.shape)
                        
                        # Логирование с улучшенной информацией
                        priority_info = f", Priority: {closest_coin.get('priority', 0):.3f}" if 'priority' in closest_coin else ""
                        print(f"[COIN] Conf: {closest_coin['conf']:.2f}, "
                              f"Dist: {closest_coin['distance']:.1f}px, "
                              f"Size: {closest_coin.get('size_ratio', 0)*100:.2f}%{priority_info}")
                    else:
                        # Монета не в зоне видимости (фильтры)
                        self.frames_without_coins += 1
                else:
                    # Детекций нет вообще
                    self.frames_without_coins += 1
                
                # Если давно не видели монет - начинаем НЕПРЕРЫВНЫЙ поиск
                if self.frames_without_coins >= self.search_threshold:
                    self.start_search()
                
                # Обновление FPS
                self.frame_counter.log()
                
                # Отображение (если включено и поддерживается)
                if self.show_window:
                    try:
                        annotated = results[0].plot()
                        
                        # Добавляем FPS на кадр
                        fps_text = f"FPS: {self.frame_counter.fps:.1f}"
                        cv2.putText(annotated, fps_text, (10, 30),
                                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                        
                        cv2.imshow('YOLOv10 Bot', annotated)
                        
                        # Выход по 'q'
                        if cv2.waitKey(1) & 0xFF == ord('q'):
                            print("\n[STOP] Ostanovka bota...")
                            break
                    except cv2.error as e:
                        print(f"[ERROR] OpenCV GUI error: {e}")
                        print("[INFO] Disabling display window...")
                        self.show_window = False
        
        except KeyboardInterrupt:
            print("\n[STOP] Ostanovka bota (Ctrl+C)...")
        
        finally:
            if self.show_window:
                try:
                    cv2.destroyAllWindows()
                except cv2.error:
                    pass  # Игнорируем ошибки при закрытии окон
            print("[EXIT] Bot ostanovlen")


def main():
    parser = argparse.ArgumentParser(description='YOLOv10 Bot dlya MM2')
    parser.add_argument('--weights', type=str, 
                       default='weights/ball_v10.pt',
                       help='Put\' k vesam YOLOv10')
    parser.add_argument('--conf', type=float, default=0.25,
                       help='Porog uverennosti (0.0-1.0)')
    parser.add_argument('--no-window', action='store_true',
                       help='Ne pokazyvat\' okno s detekciyami')
    
    args = parser.parse_args()
    
    # Проверка существования весов
    if not os.path.exists(args.weights):
        print(f"[ERROR] Vesa ne naydeny: {args.weights}")
        print("\n[INFO] Snacha obuchy model' v Colab!")
        print("Sm. instrukcii v: YOLOV10_COLAB.md")
        return
    
    # Запуск бота
    bot = YOLOv10Bot(
        weights_path=args.weights,
        conf_thres=args.conf,
        show_window=not args.no_window
    )
    
    bot.run()


if __name__ == '__main__':
    main()

