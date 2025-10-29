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
        
        # Параметры движения (улучшены)
        self.min_distance = 0.03  # Минимальное расстояние до объекта
        self.turn_threshold = 35  # Порог для поворота (пиксели) - уменьшен для более плавного движения
        self.smooth_turn_factor = 0.3  # Фактор плавности поворота
        
        # Параметры поворота камеры
        self.frames_without_coins = 0  # Счётчик кадров без монет
        self.search_threshold = 8  # После скольких кадров начинать поиск (увеличено)
        self.is_searching = False  # Флаг режима поиска
        self.is_moving_to_coin = False  # Флаг движения к монете
        self.search_direction = 1  # 1 для вправо, -1 для влево
        self.search_counter = 0  # Счётчик для поиска
        
        # Антизастревание (улучшено)
        self.frames_since_jump = 0  # Кадров с последнего прыжка
        self.jump_interval = 15  # Прыгать каждые N кадров (~1 сек)
        
        # Обнаружение застревания
        self.last_coin_position = None  # Последняя позиция конфеты
        self.position_history = []  # История позиций для обнаружения застревания
        self.stuck_frames = 0  # Кадров в застревании
        self.stuck_threshold = 30  # После скольких кадров считать застрявшим
        
        # Приоритизация конфет
        self.last_collected_coins = []  # История собранных конфет (избегаем повторного сбора)
        
        # Точность управления камерой
        self.camera_turn_threshold = 60  # Порог для поворота камеры
        self.camera_adjustment_speed = 0.12  # Скорость поворота камеры
        
        # Улучшенное движение
        self.target_coin = None  # Текущая цель
        self.movement_smoothness = 0.4  # Плавность движения
        
        print("[OK] Bot gotov k rabote!")
        print(f"[INFO] Confidence threshold: {self.conf_thres}")
        print(f"[INFO] Show window: {self.show_window}")
        print(f"[INFO] Улучшенная логика сбора конфет активирована!")
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
        Находит ближайшую монету/мячик с улучшенной логикой приоритизации
        
        Args:
            boxes: Список детекций
            frame_shape: Размеры кадра (height, width)
        
        Returns:
            Координаты центра ближайшего объекта или None
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
                
                # Размер объекта (для определения близости)
                box_width = x2 - x1
                box_height = y2 - y1
                box_area = box_width * box_height
                box_size = np.sqrt(box_area)  # Используем корень площади для нормализации
                
                # Нормализованные координаты
                norm_x = cx / frame_w
                norm_y = cy / frame_h
                norm_size = box_size / min(frame_w, frame_h)
                
                # Улучшенные фильтры (менее строгие, но логичные)
                # Игнорируем объекты в очень верхней части (недостижимы)
                if norm_y < 0.25:
                    continue
                
                # Игнорируем очень маленькие объекты (слишком далеко)
                if norm_size < 0.02:
                    continue
                
                # Расстояние до центра экрана
                pixel_distance = np.sqrt((cx - center_x)**2 + (cy - center_y)**2)
                
                # Комплексная оценка приоритета:
                # 1. Больший размер = выше приоритет (ближе)
                # 2. Ближе к центру = выше приоритет
                # 3. Выше уверенность = выше приоритет
                # Используем взвешенную оценку
                size_score = box_size * 2.0  # Большой вес размера
                center_score = max(0, 100 - pixel_distance / 10)  # Ближе к центру = лучше
                conf_score = conf * 50  # Уверенность модели
                
                priority_score = size_score + center_score + conf_score
                
                candidates.append({
                    'cx': cx,
                    'cy': cy,
                    'conf': conf,
                    'distance': pixel_distance,
                    'size': box_size,
                    'norm_x': norm_x,
                    'norm_y': norm_y,
                    'priority': priority_score,
                    'box': (x1, y1, x2, y2)
                })
        
        if not candidates:
            return None
        
        # Сортируем по приоритету (больше = лучше)
        candidates.sort(key=lambda x: x['priority'], reverse=True)
        best_coin = candidates[0]
        
        return best_coin
    
    def move_to_coin(self, coin, frame_shape):
        """
        Двигается к монете/мячику с улучшенной логикой и обнаружением застревания
        
        Args:
            coin: Информация о монете (cx, cy, conf, distance, size, ...)
            frame_shape: Размеры кадра
        """
        frame_h, frame_w = frame_shape[:2]
        center_x = frame_w / 2
        
        cx = coin['cx']
        cy = coin['cy']
        coin_size = coin.get('size', 10)
        
        # Если был в режиме поиска - останавливаем поиск
        if self.is_searching:
            self.control.release_all_keys()
            self.is_searching = False
            self.stuck_frames = 0  # Сбрасываем счётчик застревания
            print("[STOP SEARCH] Konfeta naydena!")
        
        # Обнаружение застревания - проверяем, движется ли конфета ближе
        current_coin_pos = (cx, cy, coin_size)
        
        if self.last_coin_position is not None:
            last_cx, last_cy, last_size = self.last_coin_position
            
            # Вычисляем изменение позиции
            pos_delta = np.sqrt((cx - last_cx)**2 + (cy - last_cy)**2)
            size_change = coin_size - last_size
            
            # Если позиция почти не меняется И размер не увеличивается (не приближаемся)
            # и прошло достаточно времени - считаем застрявшим
            if pos_delta < 5 and size_change < 2:
                self.stuck_frames += 1
            else:
                self.stuck_frames = max(0, self.stuck_frames - 2)  # Уменьшаем счётчик если движемся
        
        self.last_coin_position = current_coin_pos
        
        # Если застряли - применяем специальную стратегию
        if self.stuck_frames >= self.stuck_threshold:
            print(f"[STUCK] Zastryali! Primenyaem strategiyu vykhoda (frame {self.stuck_frames})")
            self.control.release_all_keys()
            
            # Стратегия: отступаем назад и в сторону, затем прыгаем
            time.sleep(0.05)
            self.control.press('down')  # Отступаем
            if cx < center_x:
                self.control.press('right')  # В сторону
            else:
                self.control.press('left')
            self.control.press('jump')
            time.sleep(0.2)
            self.control.release_all_keys()
            self.stuck_frames = 0  # Сбрасываем после попытки выхода
            self.frames_since_jump = 0
            time.sleep(0.1)
            return  # Выходим, даём кадру обработаться
        
        # Улучшенный поворот с плавностью
        offset = cx - center_x
        abs_offset = abs(offset)
        
        current_actions = self.control.current_actions()
        
        # Если конфета очень близко по размеру - возможно мы её собираем
        # Используем адаптивный порог в зависимости от размера экрана
        size_threshold = max(frame_w * 0.12, frame_h * 0.12)  # ~12% от размера экрана
        if coin_size > size_threshold:  # Большой размер = очень близко или собираем
            # Просто идём прямо и прыгаем для сбора
            if 'up' not in current_actions:
                self.control.release_all_keys()
            self.control.press('up')
            # Чаще прыгаем при близкой конфете
            if self.frames_since_jump >= self.jump_interval // 2:
                self.control.press('jump')
                time.sleep(0.08)
                self.control.release('jump')
                self.frames_since_jump = 0
        elif abs_offset < self.turn_threshold:
            # Монета по центру - идём вперёд
            if 'up' not in current_actions or len(current_actions) > 1:
                self.control.release_all_keys()
                self.control.press('up')
        else:
            # Нужен поворот - используем плавную логику
            turn_needed = offset < -self.turn_threshold / 2  # Уменьшен порог для более частых коррекций
            
            if turn_needed:
                # Поворачиваем влево (конфета слева)
                if 'left' not in current_actions or 'up' not in current_actions:
                    self.control.release_all_keys()
                    self.control.press('left')
                    self.control.press('up')
            else:
                # Поворачиваем вправо (конфета справа)
                if 'right' not in current_actions or 'up' not in current_actions:
                    self.control.release_all_keys()
                    self.control.press('right')
                    self.control.press('up')
        
        self.is_moving_to_coin = True
        
        # Регулярные прыжки для преодоления препятствий (антизастревание)
        self.frames_since_jump += 1
        if self.frames_since_jump >= self.jump_interval:
            self.control.press('jump')
            time.sleep(0.08)  # Немного короче для более отзывчивого управления
            self.control.release('jump')
            self.frames_since_jump = 0
    
    def turn_camera_to_coin(self, coin, frame_shape):
        """
        Поворачивает камеру к монете с более точным управлением
        
        Args:
            coin: Информация о монете (cx, cy, conf, distance, ...)
            frame_shape: Размеры кадра
        """
        frame_h, frame_w = frame_shape[:2]
        center_x = frame_w / 2
        cx = coin['cx']
        
        # Определяем, насколько далеко монета от центра
        offset = cx - center_x
        abs_offset = abs(offset)
        
        # Используем адаптивный порог - если конфета далеко по размеру, увеличиваем порог
        coin_size = coin.get('size', 10)
        adaptive_threshold = self.camera_turn_threshold
        
        # Если конфета маленькая (далеко) - камера нужна только при большом смещении
        if coin_size < frame_w * 0.05:
            adaptive_threshold *= 1.5
        
        # Поворачиваем камеру только если монета действительно сильно сбоку
        # И не слишком часто (не каждый кадр)
        if abs_offset > adaptive_threshold:
            turn_duration = min(self.camera_adjustment_speed, abs_offset / 1000)
            
            if offset < -adaptive_threshold:
                # Монета слева - поворачиваем камеру влево
                self.control.press('turn_left')
                time.sleep(turn_duration)
                self.control.release('turn_left')
            elif offset > adaptive_threshold:
                # Монета справа - поворачиваем камеру вправо
                self.control.press('turn_right')
                time.sleep(turn_duration)
                self.control.release('turn_right')
    
    def start_search(self):
        """
        Улучшенный поиск монет с более умной стратегией
        """
        if not self.is_searching:
            self.is_searching = True
            self.is_moving_to_coin = False
            self.search_counter = 0
            self.last_coin_position = None  # Сбрасываем историю
            self.stuck_frames = 0
            print("[SEARCH] Intellektual'nyy poisk konfet...")
        
        self.search_counter += 1
        
        # Останавливаем движение персонажа (но продолжаем искать камерой)
        current_actions = self.control.current_actions()
        if 'up' in current_actions or 'left' in current_actions or 'right' in current_actions:
            self.control.release('up')
            self.control.release('left')
            self.control.release('right')
        
        # Чередуем направление поиска для покрытия всех углов
        # Каждые 15 кадров меняем направление
        if self.search_counter % 30 < 15:
            # Поворачиваем вправо
            self.control.press('turn_right')
            time.sleep(self.camera_adjustment_speed * 1.2)  # Немного медленнее для поиска
            self.control.release('turn_right')
        else:
            # Поворачиваем влево
            self.control.press('turn_left')
            time.sleep(self.camera_adjustment_speed * 1.2)
            self.control.release('turn_left')
        
        # Пауза для анализа кадра
        time.sleep(0.03)
        
        # Прыжки во время поиска (чтобы не застревать)
        self.frames_since_jump += 1
        if self.frames_since_jump >= self.jump_interval * 2:  # Реже во время поиска
            self.control.press('jump')
            time.sleep(0.1)
            self.control.release('jump')
            self.frames_since_jump = 0
    
    def run(self):
        """
        Основной цикл бота с улучшенной логикой
        """
        try:
            for img, img0 in self.stream:
                # Детекция объектов (используем уменьшенное изображение)
                results = self.model(img, conf=self.conf_thres, verbose=False)
                
                # Обработка результатов
                if len(results) > 0 and len(results[0].boxes) > 0:
                    boxes = results[0].boxes
                    
                    # Находим лучшую конфету с приоритизацией
                    closest_coin = self.find_closest_coin(boxes, img.shape)
                    
                    if closest_coin:
                        # Конфета найдена! Сбрасываем счётчики
                        self.frames_without_coins = 0
                        
                        # Сохраняем целевую конфету
                        self.target_coin = closest_coin
                        
                        # Поворачиваем камеру к конфете (если она сбоку) - делаем это реже
                        # чтобы не мешать движению персонажа (каждые несколько кадров)
                        frame_counter = getattr(self, '_frame_count', 0)
                        self._frame_count = frame_counter + 1
                        if not self.is_moving_to_coin or frame_counter % 5 == 0:
                            self.turn_camera_to_coin(closest_coin, img.shape)
                        
                        # Двигаемся к конфете
                        self.move_to_coin(closest_coin, img.shape)
                        
                        # Улучшенное логирование
                        size_info = f"Size: {closest_coin.get('size', 0):.1f}"
                        priority_info = f"Priority: {closest_coin.get('priority', 0):.1f}"
                        stuck_info = f"Stuck: {self.stuck_frames}" if self.stuck_frames > 0 else ""
                        
                        log_msg = (f"[CONFETA] Conf: {closest_coin['conf']:.2f}, "
                                  f"Dist: {closest_coin['distance']:.1f}px, {size_info}")
                        if stuck_info:
                            log_msg += f", {stuck_info}"
                        print(log_msg)
                    else:
                        # Конфета не в зоне видимости (фильтры сработали)
                        self.frames_without_coins += 1
                        # Сбрасываем целевую конфету если её больше не видно
                        if self.frames_without_coins > 3:
                            self.target_coin = None
                            self.last_coin_position = None
                else:
                    # Детекций нет вообще
                    self.frames_without_coins += 1
                    self.target_coin = None
                    self.last_coin_position = None
                
                # Если давно не видели конфет - начинаем поиск
                if self.frames_without_coins >= self.search_threshold:
                    self.start_search()
                
                # Обновление FPS
                self.frame_counter.log()
                
                # Отображение (если включено и поддерживается)
                if self.show_window:
                    try:
                        annotated = results[0].plot()
                        
                        # Добавляем информацию на кадр
                        fps_text = f"FPS: {self.frame_counter.fps:.1f}"
                        cv2.putText(annotated, fps_text, (10, 30),
                                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                        
                        # Добавляем статус
                        status_y = 70
                        if self.is_searching:
                            status_text = "SEARCHING..."
                            cv2.putText(annotated, status_text, (10, status_y),
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                        elif self.target_coin:
                            status_text = f"MOVING | Stuck: {self.stuck_frames}"
                            color = (0, 255, 0) if self.stuck_frames == 0 else (0, 165, 255)
                            cv2.putText(annotated, status_text, (10, status_y),
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                        
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

