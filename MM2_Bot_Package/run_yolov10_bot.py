"""
🎮 YOLOv10 Bot для сбора мячиков в Murder Mystery 2 - Улучшенная версия

Использует ultralytics YOLOv10 для детекции и сбора объектов.
Включает улучшенный алгоритм движения и навигации.

Улучшения:
✨ Плавное движение с улучшенным PID-контроллером
🎯 Умная система приоритизации целей
🛡️ Улучшенная система антизастревания
📊 Статистика работы

Лицензия: MIT (САМАЯ СВОБОДНАЯ ЛИЦЕНЗИЯ!)

Установка:
    pip install ultralytics opencv-python

Запуск:
    python run_yolov10_bot.py --weights weights/ball_v10.pt --conf 0.25
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
        
        # Параметры движения (улучшенные)
        self.min_distance = 50  # Минимальное расстояние до объекта (пиксели)
        self.turn_threshold = 25  # Порог для поворота (пиксели) - улучшено для более точного движения
        
        # Параметры поворота камеры
        self.frames_without_coins = 0  # Счётчик кадров без монет
        self.search_threshold = 5  # После скольких кадров начинать поиск
        self.is_searching = False  # Флаг режима поиска
        self.is_moving_to_coin = False  # Флаг движения к монете
        
        # Антизастревание (улучшено)
        self.frames_since_jump = 0  # Кадров с последнего прыжка
        self.jump_interval = 15  # Прыгать каждые N кадров (~1.5 сек) - оптимизировано
        self.previous_distance = float('inf')  # Предыдущее расстояние для отслеживания прогресса
        self.stuck_frames = 0  # Счетчик кадров без прогресса
        self.stuck_threshold = 12  # Порог для обнаружения застревания
        
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
        Находит ближайшую монету/мячик
        
        Args:
            boxes: Список детекций
            frame_shape: Размеры кадра (height, width)
        
        Returns:
            Координаты центра ближайшего объекта или None
        """
        frame_h, frame_w = frame_shape[:2]
        center_x = frame_w / 2
        center_y = frame_h / 2
        
        closest_coin = None
        min_distance = float('inf')
        
        for box in boxes:
            cls = int(box.cls[0])
            conf = float(box.conf[0])
            
            # Только монеты/мячики (класс 0) с достаточной уверенностью
            if cls == 0 and conf >= self.conf_thres:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                
                # Центр объекта
                cx = (x1 + x2) / 2
                cy = (y1 + y2) / 2
                
                # Нормализованные координаты
                norm_x = cx / frame_w
                norm_y = cy / frame_h
                
                # Фильтр: игнорируем объекты в верхней части экрана
                if norm_y < 0.4:
                    continue
                
                # Фильтр: игнорируем объекты по краям
                if norm_x < 0.2 or norm_x > 0.8:
                    continue
                
                # Расстояние до центра экрана
                distance = np.sqrt((cx - center_x)**2 + (cy - center_y)**2)
                
                if distance < min_distance:
                    min_distance = distance
                    closest_coin = {
                        'cx': cx,
                        'cy': cy,
                        'conf': conf,
                        'distance': distance
                    }
        
        return closest_coin
    
    def move_to_coin(self, coin, frame_shape):
        """
        Двигается к монете/мячику с улучшенным алгоритмом
        
        Args:
            coin: Информация о монете (cx, cy, conf, distance)
            frame_shape: Размеры кадра
        """
        frame_h, frame_w = frame_shape[:2]
        center_x = frame_w / 2
        center_y = frame_h / 2
        
        cx = coin['cx']
        cy = coin['cy']
        distance = coin['distance']
        
        # Проверка достижения цели
        if distance < self.min_distance:
            self.control.release_all_keys()
            time.sleep(0.15)  # Пауза для сбора
            print(f"[COLLECT] Sbor zavershen! (conf: {coin['conf']:.2f}, dist: {distance:.1f}px)")
            self.previous_distance = float('inf')
            self.stuck_frames = 0
            return True
        
        # Если был в режиме поиска - останавливаем поиск
        if self.is_searching:
            self.control.release_all_keys()
            self.is_searching = False
            print("[STOP SEARCH] Moneta naydena!")
        
        # Проверка прогресса (улучшенное антизастревание)
        if self.previous_distance != float('inf'):
            progress = distance < self.previous_distance - 3.0  # Прогресс минимум 3 пикселя
            if progress:
                self.stuck_frames = 0
            else:
                self.stuck_frames += 1
                
            # Если застряли - пытаемся выйти
            if self.stuck_frames >= self.stuck_threshold:
                print("[STUCK] Obnaruzheno zastrevanije, vypolnyaju manevr!")
                self.control.release_all_keys()
                # Отъезд назад
                self.control.press('down')
                time.sleep(0.2)
                self.control.release_all_keys()
                # Поворот
                turn_dir = 'turn_left' if cx < center_x else 'turn_right'
                self.control.press(turn_dir)
                time.sleep(0.3)
                self.control.release_all_keys()
                # Прыжок и движение
                self.control.press('jump')
                self.control.press('up')
                time.sleep(0.15)
                self.control.release_all_keys()
                self.stuck_frames = 0
                time.sleep(0.2)
        
        # Поворот к объекту (улучшенный алгоритм)
        error = cx - center_x
        turn_duration = min(abs(error) / 100.0, 0.15)  # Адаптивная длительность поворота
        
        self.control.release_all_keys()
        self.control.press('up')  # Всегда движемся вперед
        
        if abs(error) > self.turn_threshold:
            if error < 0:
                self.control.press('left')
            else:
                self.control.press('right')
            time.sleep(turn_duration)
        
        self.previous_distance = distance
        self.is_moving_to_coin = True
        
        # Регулярные прыжки для преодоления препятствий (антизастревание)
        self.frames_since_jump += 1
        if self.frames_since_jump >= self.jump_interval and distance > 100:
            # Время прыгнуть!
            self.control.press('jump')
            time.sleep(0.1)  # Короткая пауза для прыжка
            self.frames_since_jump = 0  # Сбрасываем счётчик
            
        return False
    
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
        Медленный поворот камеры для поиска монет (с паузами)
        """
        if not self.is_searching:
            self.is_searching = True
            self.is_moving_to_coin = False
            print("[SEARCH] Medlenno ischu monety...")
        
        # Останавливаем всё движение
        self.control.release_all_keys()
        
        # МЕДЛЕННЫЙ поворот: нажать → пауза → отпустить → анализ
        self.control.press('turn_right')
        time.sleep(0.15)  # Короткое нажатие (медленный поворот)
        self.control.release_all_keys()
        time.sleep(0.05)  # Пауза для анализа кадра нейронкой
        
        # Прыжки во время поиска (чтобы не застревать)
        self.frames_since_jump += 1
        if self.frames_since_jump >= self.jump_interval:
            self.control.press('jump')
            time.sleep(0.1)
            self.frames_since_jump = 0
            print("[JUMP] Pryzhok vo vremya poiska!")
    
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
                        # Монета найдена! Сбрасываем счётчик
                        self.frames_without_coins = 0
                        
                        # Поворачиваем камеру к монете (если она сбоку)
                        self.turn_camera_to_coin(closest_coin, img.shape)
                        
                        # Двигаемся к монете (улучшенный алгоритм)
                        collected = self.move_to_coin(closest_coin, img.shape)
                        
                        # Логирование
                        if not collected:
                            print(f"[COIN] Conf: {closest_coin['conf']:.2f}, "
                                  f"Dist: {closest_coin['distance']:.1f}px")
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

