"""
RT-DETR Bot v8 Final: Clean, no crashes, smooth PID, GPU auto
Core: detect → PID → collect → search. FPS 5+, multi-coin.
"""

import sys
import os
import argparse
import time
import numpy as np
import torch
import random

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False
    print("[WARNING] cv2 not installed. No GUI. pip install opencv-contrib-python")

sys.path.insert(0, os.path.dirname(__file__))

from ultralytics import YOLO
from roblox.screen import CaptureStream
from roblox.control import Control
from roblox.utils import FrameCounter

class RTDETRBot:
    def __init__(self, weights_path, conf_thres=0.25, show_window=True):
        print(f"[INIT] Загрузка RT-DETR: {weights_path}")
        
        self.device = '0' if torch.cuda.is_available() else ''
        print(f"[INFO] Device: {'cuda:0' if self.device else 'CPU'}")
        
        self.model = YOLO(weights_path)
        self.conf_thres = conf_thres
        self.show_window = show_window and CV2_AVAILABLE
        
        if self.show_window:
            print("[INFO] GUI enabled")
        else:
            print("[INFO] No GUI")
        
        print("[INIT] Захват экрана...")
        self.stream = CaptureStream("Roblox", saveInterval=0)
        
        self.control = Control()
        self.frame_counter = FrameCounter()
        self.frame_counter.fps = 0.0
        
        # PID
        self.pid_kp = 0.03
        self.pid_ki = 0.001
        self.pid_kd = 0.02
        self.integral = 0.0
        self.prev_error = 0.0
        self.dt = 0.05
        self.integral_limit = 100.0
        
        # Params (улучшенные)
        self.min_distance = 40  # Базовое минимальное расстояние
        self.min_distance_close = 25  # Для крупных объектов
        self.turn_threshold = 30
        self.turn_threshold_near = 20  # Более точные повороты вблизи
        self.paste_delay = 0.3
        self.predict_size = 320
        
        self.frames_without_coins = 0
        self.search_threshold = 5
        self.is_searching = False
        self.is_moving_to_coin = False
        self.frames_since_jump = 0
        self.jump_interval = 15
        self.previous_distance = float('inf')
        self.frames_after_collect = 0
        self.post_collect_frames = 10
        self.retry_frames = 0
        self.retry_threshold = 3
        
        # Улучшенное антизастревание
        self.stuck_frames = 0
        self.stuck_threshold = 25
        self.no_progress_frames = 0
        
        # Умный поиск
        self.search_pattern_index = 0
        self.search_patterns = [
            ('turn_right', 0.2),
            ('turn_left', 0.2),
            ('turn_right', 0.15),
            ('turn_left', 0.15),
        ]
        
        print("[OK] Bot готов!")
        print(f"[INFO] Conf: {self.conf_thres}, Min dist: {self.min_distance}px")
        print("\n[START] Ctrl+C для остановки\n")
    
    def find_closest_coins(self, results, frame_shape, max_coins=2):
        """
        Улучшенная приоритизация монет с несколькими факторами
        """
        frame_h, frame_w = frame_shape[:2]
        center_x = frame_w / 2
        center_y = frame_h / 2
        
        coins = []
        boxes = results[0].boxes
        if boxes is not None:
            for i in range(len(boxes)):
                conf = boxes.conf[i].item()
                cls = int(boxes.cls[i].item())
                
                if cls == 0 and conf >= self.conf_thres:
                    xyxy = boxes.xyxy[i].cpu().numpy()
                    x1, y1, x2, y2 = xyxy
                    
                    cx = (x1 + x2) / 2
                    cy = (y1 + y2) / 2
                    
                    # Размер объекта
                    area = (x2 - x1) * (y2 - y1)
                    size_ratio = area / (frame_w * frame_h)
                    
                    norm_x = cx / frame_w
                    norm_y = cy / frame_h
                    
                    # Мягкие фильтры
                    if norm_y < 0.3:
                        continue
                    
                    edge_penalty = 1.0
                    if norm_x < 0.15 or norm_x > 0.85:
                        edge_penalty = 0.7
                    elif norm_x < 0.2 or norm_x > 0.8:
                        edge_penalty = 0.85
                    
                    distance = np.sqrt((cx - center_x)**2 + (cy - center_y)**2)
                    
                    # Улучшенный скор
                    distance_score = 1.0 / (1.0 + distance / 100.0)
                    confidence_score = conf
                    size_score = min(size_ratio * 100, 1.0)
                    angle_score = 1.0 - min(abs(cx - center_x) / frame_w * 2, 0.5)
                    
                    priority = (
                        distance_score * 0.4 +
                        confidence_score * 0.3 +
                        size_score * 0.2 +
                        angle_score * 0.1
                    ) * edge_penalty
                    
                    coins.append({
                        'cx': cx, 'cy': cy, 'conf': conf, 'distance': distance,
                        'score': -priority,  # Для сортировки по возрастанию (лучшие первыми)
                        'size_ratio': size_ratio,
                        'priority': priority
                    })
        
        coins.sort(key=lambda c: c['score'])
        return coins[:max_coins]
    
    def move_to_coin(self, coin, frame_shape):
        frame_w = frame_shape[1]
        center_x = frame_w / 2
        center_y = frame_shape[0] / 2
        
        cx = coin['cx']
        cy = coin['cy']
        distance = coin['distance']
        conf = coin['conf']
        
        current_distance = distance
        error = cx - center_x
        self.integral += error * self.dt
        self.integral = np.clip(self.integral, -self.integral_limit, self.integral_limit)
        derivative = (error - self.prev_error) / self.dt if self.dt > 0 else 0
        pid_output = self.pid_kp * error + self.pid_ki * self.integral + self.pid_kd * derivative
        
        progress = self.previous_distance - current_distance > 10
        self.prev_error = error
        
        if self.is_searching:
            self.control.release_all_keys()
            self.is_searching = False
            print("[STOP SEARCH] Монета найдена!")
        
        # Адаптивное расстояние сбора в зависимости от размера объекта
        size_ratio = coin.get('size_ratio', 0)
        collect_distance = self.min_distance_close if size_ratio > 0.01 else self.min_distance
        
        if distance < collect_distance:
            self.control.release_all_keys()
            time.sleep(self.paste_delay)
            
            print(f"[COLLECT] Подобрали! (conf: {conf:.2f}, dist: {distance:.1f}px, size: {size_ratio*100:.2f}%)")
            self.integral = 0.0
            self.previous_distance = float('inf')
            self.is_moving_to_coin = False
            self.frames_after_collect = self.post_collect_frames
            self.no_progress_frames = 0
            return True
        
        # Адаптивный порог поворота
        adaptive_threshold = self.turn_threshold_near if distance < 150 else self.turn_threshold
        
        # Проверка прогресса
        if not progress:
            self.no_progress_frames += 1
            if self.no_progress_frames > self.stuck_threshold:
                self.stuck_frames += 1
                print(f"[STUCK] Застревание! Попытка выхода ({self.stuck_frames})...")
                self.control.release_all_keys()
                time.sleep(0.1)
                self.control.press('down')
                time.sleep(0.15)
                self.control.press('jump')
                time.sleep(0.2)
                if self.stuck_frames % 2 == 0:
                    self.control.press('left')
                    time.sleep(0.2)
                else:
                    self.control.press('right')
                    time.sleep(0.2)
                self.control.release_all_keys()
                self.no_progress_frames = 0
                self.previous_distance = float('inf')
                return False
        else:
            self.no_progress_frames = 0
        
        turn_time = abs(pid_output) / 10
        turn_time = max(0.05, min(0.2, turn_time))
        if distance < 200:
            turn_time *= 0.5
        
        self.control.release_all_keys()
        
        if abs(error) > adaptive_threshold:
            if error < 0:
                self.control.press('left')
            else:
                self.control.press('right')
            self.control.press('up')
            time.sleep(turn_time)
        else:
            self.control.press('up')
            time.sleep(0.1)
        
        if not progress:
            self.retry_frames += 1
            if self.retry_frames > self.retry_threshold:
                print("[RETRY] Нет прогресса — retry!")
                self.control.release_all_keys()
                time.sleep(0.2)
                self.control.press('jump')
                time.sleep(0.3)
                self.control.press('up')
                self.retry_frames = 0
        else:
            self.retry_frames = 0
        
        self.previous_distance = current_distance
        self.is_moving_to_coin = True
        
        self.frames_since_jump += 1
        if self.frames_since_jump >= self.jump_interval and distance > 150:
            self.control.press('jump')
            time.sleep(0.1)
            self.frames_since_jump = 0
            print("[JUMP] Для приближения!")
        
        return False
    
    def start_search(self):
        if not self.is_searching:
            self.is_searching = True
            self.is_moving_to_coin = False
            self.stuck_frames = 0
            print(f"[SEARCH] Умный поиск (pattern: {self.search_pattern_index % 4})...")
        
        self.control.release_all_keys()
        
        # Используем паттерны поиска
        pattern_type, pattern_duration = self.search_patterns[self.search_pattern_index % len(self.search_patterns)]
        self.control.press(pattern_type)
        time.sleep(pattern_duration)
        self.control.release_all_keys()
        
        self.search_pattern_index += 1
        time.sleep(0.08)
        
        # Периодическое движение вперёд
        if self.frames_without_coins % 10 == 5:
            self.control.press('up')
            time.sleep(0.2)
            self.control.release_all_keys()
        
        self.frames_since_jump += 1
        if self.frames_since_jump >= self.jump_interval:
            self.control.press('jump')
            time.sleep(0.1)
            self.frames_since_jump = 0
    
    def run(self):
        try:
            prev_time = time.time()
            for img, img0 in self.stream:
                results = self.model(img, imgsz=self.predict_size, device=self.device, conf=self.conf_thres, verbose=False)
                
                collected = False
                if len(results) > 0:
                    coins = self.find_closest_coins(results, img.shape, max_coins=2)
                    
                    if coins:
                        self.frames_without_coins = 0
                        
                        closest_coin = coins[0]
                        
                        collected = self.move_to_coin(closest_coin, img.shape)
                        
                        print(f"[DETECT] Conf: {closest_coin['conf']:.2f}, Dist: {closest_coin['distance']:.1f}px (top {len(coins)})")
                    else:
                        if self.frames_after_collect <= 0:
                            self.frames_without_coins += 1
                        else:
                            self.frames_after_collect -= 1
                else:
                    if self.frames_after_collect <= 0:
                        self.frames_without_coins += 1
                    else:
                        self.frames_after_collect -= 1
                
                if self.frames_without_coins >= self.search_threshold and self.frames_after_collect <= 0:
                    self.start_search()
                
                current_time = time.time()
                self.dt = current_time - prev_time
                prev_time = current_time
                self.frame_counter.log()
                
                # GUI optional
                if self.show_window:
                    try:
                        annotated = results[0].plot() if len(results) > 0 else img.copy()
                        
                        fps_text = f"FPS: {self.frame_counter.fps:.1f}"
                        status = "SEARCH" if self.is_searching else ("MOVING" if self.is_moving_to_coin else "IDLE")
                        cv2.putText(annotated, fps_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                        cv2.putText(annotated, f"Status: {status}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
                        if collected:
                            cv2.putText(annotated, "ПОДОБРАЛИ!", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                        
                        cv2.imshow('RT-DETR Bot v8', annotated)
                        
                        if cv2.waitKey(1) & 0xFF == ord('q'):
                            break
                    except Exception as e:
                        print(f"[ERROR] GUI: {e}")
                        self.show_window = False
        
        except KeyboardInterrupt:
            print("\n[STOP] Ctrl+C...")
        
        finally:
            if self.show_window:
                try:
                    cv2.destroyAllWindows()
                except Exception as e:
                    print(f"[ERROR] GUI close: {e}")
            print("[EXIT] Bot остановлен")

def main():
    parser = argparse.ArgumentParser(description='RT-DETR Bot v8 Final')
    parser.add_argument('--weights', type=str, default='weights/ball_rtdetr.pt', help='Путь к весам')
    parser.add_argument('--conf', type=float, default=0.25, help='Порог уверенности')
    parser.add_argument('--no-window', action='store_true', help='Без окна')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.weights):
        print(f"[ERROR] Веса не найдены: {args.weights}")
        return
    
    bot = RTDETRBot(args.weights, args.conf, not args.no_window)
    bot.run()

if __name__ == '__main__':
    main()