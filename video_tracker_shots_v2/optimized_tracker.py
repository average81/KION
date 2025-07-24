"""
Оптимизированный трекер объектов в видео
Точная копия функциональности из integrated_tracker_optimized.py
"""

import cv2
import numpy as np
import torch
from ultralytics import YOLO
import pandas as pd
# from moviepy.editor import VideoFileClip, AudioFileClip  # Убираем зависимость от moviepy
import os
from collections import deque, defaultdict
import time
from typing import List, Tuple, Optional, Dict, Any
import logging
from pathlib import Path
import shutil
import sys

# Настройка логирования
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class OptimizedBoundingBoxTracker:
    """Оптимизированный трекер объектов с улучшенной производительностью"""
    
    def __init__(self, max_history: int = 30, interpolation_frames: int = 20, 
                 iou_threshold: float = 0.3, min_detections_for_tracking: int = 6,
                 min_confidence: float = 0.7):
        self.objects: Dict[int, 'TrackedObject'] = {}
        self.next_id = 0
        self.max_history = max_history
        self.interpolation_frames = interpolation_frames
        self.iou_threshold = iou_threshold
        self.min_detections_for_tracking = min_detections_for_tracking
        self.min_confidence = min_confidence
        self.current_frame = 0
        
        # Кэш для ускорения вычислений
        self._iou_cache = {}
        self._frame_cache_size = 100
        
    def update(self, detections: List[Tuple], frame_num: int) -> List[Tuple]:
        """Обновляет трекер с новыми детекциями"""
        self.current_frame = frame_num
        self._clear_old_cache()
        
        if not detections:
            return self._get_all_boxes()
            
        # Сопоставляем детекции с существующими объектами
        matched_detections, unmatched_detections = self._match_detections(detections)
        
        # Обновляем существующие объекты
        for detection, obj_id in matched_detections:
            self.objects[obj_id].update(detection, frame_num)
            
        # Создаем новые объекты для несопоставленных детекций
        for detection in unmatched_detections:
            self._create_new_object(detection, frame_num)
            
        # Удаляем устаревшие объекты
        self._remove_expired_objects()
        
        return self._get_all_boxes()
    
    def _match_detections(self, detections: List[Tuple]) -> Tuple[List[Tuple], List[Tuple]]:
        """Сопоставляет детекции с существующими объектами используя IoU и расстояние (как в оригинале)"""
        if not self.objects:
            return [], detections
            
        matched = []
        unmatched = []
        
        for detection in detections:
            best_match_id = None
            best_iou = 0.0
            
            # Вычисляем центр новой детекции
            new_center_x = (detection[0] + detection[2]) / 2
            new_center_y = (detection[1] + detection[3]) / 2
            
            for obj_id, obj in self.objects.items():
                last_box = obj.get_last_box()
                if last_box is not None:
                    iou = self._calculate_iou(detection[:4], last_box)
                    
                    # Вычисляем расстояние между центрами
                    last_center_x = (last_box[0] + last_box[2]) / 2
                    last_center_y = (last_box[1] + last_box[3]) / 2
                    distance = np.sqrt((new_center_x - last_center_x)**2 + (new_center_y - last_center_y)**2)
                    
                    # Условия сопоставления как в оригинале
                    if (iou > best_iou and iou > self.iou_threshold and distance < 100):
                        best_iou = iou
                        best_match_id = obj_id
                        logger.debug(f"�� Найдено совпадение: объект {obj_id}, IoU={iou:.3f}, расстояние={distance:.1f}")
            
            if best_match_id is not None:
                matched.append((detection, best_match_id))
            else:
                unmatched.append(detection)
        
        return matched, unmatched
    
    def _create_new_object(self, detection: Tuple, frame_num: int):
        """Создает новый объект для отслеживания"""
        obj = TrackedObject(
            self.next_id, detection, frame_num, 
            self.max_history, self.interpolation_frames
        )
        self.objects[self.next_id] = obj
        self.next_id += 1
        logger.info(f"🆕 Создан новый объект {obj.object_id}")
    
    def _remove_expired_objects(self):
        """Удаляет устаревшие объекты"""
        expired_objects = []
        
        for obj_id, obj in self.objects.items():
            frames_since_creation = self.current_frame - obj.creation_frame
            frames_since_seen = self.current_frame - obj.last_seen
            
            # Удаляем объекты старше 300 кадров (~12 секунд)
            if frames_since_creation > 300:
                expired_objects.append(obj_id)
                logger.info(f"🗑️ Удален старый объект {obj_id} (возраст: {frames_since_creation} кадров)")
                continue
                
            # Удаляем объекты неактивные более 25 кадров (~1 секунда) для неактивных (как в оригинале)
            if not obj.is_tracking and frames_since_seen > 25:
                expired_objects.append(obj_id)
                logger.info(f"🗑️ Удален неактивный объект {obj_id} (детекций: {obj.detection_count}, неактивен: {frames_since_seen} кадров)")
                continue
            # Удаляем активные объекты через 20 кадров (~0.8 секунды)
            elif obj.is_tracking and frames_since_seen > 20:
                expired_objects.append(obj_id)
                logger.info(f"🗑️ Удален активный объект {obj_id} (неактивен: {frames_since_seen} кадров)")
                continue
        
        for obj_id in expired_objects:
            del self.objects[obj_id]
    
    def _get_all_boxes(self) -> List[Tuple]:
        """Возвращает все активные рамки"""
        active_boxes = []
        
        for obj in self.objects.values():
            if obj.is_tracking:
                obj.check_interpolation(self.current_frame)
                box, conf = obj.get_smoothed_box()
                
                if box is None:
                    box, conf = obj.get_interpolated_box(self.current_frame)
                
                # Fallback на последнюю рамку для стабильности
                if box is None and len(obj.history) > 0:
                    last_box, last_conf = obj.history[-1]
                    box = last_box[:4]
                    conf = last_conf
                    logger.debug(f"🔄 Fallback для объекта {obj.object_id}")
                
                if box is not None:
                    active_boxes.append((*box, conf, obj.object_id))
                    logger.debug(f"✅ Объект {obj.object_id} отображается (интерполяция: {obj.is_interpolating})")
        
        return active_boxes
    
    def _calculate_distance(self, box1: Tuple, box2: Tuple) -> float:
        """Вычисляет расстояние между центрами рамок"""
        center1 = ((box1[0] + box1[2]) / 2, (box1[1] + box1[3]) / 2)
        center2 = ((box2[0] + box2[2]) / 2, (box2[1] + box2[3]) / 2)
        return np.sqrt((center1[0] - center2[0])**2 + (center1[1] - center2[1])**2)
    
    def _calculate_iou(self, box1: Tuple, box2: Tuple) -> float:
        """Вычисляет IoU между двумя рамками с кэшированием"""
        cache_key = tuple(sorted([box1, box2]))
        if cache_key in self._iou_cache:
            return self._iou_cache[cache_key]
        
        x1, y1, x2, y2 = box1
        x3, y3, x4, y4 = box2
        
        # Вычисляем пересечение
        x_left = max(x1, x3)
        y_top = max(y1, y3)
        x_right = min(x2, x4)
        y_bottom = min(y2, y4)
        
        if x_right < x_left or y_bottom < y_top:
            iou = 0.0
        else:
            intersection = (x_right - x_left) * (y_bottom - y_top)
            area1 = (x2 - x1) * (y2 - y1)
            area2 = (x4 - x3) * (y4 - y3)
            union = area1 + area2 - intersection
            iou = intersection / union if union > 0 else 0.0
        
        self._iou_cache[cache_key] = iou
        return iou
    
    def _clear_old_cache(self):
        """Очищает старый кэш для экономии памяти"""
        if len(self._iou_cache) > self._frame_cache_size:
            self._iou_cache.clear()
    
    def reset(self):
        """Сбрасывает трекер при смене сцены"""
        self.objects.clear()
        self.next_id = 0
        self._iou_cache.clear()
        logger.info("🔄 Трекер сброшен")


class TrackedObject:
    """Класс для отслеживания отдельного объекта"""
    
    def __init__(self, object_id: int, initial_detection: Tuple, creation_frame: int,
                 max_history: int, interpolation_frames: int):
        self.object_id = object_id
        self.creation_frame = creation_frame
        self.last_seen = creation_frame
        self.detection_count = 1
        self.is_tracking = False
        self.is_interpolating = False
        self.interpolation_start_frame = 0
        self.interpolation_start = None
        self.interpolation_end = None
        
        # Оптимизированное хранение истории
        self.history = deque(maxlen=max_history)
        self.interpolation_frames = interpolation_frames
        
        # Добавляем первую детекцию
        self.history.append((initial_detection[:4], initial_detection[4]))
        
        # Активируем трекинг после достаточного количества детекций
        self._check_tracking_activation()
        
    def update(self, detection: Tuple, frame_num: int):
        """Обновляет объект новой детекцией"""
        self.last_seen = frame_num
        self.detection_count += 1
        self.is_interpolating = False
        
        # Добавляем в историю
        self.history.append((detection[:4], detection[4]))
        
        # Проверяем активацию трекинга
        self._check_tracking_activation()
        
        # Логируем обновление
        if self.is_tracking:
            last_box = self.get_last_box()
            if last_box is not None:
                distance = self._calculate_distance(detection[:4], last_box)
                logger.debug(f"📏 Объект {self.object_id}: расстояние {distance:.1f}px, IoU: {self._calculate_iou(detection[:4], last_box):.2f}")
        
    def _check_tracking_activation(self):
        """Проверяет, нужно ли активировать трекинг"""
        if not self.is_tracking and self.detection_count >= 6:  # Активируем после 6 детекций как в оригинале
            self.is_tracking = True
            logger.info(f"🎯 Активирован трекинг для объекта {self.object_id} (детекций: {self.detection_count})")
        elif not self.is_tracking and self.detection_count % 1 == 0:  # Отладка каждую детекцию
            logger.info(f"📊 Объект {self.object_id}: {self.detection_count} детекций, трекинг: {self.is_tracking}")
    
    def check_interpolation(self, current_frame: int):
        """Проверяет, нужно ли начать интерполяцию"""
        frames_since_seen = current_frame - self.last_seen
        
        # Начинаем интерполяцию после 3 кадров без детекции
        if frames_since_seen > 3 and not self.is_interpolating:
            self.is_interpolating = True
            self.interpolation_start_frame = current_frame
            
            if len(self.history) >= 2:
                recent_boxes = list(self.history)[-2:]
                self.interpolation_start = recent_boxes[-1][:4]
                self.interpolation_end = recent_boxes[-2][:4]
            
            logger.debug(f"🔄 Начата интерполяция для объекта {self.object_id}")
        
        # Завершаем интерполяцию
        elif self.is_interpolating:
            if frames_since_seen > self.interpolation_frames * 2.5:
                self.is_interpolating = False
                logger.debug(f"⏹️ Завершена интерполяция для объекта {self.object_id}")
    
    def get_smoothed_box(self) -> Tuple[Optional[Tuple], Optional[float]]:
        """Возвращает сглаженную рамку"""
        if len(self.history) < 3:
            last_box = self.get_last_box()
            last_conf = self.history[-1][1] if self.history else None
            return last_box, last_conf
        
        # Простое сглаживание по последним 3 кадрам
        recent_boxes = list(self.history)[-3:]
        smoothed_box = self._average_boxes([box for box, _ in recent_boxes])
        avg_conf = sum(conf for _, conf in recent_boxes) / len(recent_boxes)
        
        return smoothed_box, avg_conf
    
    def get_interpolated_box(self, current_frame: int) -> Tuple[Optional[Tuple], Optional[float]]:
        """Возвращает интерполированную рамку"""
        if not self.is_interpolating or self.interpolation_start is None:
            return None, None
        
        frames_since_start = current_frame - self.interpolation_start_frame
        if frames_since_start > self.interpolation_frames:
            return None, None
        
        # Линейная интерполяция
        progress = frames_since_start / self.interpolation_frames
        interpolated_box = self._interpolate_boxes(self.interpolation_start, self.interpolation_end, progress)
        
        return interpolated_box, 0.5  # Средняя уверенность для интерполированных рамок
    
    def get_last_box(self) -> Optional[Tuple]:
        """Возвращает последнюю рамку"""
        return self.history[-1][0] if self.history else None
    
    def _average_boxes(self, boxes: List[Tuple]) -> Tuple:
        """Вычисляет среднюю рамку"""
        if not boxes:
            return None
        
        avg_box = [0, 0, 0, 0]
        for box in boxes:
            for i in range(4):
                avg_box[i] += box[i]
        
        return tuple(x / len(boxes) for x in avg_box)
    
    def _interpolate_boxes(self, box1: Tuple, box2: Tuple, progress: float) -> Tuple:
        """Интерполирует между двумя рамками"""
        return tuple(b1 + (b2 - b1) * progress for b1, b2 in zip(box1, box2))
    
    def _calculate_distance(self, box1: Tuple, box2: Tuple) -> float:
        """Вычисляет расстояние между центрами рамок"""
        center1 = ((box1[0] + box1[2]) / 2, (box1[1] + box1[3]) / 2)
        center2 = ((box2[0] + box2[2]) / 2, (box2[1] + box2[3]) / 2)
        return np.sqrt((center1[0] - center2[0])**2 + (center1[1] - center2[1])**2)
    
    def _calculate_iou(self, box1: Tuple, box2: Tuple) -> float:
        """Вычисляет IoU между двумя рамками"""
        x1, y1, x2, y2 = box1
        x3, y3, x4, y4 = box2
        
        x_left = max(x1, x3)
        y_top = max(y1, y3)
        x_right = min(x2, x4)
        y_bottom = min(y2, y4)
        
        if x_right < x_left or y_bottom < y_top:
            return 0.0
        
        intersection = (x_right - x_left) * (y_bottom - y_top)
        area1 = (x2 - x1) * (y2 - y1)
        area2 = (x4 - x3) * (y4 - y3)
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0.0


class OptimizedSceneDetector:
    """Оптимизированный детектор смен сцен"""
    
    def __init__(self, threshold: float = 10.0, min_scene_length: float = 0.7, 
                 required_consecutive: int = 1):
        self.threshold = threshold
        self.min_scene_length = min_scene_length
        self.required_consecutive = required_consecutive
        self.scene_changes = []
        self.prev_frame = None
        self.consecutive_detections = 0
        self.last_scene_change = 0
        
    def detect_scene_change(self, frame: np.ndarray, frame_num: int, fps: float) -> bool:
        """Детектирует смену сцены"""
        if self.prev_frame is None:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            self.prev_frame = gray
            return False
        
        # Вычисляем разность кадров (как в неоптимизированной версии)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        diff = cv2.absdiff(gray, self.prev_frame)
        mean_diff = np.mean(diff)
        normalized_diff = (mean_diff / 255.0) * 100  # Нормализация как в оригинале
        
        # Проверяем минимальную длину сцены
        min_frames = int(self.min_scene_length * fps)
        if frame_num - self.last_scene_change < min_frames:
            self.prev_frame = gray
            return False
        
        # Детектируем смену сцены (используем нормализованную разность)
        if normalized_diff > self.threshold:
            self.consecutive_detections += 1
            if self.consecutive_detections >= self.required_consecutive:
                self.scene_changes.append(frame_num)
                self.last_scene_change = frame_num
                self.consecutive_detections = 0
                logger.info(f"🎬 Смена сцены в кадре {frame_num} (разность: {normalized_diff:.1f})")
                self.prev_frame = gray
                return True
        else:
            self.consecutive_detections = 0
        
        self.prev_frame = gray
        return False
    
    def get_current_scene(self, frame_num: int) -> int:
        """Возвращает номер текущей сцены"""
        scene_num = 0
        for change_frame in self.scene_changes:
            if frame_num >= change_frame:
                scene_num += 1
        return scene_num
    
    def save_analysis(self, output_file: str, fps: float, total_frames: int = None):
        """Сохраняет анализ сцен в CSV с полуоткрытыми интервалами (end_frame не включительно)"""
        logger.info(f"[scene_analysis] Вызван save_analysis. Сцен: {len(self.scene_changes)}. Путь: {output_file}")
        if not self.scene_changes:
            logger.warning("[scene_analysis] Нет смен сцен, файл не будет создан.")
            return
        data = []
        prev_frame = 0
        for i, change_frame in enumerate(self.scene_changes):
            end_frame = change_frame - 1  # полуоткрытый интервал
            duration = (end_frame - prev_frame + 1) / fps
            data.append({
                'scene': i,
                'start_frame': prev_frame,
                'end_frame': end_frame,
                'duration_seconds': duration
            })
            prev_frame = change_frame
        # Добавляем последнюю сцену до конца видео
        if total_frames is not None:
            end_frame = total_frames - 1
        else:
            end_frame = 'end'
        data.append({
            'scene': len(self.scene_changes),
            'start_frame': prev_frame,
            'end_frame': end_frame,
            'duration_seconds': 'unknown'
        })
        df = pd.DataFrame(data)
        try:
            df.to_csv(output_file, index=False)
            logger.info(f"📊 Анализ сцен сохранен в {output_file}")
        except Exception as e:
            logger.error(f"[scene_analysis] Ошибка при сохранении файла: {e}")


class OptimizedVideoProcessor:
    """Оптимизированный процессор видео"""
    
    def __init__(self, model_path: str = "yolov8n-pose.pt"):
        self.model = YOLO(model_path)
        self.scene_detector = OptimizedSceneDetector()
        self.tracker = OptimizedBoundingBoxTracker()
        self.color_cache = {}  # Кэш цветов для ID
        
    def process_video(self, input_path: str, output_path: str):
        """Обрабатывает видео с детекцией, трекингом и сменой сцен"""
        cap = cv2.VideoCapture(input_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Настройка видеозаписи
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        logger.info(f"🎬 Начинаем обработку видео: {total_frames} кадров, {fps:.1f} FPS")
        
        frame_count = 0
        last_scene_change = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            
            # Детекция смены сцены
            scene_changed = self.scene_detector.detect_scene_change(frame, frame_count, fps)
            if scene_changed:
                self.tracker.reset()
                last_scene_change = frame_count
            
            # YOLO детекция (каждый кадр)
            results = self.model(frame, conf=0.6, iou=0.35, max_det=15, verbose=False)
            
            # Фильтрация детекций людей
            detections = []
            for result in results:
                boxes = result.boxes
                if boxes is not None:
                    for box in boxes:
                        if box.cls == 0:  # класс "person"
                            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                            conf = box.conf[0].cpu().numpy()
                            detections.append((x1, y1, x2, y2, conf))
            
            # Удаление дубликатов
            detections = self._filter_duplicates(detections)
            
            # Обновление трекера
            tracked_boxes = self.tracker.update(detections, frame_count)
            
            # Отрисовка результатов
            processed_frame = self._draw_results(frame, tracked_boxes, frame_count, fps)
            
            # Запись кадра
            out.write(processed_frame)
            
            # Прогресс
            if frame_count % 100 == 0:
                progress = (frame_count / total_frames) * 100
                logger.info(f"📈 Прогресс: {progress:.1f}% ({frame_count}/{total_frames})")
        
        # Завершение
        cap.release()
        out.release()
        
        # Сохранение анализа
        self.scene_detector.save_analysis(str(Path("output") / "scene_analysis_optimized.csv"), fps)
        
        # Присоединение аудио
        self._attach_audio(input_path, output_path)
        
        logger.info("✅ Обработка завершена!")
    
    def _filter_duplicates(self, detections: List[Tuple]) -> List[Tuple]:
        """Фильтрует дублирующиеся детекции"""
        if len(detections) <= 1:
            return detections
        
        filtered = []
        for i, det1 in enumerate(detections):
            is_duplicate = False
            for j, det2 in enumerate(detections):
                if i != j:
                    iou = self._calculate_iou(det1[:4], det2[:4])
                    distance = self._calculate_distance(det1[:4], det2[:4])
                    
                    # Более строгая фильтрация: IoU > 0.3 или близкое расстояние
                    if iou > 0.3 or (iou > 0.1 and distance < 30):
                        # Оставляем детекцию с большей уверенностью
                        if det1[4] < det2[4]:
                            is_duplicate = True
                            logger.debug(f"🚫 Удален дубликат: IoU={iou:.2f}, расстояние={distance:.1f}")
                            break
            if not is_duplicate:
                filtered.append(det1)
        
        return filtered
    
    def _get_color_for_id(self, obj_id: int) -> Tuple[int, int, int]:
        """Генерирует уникальный цвет для ID объекта"""
        if obj_id not in self.color_cache:
            # Генерируем цвет на основе ID
            hue = (obj_id * 137.5) % 360  # Золотое сечение для равномерного распределения
            # Конвертируем HSV в BGR
            h = hue / 60
            i = int(h)
            f = h - i
            p = 0
            q = 255 * (1 - f)
            t = 255 * f
            
            if i == 0:
                r, g, b = 255, t, p
            elif i == 1:
                r, g, b = q, 255, p
            elif i == 2:
                r, g, b = p, 255, t
            elif i == 3:
                r, g, b = p, q, 255
            elif i == 4:
                r, g, b = t, p, 255
            else:
                r, g, b = 255, p, q
            
            # Увеличиваем яркость для лучшей видимости
            brightness = 0.8
            r = min(255, int(r * brightness))
            g = min(255, int(g * brightness))
            b = min(255, int(b * brightness))
            
            self.color_cache[obj_id] = (b, g, r)  # BGR формат для OpenCV
        
        return self.color_cache[obj_id]
    
    def _draw_results(self, frame: np.ndarray, tracked_boxes: List[Tuple], 
                     frame_num: int, fps: float, shot_id: int) -> np.ndarray:
        """Отрисовывает результаты на кадре"""
        result_frame = frame.copy()
        
        # Отладочная информация
        if frame_num % 30 == 0:  # Каждые 30 кадров
            logger.info(f"🎨 Отрисовка: {len(tracked_boxes)} объектов в кадре {frame_num}")
        
        # Отрисовка рамок с разноцветными ID
        for box in tracked_boxes:
            x1, y1, x2, y2, conf, obj_id = box
            color = self._get_color_for_id(obj_id)
            
            # Рамка
            cv2.rectangle(result_frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 3)
            
            # Текст ID ВНУТРИ рамки (левый верхний угол)
            text = f"ID: {obj_id}"
            (text_width, text_height), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
            text_x = int(x1) + 5
            text_y = int(y1) + text_height + 5
            # Фон для читаемости
            cv2.rectangle(result_frame, (text_x - 2, text_y - text_height - 2), (text_x + text_width + 2, text_y + 2), color, -1)
            # Текст
            cv2.putText(result_frame, text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # Показываем уверенность (confidence) внутри рамки, в левом нижнем углу
            conf_text = f"{conf:.2f}"
            (conf_width, conf_height), _ = cv2.getTextSize(conf_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            conf_bg_x1 = int(x1)
            conf_bg_y1 = int(y2) - conf_height - 8
            conf_bg_x2 = int(x1) + conf_width + 10
            conf_bg_y2 = int(y2)
            # Фон для читаемости
            cv2.rectangle(result_frame, (conf_bg_x1, conf_bg_y1), (conf_bg_x2, conf_bg_y2), color, -1)
            # Текст confidence
            cv2.putText(result_frame, conf_text, (conf_bg_x1 + 5, conf_bg_y2 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Отображение номера текущего шота
        cv2.putText(result_frame, f"Shot {shot_id}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 3)
        cv2.putText(result_frame, f"Shot {shot_id}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 1)
        
        return result_frame
    
    def _calculate_distance(self, box1: Tuple, box2: Tuple) -> float:
        """Вычисляет расстояние между центрами рамок"""
        center1 = ((box1[0] + box1[2]) / 2, (box1[1] + box1[3]) / 2)
        center2 = ((box2[0] + box2[2]) / 2, (box2[1] + box2[3]) / 2)
        return np.sqrt((center1[0] - center2[0])**2 + (center1[1] - center2[1])**2)
    
    def _calculate_iou(self, box1: Tuple, box2: Tuple) -> float:
        """Вычисляет IoU между двумя рамками"""
        x1, y1, x2, y2 = box1
        x3, y3, x4, y4 = box2
        
        x_left = max(x1, x3)
        y_top = max(y1, y3)
        x_right = min(x2, x4)
        y_bottom = min(y2, y4)
        
        if x_right < x_left or y_bottom < y_top:
            return 0.0
        
        intersection = (x_right - x_left) * (y_bottom - y_top)
        area1 = (x2 - x1) * (y2 - y1)
        area2 = (x4 - x3) * (y4 - y3)
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0.0
    
    def _attach_audio(self, input_path: str, output_path: str):
        """Присоединяет аудио к обработанному видео (заглушка без moviepy)"""
        logger.info("📹 Видео сохранено без аудио (moviepy не установлен)")


class ShotsVideoProcessor(OptimizedVideoProcessor):
    """Расширенный процессор для обработки видео с разбивкой на шоты и сохранением фото"""
    
    def __init__(self, model_path: str = "yolov8n.pt"):
        super().__init__(model_path)
        self.output_root = Path("output")
        self.shots_dir = self.output_root / "shorts"
        self.shaped_shots_dir = self.output_root / "shaped_shorts"
        self.photos_dir = self.output_root / "pictures"
        self.scenes_dir = self.output_root / "scenes"
        self.shots_dir.mkdir(parents=True, exist_ok=True)
        self.shaped_shots_dir.mkdir(parents=True, exist_ok=True)
        self.photos_dir.mkdir(parents=True, exist_ok=True)
        self.scenes_dir.mkdir(parents=True, exist_ok=True)
        
        # Для сохранения фото каждые 15 кадров (0.5 сек при 30fps)
        self.photo_interval_frames = 15  # кадры
        self.last_photo_frame = {}  # {obj_id: last_frame}
        
    def process_video_with_shots(self, input_path: str, output_path: str):
        logger.info(f"🎬 Начинаем обработку с разбивкой на шоты: {input_path}")
        scene_csv_path = "output/scene_analysis_optimized.csv"
        # Если файл с анализом сцен отсутствует — создаём его автоматически
        if not os.path.exists(scene_csv_path):
            logger.info("[scene_analysis] Файл не найден, запускаем анализ сцен...")
            self._detect_and_save_scenes(input_path, scene_csv_path)
        scenes_df = pd.read_csv(scene_csv_path)
        logger.info(f"📊 Найдено {len(scenes_df)} сцен для разбивки на шоты")
        
        # Создаем общий CSV файл для всех людей
        csv_data = []
        
        # Обрабатываем каждую сцену отдельно
        for shot_id, (_, scene) in enumerate(scenes_df.iterrows()):
            start_frame = int(scene['start_frame'])
            end_frame_str = str(scene['end_frame']).strip()
            # Пропускаем последнюю сцену с "end"
            if end_frame_str == "end":
                continue
            end_frame = int(end_frame_str)
            logger.info(f"🎬 Обрабатываем шот {shot_id}: кадры {start_frame}-{end_frame}")
            # Обрабатываем шот: сохраняем видео, трекинг и фото в одном проходе
            shot_csv_data = self._process_single_shot(input_path, start_frame, end_frame, shot_id)
            csv_data.extend(shot_csv_data)
        
        # Сохраняем общий CSV файл
        if csv_data:
            df = pd.DataFrame(csv_data)
            df.to_csv(str(self.shaped_shots_dir / "people_tracking_results.csv"), index=False)
            logger.info(f"📊 Сохранен CSV файл с {len(csv_data)} записями в {self.shaped_shots_dir}")

        # Сохраняем анализ сцен
        cap = cv2.VideoCapture(input_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        cap.release()
        # Перед сохранением анализа сцен удаляем старый файл, если он есть
        scene_csv_path = str(Path("output") / "scene_analysis_optimized.csv")
        if os.path.exists(scene_csv_path):
            os.remove(scene_csv_path)
        self.scene_detector.save_analysis(scene_csv_path, fps)

        logger.info("✅ Обработка с шотами завершена!")
    
    def _detect_and_save_scenes(self, input_path: str, output_csv: str):
        cap = cv2.VideoCapture(input_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = 0
        self.scene_detector = OptimizedSceneDetector()
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame_count += 1
            self.scene_detector.detect_scene_change(frame, frame_count, fps)
        cap.release()
        self.scene_detector.save_analysis(output_csv, fps, total_frames=total_frames)
        logger.info(f"[scene_analysis] Анализ сцен завершён и сохранён в {output_csv}")
    
    def _process_single_shot(self, input_path: str, start_frame: int, end_frame: int, shot_id: int):
        """Обрабатывает один шот: видео + трекинг + фото в одном проходе"""
        logger.info(f"🎬 Обрабатываем шот {shot_id}: кадры {start_frame}-{end_frame}")
        
        cap = cv2.VideoCapture(input_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        # Получаем размеры кадра
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        ret, frame = cap.read()
        if not ret:
            raise ValueError(f"Не удалось прочитать кадр {start_frame}")
        
        height, width = frame.shape[:2]
        
        # Создаем видеописатель для шота (без разметки)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        shot_path = self.shots_dir / f"shot_{shot_id:03d}.mp4"
        out = cv2.VideoWriter(str(shot_path), fourcc, fps, (width, height))
        # Создаем видеописатель для шота с разметкой
        shaped_shot_path = self.shaped_shots_dir / f"shot_{shot_id:03d}.mp4"
        out_shaped = cv2.VideoWriter(str(shaped_shot_path), fourcc, fps, (width, height))
        
        # Сбрасываем трекер для нового шота
        self.tracker.reset()
        self.last_photo_frame = {}
        
        # Очищаем все атрибуты объектов, которые могли остаться
        for obj in self.tracker.objects.values():
            if hasattr(obj, 'first_photo_saved'):
                delattr(obj, 'first_photo_saved')
        
        # Данные для CSV
        csv_data = []
        
        # Переходим к начальному кадру
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        
        # Обрабатываем кадры шота
        for frame_num in range(start_frame, end_frame + 1):
            ret, frame = cap.read()
            if not ret:
                break
            
            # Детекция людей
            results = self.model(frame, verbose=False)
            detections = []
            
            for result in results:
                boxes = result.boxes
                if boxes is not None:
                    for box in boxes:
                        if box.cls == 0:  # person class
                            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                            conf = box.conf[0].cpu().numpy()
                            if conf > 0.5:
                                detections.append((x1, y1, x2, y2, conf))
            
            # Фильтруем дубликаты
            detections = self._filter_duplicates(detections)
            
            # Обновляем трекер
            tracked_boxes = self.tracker.update(detections, frame_num)
            
            # Сохраняем фото если нужно
            self._save_photos_if_needed(frame, tracked_boxes, shot_id, frame_num, self.tracker)
            
            # Отрисовываем результаты
            result_frame = self._draw_results(frame, tracked_boxes, frame_num, fps, shot_id)
            
            # Записываем кадр в видео шота (без разметки)
            out.write(frame)
            # Записываем кадр в видео шота с разметкой
            out_shaped.write(result_frame)
            
            # Добавляем данные в CSV
            for box in tracked_boxes:
                x1, y1, x2, y2, conf, obj_id = box
                csv_data.append({
                    'shot_id': shot_id,
                    'frame': frame_num,
                    'object_id': obj_id,
                    'x1': x1,
                    'y1': y1,
                    'x2': x2,
                    'y2': y2,
                    'confidence': conf
                })
        
        out.release()
        out_shaped.release()
        cap.release()
        
        return csv_data
    
    def _save_photos_if_needed(self, frame: np.ndarray, tracked_boxes: List[Tuple], 
                              shot_id: int, frame_count: int, shot_tracker: OptimizedBoundingBoxTracker):
        """Сохраняет фото людей только если объект есть в tracked_boxes (рамка есть на кадре). Первая фотка совпадает с появлением рамки. Если рамка исчезла — сбрасываем таймер."""
        logger.info(f"🔍 Шот {shot_id}, кадр {frame_count}: tracked_boxes={len(tracked_boxes)}")
        # Сохраняем список активных obj_id на этом кадре
        current_ids = set()
        for box in tracked_boxes:
            x1, y1, x2, y2, conf, obj_id = box
            current_ids.add(obj_id)
            logger.info(f"🔍 Шот {shot_id}, кадр {frame_count}: обрабатываем объект {obj_id} в позиции ({x1:.1f}, {y1:.1f})")
            
            # Сохраняем фото только если объект есть в tracked_boxes (рамка реально есть)
            # Проверяем, нужно ли сохранить фото
            just_appeared = obj_id not in self.last_photo_frame
            should_save = False
            if just_appeared:
                should_save = True
                self.last_photo_frame[obj_id] = frame_count
                logger.info(f"🔍 Шот {shot_id}, объект {obj_id}: первый раз, should_save=True")
            else:
                # Каждые 15 кадров (0.5 сек)
                if frame_count - self.last_photo_frame[obj_id] >= self.photo_interval_frames:
                    should_save = True
                    self.last_photo_frame[obj_id] = frame_count
                    logger.info(f"🔍 Шот {shot_id}, объект {obj_id}: не первый раз, should_save=True")
                else:
                    logger.info(f"🔍 Шот {shot_id}, объект {obj_id}: не первый раз, should_save=False")
            
            if should_save:
                # Вырезаем изображение человека
                x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
                person_img = frame[y1:y2, x1:x2]
                if person_img.size > 0:
                    # Создаем папку шота и подпапку ID
                    shot_dir = self.photos_dir / f"shot_{shot_id:03d}"
                    shot_dir.mkdir(exist_ok=True)
                    id_dir = shot_dir / f"ID_{obj_id:03d}"
                    id_dir.mkdir(exist_ok=True)
                    img_path = id_dir / f"shot_{shot_id:03d}_frame_{frame_count:06d}.jpg"
                    cv2.imwrite(str(img_path), person_img)
                    
                    if just_appeared:
                        logger.info(f"📸 Первое фото при появлении рамки (кадр {frame_count}): {img_path}")
                    else:
                        logger.info(f"📸 Сохранено фото каждые 15 кадров (кадр {frame_count}): {img_path}")
        
        # Сброс last_photo_frame для исчезнувших объектов
        ids_to_remove = [obj_id for obj_id in self.last_photo_frame if obj_id not in current_ids]
        for obj_id in ids_to_remove:
            del self.last_photo_frame[obj_id]


def clear_output_dirs():
    """Очищает папку output и все вложенные выходные папки перед началом работы"""
    output_root = Path("output")
    if output_root.exists():
        shutil.rmtree(output_root)
    (output_root / "shorts").mkdir(parents=True, exist_ok=True)
    (output_root / "shaped_shorts").mkdir(parents=True, exist_ok=True)
    (output_root / "pictures").mkdir(parents=True, exist_ok=True)
    (output_root / "scenes").mkdir(parents=True, exist_ok=True)


def main():
    """Главная функция"""
    clear_output_dirs()
    # Получаем имя входного файла из аргументов командной строки, по умолчанию 'in1.mp4'
    input_video = sys.argv[1] if len(sys.argv) > 1 else "mister-i-missis-smit-2005_1.mkv"
    processor = ShotsVideoProcessor()
    processor.process_video_with_shots(input_video, "output_video_optimized.mp4")


if __name__ == "__main__":
    main() 