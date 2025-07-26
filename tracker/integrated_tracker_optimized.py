import cv2
import tqdm
import numpy as np
from ultralytics import YOLO
import pandas as pd
from moviepy import VideoFileClip, AudioFileClip
import os
from collections import deque, defaultdict
from typing import List, Tuple, Optional, Dict, Any
import logging
from face_recognition import FaceRecognition
from .base_tracker import BaseVideoProcessor

# Аудио обработка
import librosa
import librosa.display

# Настройка логирования
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

script_dir = os.path.dirname(os.path.abspath(__file__))

class OptimizedBoundingBoxTracker():
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
        
    def update(self, detections: List[List], frame_num: int) -> List[List]:
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
    
    def _match_detections(self, detections: List[List]) -> Tuple[List[Tuple], List[List]]:
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
                    iou = self._calculate_iou(tuple(detection[:4]), tuple(last_box))
                    
                    # Вычисляем расстояние между центрами
                    last_center_x = (last_box[0] + last_box[2]) / 2
                    last_center_y = (last_box[1] + last_box[3]) / 2
                    distance = np.sqrt((new_center_x - last_center_x)**2 + (new_center_y - last_center_y)**2)
                    
                    # Условия сопоставления как в оригинале
                    if (iou > best_iou and iou > self.iou_threshold and distance < 100):
                        best_iou = iou
                        best_match_id = obj_id
                        logger.debug(f"🔍 Найдено совпадение: объект {obj_id}, IoU={iou:.3f}, расстояние={distance:.1f}")
            
            if best_match_id is not None:
                matched.append((detection, best_match_id))
            else:
                unmatched.append(detection)
        
        return matched, unmatched
    
    def _create_new_object(self, detection: List, frame_num: int):
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
    
    def _get_all_boxes(self) -> List[List]:
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
                    active_boxes.append([*box, conf, obj.object_id])
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
    """Оптимизированный класс для отслеживания отдельного объекта"""
    
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

    def final_scene(self, frame_num: int):
        self.scene_changes.append(frame_num)

        
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
    
    def save_analysis(self, output_file: str, fps: float):
        """Сохраняет анализ сцен в CSV"""
        if not self.scene_changes:
            return
        
        data = []
        prev_frame = 0
        
        for i, change_frame in enumerate(self.scene_changes):
            duration = (change_frame - prev_frame) / fps
            data.append({
                'scene': i,
                'start_frame': prev_frame,
                'start_time': prev_frame / fps,
                'end_frame': change_frame,
                'end_time': change_frame / fps,
                'duration_seconds': duration
            })
            prev_frame = change_frame
        

        
        df = pd.DataFrame(data)
        df.to_csv(output_file, index=False)
        logger.info(f"📊 Анализ шотов сохранен в {output_file}")


class OptimizedVideoProcessor(BaseVideoProcessor):
    """Оптимизированный процессор видео"""
    
    def __init__(self, model_path: str = script_dir + "/yolov8n.pt", detector = 'mmod', force_update = False):
        # Инициализация базового класса
        super().__init__(model_path, detector, force_update)
        self.scene_detector = OptimizedSceneDetector(threshold = 15)
        self.tracker = OptimizedBoundingBoxTracker()

    def process_video(self, video_path: str, output_path: str):
        self.video_path = video_path
        self.output_path = output_path if output_path else ""
        if not os.path.exists(self.video_path):
            logger.error(f"🚨 Видео не найдено: {self.video_path}")
            return

        """Обрабатывает видео с детекцией, трекингом и сменой сцен"""
        self.clip = VideoFileClip(self.video_path)

        length = self.clip.duration
        width = self.clip.size[0]
        height = self.clip.size[1]
        total_frames = self.clip.n_frames
        fps = total_frames/length

        
        # Настройка видеозаписи
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        shorts_path = output_path + '/shorts'
        if not os.path.exists(shorts_path):
            os.makedirs(shorts_path)
        prefix = 'input_short_'

        logger.info(f"🎬 Начинаем обработку видео: {total_frames} кадров, {fps:.1f} FPS")
        video_scenes = []   #Начало и конец сцен в секундах
        scene_start_frame = 0
        frame_count = 0
        self.names = {}
        shapes_df = pd.DataFrame(columns=['id', 'frame', 'shape', 'face_desc'])
        #Обработка субтитров
        subtitles_path = video_path[:-4] + '.srt'
        subtitles_scenes = self.analyze_subtitles(subtitles_path, video_path)

        while frame_count < total_frames:
            frame = self.clip.get_frame(frame_count/self.clip.fps)

            frame_count += 1
            
            # Детекция смены сцены
            scene_changed = self.scene_detector.detect_scene_change(frame, frame_count, fps)
            if scene_changed:
                self.names = {}  # Сброс имен после смены сцены
                self.tracker.reset()
                scene = (scene_start_frame/fps, (frame_count-1)/fps)
                video_scenes.append(scene)
                short_path = os.path.join(shorts_path, f"{prefix}{len(video_scenes)}.mp4")
                clip = VideoFileClip(video_path).subclipped(scene[0] + 1/2/fps, scene[1])
                clip.write_videofile(short_path, codec='libx264', audio_codec='aac')
                clip.close()
                logger.info(f"Сохранена сцена {len(video_scenes)} в {short_path}")
                scene_start_frame = frame_count
                self.shapes_list.append(shapes_df)
                shapes_df = pd.DataFrame(columns=['id', 'frame', 'shape', 'face_desc'])
                # Присоединение аудио
                #self._attach_audio(f"{prefix}{len(video_scenes) + 1}tmp.mp4", f"{prefix}{len(video_scenes) + 1}.mp4")
                #out.release()
                #output_path = os.path.join(shorts_path, f"{prefix}{len(video_scenes) + 1}tmp.mp4")
                #out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
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
                            detections.append([x1, y1, x2, y2, conf])
            
            # Удаление дубликатов
            detections = self._filter_duplicates(detections)
            # Обновление трекера
            tracked_boxes = self.tracker.update(detections, frame_count)
            # Обработка лиц
            if tracked_boxes:
                for box in tracked_boxes:
                    x1, y1, x2, y2 = box[:4]
                    x1 = int(x1)
                    y1 = int(y1)
                    x2 = int(x2)
                    y2 = int(y2)
                    obj_id = box[5]

                    if obj_id not in self.names:
                        #выбираем изображение и меняем BGR на RGB
                        human_img = np.array(frame[y1:y2, x1:x2, ::-1], dtype=np.uint8)
                        face = self.face_recognition.detector.shape_of_image(human_img)
                        if face is None:
                            description = None
                        else:
                            description = self.face_recognition.facerec.compute_face_descriptor(human_img, face)
                        box.append(face)
                        if description is not None:
                            name = self.face_recognition.find_name_desc(description)
                            if name != "Unknown":
                                if len(self.face_recognition.local_dataset[self.face_recognition.local_dataset['name'] == name]) < 10:
                                    #print(len(self.face_recognition.local_dataset[self.face_recognition.local_dataset['name'] == name]))
                                    self.face_recognition.add_face_desc(description, name)
                                    logging.info(f"👤 Сохранено лицо для объекта {name}")
            # Сохраняем информацию о треках
            if len(tracked_boxes) > 0:
                for det in tracked_boxes:
                    x1 = int(det[0])
                    y1 = int(det[1])
                    x2 = int(det[2])
                    y2 = int(det[3])
                    track_id = det[5]
                    shapes_df.loc[len(shapes_df)] = [int(track_id),frame_count-scene_start_frame,(x1, y1, x2, y2),det[6]]
        self.scene_detector.final_scene(total_frames - 1)
        #Удаляем из локального датасета записи, с именами, которые опознаны менее, чем в 10 кадрах
        for name in self.face_recognition.local_dataset['name'].unique():
            if len(self.face_recognition.local_dataset[self.face_recognition.local_dataset['name'] == name]) < 10:
                self.face_recognition.local_dataset = self.face_recognition.local_dataset[self.face_recognition.local_dataset['name'] != name]
        logger.info(f"Обработка шотов, 2 проход...")
        if not os.path.exists(output_path + f'/shaped_shorts'):
            os.makedirs(output_path + f'/shaped_shorts')
        for i, (start, end) in enumerate(video_scenes):
            short_clip = self.clip.subclipped(start, end)
            self.video_short_tracker(short_clip, output_path + f'/shaped_shorts/input_debug_{i}.mp4', i)


        # Параметры для нашего случая
        video_weight = 0.7  # больший вес для видео сцен
        audio_weight = 0.3  # меньший вес для аудио сцен
        min_duration = 10.0  # минимальная длительность сцены в секундах
        max_duration = 600.0 # максимальная длительность сцены в секундах
        merged_scenes = None
        logger.info(f"\nОбнаружено {len(video_scenes)} сцен на основе видео анализа:")
        for i, (start, end) in enumerate(video_scenes):
            logger.info(f"Сцена {i+1}: {start:.2f} - {end:.2f} сек")

        if self.clip.audio is not None:
            self.clip.audio.write_audiofile(output_path + '/sound.wav', codec='pcm_s16le')
            audio_features = self.analyze_audio()
            audio_scenes = self.detect_audio_scenes(audio_features)
            logger.info(f"\nОбнаружено {len(audio_scenes)} сцен на основе аудио анализа:")
            for i, (start, end) in enumerate(audio_scenes):
                logger.info(f"Сцена {i+1}: {start:.2f} - {end:.2f} сек")
            # Объединяем сцены с новыми параметрами
            merged_scenes = self.merge_scenes(video_scenes, audio_scenes,
                                              subtitles_scenes,
                                              min_scene_duration=min_duration)
            logger.info(f"Обнаружено {len(merged_scenes)} сцен после улучшенного объединения:")
            for i, (start, end) in enumerate(merged_scenes):
                logger.info(f"Сцена {i+1}: {start:.2f} - {end:.2f} сек (длительность: {end-start:.2f} сек)")

        #Сохраняем сцены
        if merged_scenes:
            scene_path = output_path + '/scenes'
            prefix = 'input_scene_'
            self.save_scenes_as_videos(video_path, self.clip.fps, merged_scenes, scene_path,prefix)
        
        # Сохранение анализа
        self.scene_detector.save_analysis(output_path + "/shot_analysis.csv", fps)

        self.clip.close()

        data = []
        prev_frame = 0

        for i, scene in enumerate(merged_scenes):
            duration = (scene[1] - scene[0])
            change_frame = int(scene[1] * fps)
            data.append({
                'scene': i,
                'start_frame': prev_frame,
                'start_time': prev_frame / fps,
                'end_frame': change_frame,
                'end_time': change_frame / fps,
                'duration_seconds': duration
            })
            prev_frame = change_frame


        df = pd.DataFrame(data)
        scene_analysis_path = output_path + '/scene_analysis.csv'
        df.to_csv(scene_analysis_path, index=False)
        logger.info(f"📊 Анализ сцен сохранен в {scene_analysis_path}")
        
        logger.info("✅ Обработка завершена!")
    
    def _filter_duplicates(self, detections: List[List]) -> List[List]:
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
    
    def _attach_audio(self, input_path: str, output_path: str, scene):
        """Присоединяет аудио к обработанному видео"""
        try:
            # Временный файл без аудио
            temp_path = output_path.replace('.mp4', '_temp.mp4')
            os.rename(output_path, temp_path)
            
            # Присоединение аудио
            video = VideoFileClip(temp_path).subclipped(scene[0], scene[1])
            audio = VideoFileClip(input_path).audio
            
            if audio is not None:
                final_video = video.with_audio(audio)
                final_video.write_videofile(output_path, codec='libx264', audio_codec='aac')
                final_video.close()
                audio.close()
            else:
                video.write_videofile(output_path, codec='libx264', verbose=False)
                video.close()
            
            # Удаление временного файла
            os.remove(temp_path)
            logger.info("🎵 Аудио присоединено")
            
        except Exception as e:
            logger.error(f"❌ Ошибка при присоединении аудио: {e}")
            # Восстанавливаем файл без аудио
            if os.path.exists(temp_path):
                os.rename(temp_path, output_path)

def main():
    """Главная функция"""
    processor = OptimizedVideoProcessor()
    processor.process_video("C:/Users/above/IdeaProjects/video/Video_Samples/in2.mp4", "output_video_optimized.mp4")


if __name__ == "__main__":
    main() 