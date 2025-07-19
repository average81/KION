from ultralytics import YOLO
import cv2
import numpy as np
from moviepy import VideoFileClip, AudioFileClip
import os
from collections import deque

class BoundingBoxTracker:
    """Класс для сглаживания координат и интерполяции при пропаданиях"""
    
    def __init__(self, max_history=10, min_confidence=0.5, interpolation_frames=5, iou_threshold=0.3):
        self.max_history = max_history  # количество кадров для сглаживания
        self.min_confidence = min_confidence  # минимальная уверенность
        self.interpolation_frames = interpolation_frames  # кадры для интерполяции
        self.iou_threshold = iou_threshold  # порог IoU для объединения объектов
        
        # Состояния объектов
        self.objects = {}  # {object_id: ObjectState}
        self.current_frame = 0
        self.next_object_id = 0
        
    def calculate_iou(self, box1, box2):
        """Вычисляет IoU между двумя рамками"""
        x1_1, y1_1, x2_1, y2_1 = box1
        x1_2, y1_2, x2_2, y2_2 = box2
        
        # Вычисляем пересечение
        x1_i = max(x1_1, x1_2)
        y1_i = max(y1_1, y1_2)
        x2_i = min(x2_1, x2_2)
        y2_i = min(y2_1, y2_2)
        
        if x2_i <= x1_i or y2_i <= y1_i:
            return 0.0
        
        intersection = (x2_i - x1_i) * (y2_i - y1_i)
        
        # Вычисляем объединение
        area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
        area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0.0
    
    def find_best_match(self, new_box, class_id):
        """Находит лучший совпадающий объект для нового обнаружения"""
        best_match_id = None
        best_iou = 0.0
        
        for object_id, obj in self.objects.items():
            if obj.class_id == class_id:  # только объекты того же класса
                # Получаем последнюю позицию объекта
                if len(obj.history) > 0:
                    last_box = obj.history[-1][:4]  # x1, y1, x2, y2
                    iou = self.calculate_iou(new_box, last_box)
                    
                    if iou > best_iou and iou > self.iou_threshold:
                        best_iou = iou
                        best_match_id = object_id
        
        return best_match_id, best_iou
        
    class ObjectState:
        """Состояние отдельного объекта"""
        def __init__(self, object_id, class_id, box, confidence, frame_count, max_history, interpolation_frames):
            self.object_id = object_id
            self.class_id = class_id
            self.history = deque(maxlen=max_history)  # история рамок для сглаживания
            self.last_seen = frame_count
            self.interpolation_frames = interpolation_frames
            self.is_interpolating = False
            self.interpolation_start = None
            self.interpolation_end = None
            self.interpolation_start_frame = None
            
            # Добавляем первое обнаружение
            self.history.append((box[0], box[1], box[2], box[3], confidence, frame_count))
            
        def update(self, box, confidence, frame_count):
            """Обновляет состояние объекта при новом обнаружении"""
            self.history.append((box[0], box[1], box[2], box[3], confidence, frame_count))
            self.last_seen = frame_count
            
            # Если объект был в интерполяции, прекращаем её
            if self.is_interpolating:
                self.is_interpolating = False
                self.interpolation_start = None
                self.interpolation_end = None
                self.interpolation_start_frame = None
                
        def check_interpolation(self, current_frame):
            """Проверяет, нужно ли начать интерполяцию"""
            frames_since_seen = current_frame - self.last_seen
            
            if frames_since_seen > 0 and not self.is_interpolating:
                # Начинаем интерполяцию
                self.is_interpolating = True
                self.interpolation_start_frame = current_frame
                
                # Берем последние рамки для интерполяции
                if len(self.history) >= 2:
                    recent_boxes = list(self.history)[-2:]
                    self.interpolation_start = recent_boxes[-1][:4]  # x1, y1, x2, y2
                    self.interpolation_end = recent_boxes[-2][:4]   # предыдущая рамка
                else:
                    # Если нет истории, используем последнюю рамку
                    last_box = self.history[-1][:4]
                    self.interpolation_start = last_box
                    self.interpolation_end = last_box
                    
            elif self.is_interpolating:
                # Проверяем, не истекло ли время интерполяции
                if frames_since_seen > self.interpolation_frames:
                    self.is_interpolating = False
                    return False
                    
            return True
            
        def get_smoothed_box(self):
            """Возвращает сглаженную рамку"""
            if len(self.history) == 0:
                return None, 0.0
                
            if len(self.history) == 1:
                x1, y1, x2, y2, conf, _ = self.history[0]
                return (x1, y1, x2, y2), conf
                
            # Берем последние рамки для сглаживания
            recent_boxes = list(self.history)[-min(len(self.history), 5):]
            
            # Вычисляем среднюю рамку (сглаживание)
            avg_x1 = int(np.mean([box[0] for box in recent_boxes]))
            avg_y1 = int(np.mean([box[1] for box in recent_boxes]))
            avg_x2 = int(np.mean([box[2] for box in recent_boxes]))
            avg_y2 = int(np.mean([box[3] for box in recent_boxes]))
            avg_conf = np.mean([box[4] for box in recent_boxes])
            
            return (avg_x1, avg_y1, avg_x2, avg_y2), avg_conf
            
        def get_interpolated_box(self, current_frame):
            """Возвращает интерполированную рамку"""
            if not self.is_interpolating or self.interpolation_start is None:
                return None, 0.0
                
            frames_in_interpolation = current_frame - self.interpolation_start_frame
            progress = min(1.0, frames_in_interpolation / self.interpolation_frames)
            
            # Линейная интерполяция между начальной и конечной рамкой
            start_x1, start_y1, start_x2, start_y2 = self.interpolation_start
            end_x1, end_y1, end_x2, end_y2 = self.interpolation_end
            
            interp_x1 = int(start_x1 + (end_x1 - start_x1) * progress)
            interp_y1 = int(start_y1 + (end_y1 - start_y1) * progress)
            interp_x2 = int(start_x2 + (end_x2 - start_x2) * progress)
            interp_y2 = int(start_y2 + (end_y2 - start_y2) * progress)
            
            # Уменьшаем уверенность во время интерполяции
            base_conf = 0.7 if len(self.history) > 0 else 0.5
            interp_conf = base_conf * (1.0 - progress * 0.5)  # плавно уменьшаем до 50%
            
            return (interp_x1, interp_y1, interp_x2, interp_y2), interp_conf
    
    def add_detection(self, class_id, box, confidence):
        """Добавляет новое обнаружение"""
        if confidence < self.min_confidence:
            return
            
        # Ищем лучший совпадающий объект
        best_match_id, best_iou = self.find_best_match(box, class_id)
        
        if best_match_id is not None:
            # Обновляем существующий объект
            self.objects[best_match_id].update(box, confidence, self.current_frame)
        else:
            # Создаем новый объект
            object_id = self.next_object_id
            self.next_object_id += 1
            
            self.objects[object_id] = self.ObjectState(
                object_id, class_id, box, confidence, self.current_frame, 
                self.max_history, self.interpolation_frames
            )
    
    def get_boxes(self):
        """Возвращает все рамки (сглаженные или интерполированные)"""
        active_boxes = []
        expired_objects = []
        
        for object_id, obj in self.objects.items():
            # Проверяем интерполяцию
            if obj.check_interpolation(self.current_frame):
                if obj.is_interpolating:
                    # Получаем интерполированную рамку
                    box, conf = obj.get_interpolated_box(self.current_frame)
                    if box is not None:
                        active_boxes.append((obj.class_id, box, conf, "interpolated", object_id))
                else:
                    # Получаем сглаженную рамку
                    box, conf = obj.get_smoothed_box()
                    if box is not None:
                        active_boxes.append((obj.class_id, box, conf, "smoothed", object_id))
            else:
                expired_objects.append(object_id)
        
        # Удаляем истекшие объекты
        for object_id in expired_objects:
            del self.objects[object_id]
            
        return active_boxes
    
    def update_frame_count(self, frame_count):
        """Обновляет номер текущего кадра"""
        self.current_frame = frame_count
    
    def get_stats(self):
        """Возвращает статистику по объектам"""
        active_count = sum(1 for obj in self.objects.values() if not obj.is_interpolating)
        interpolating_count = sum(1 for obj in self.objects.values() if obj.is_interpolating)
        
        return {
            "total": len(self.objects),
            "active": active_count,
            "interpolating": interpolating_count
        }

if __name__ == "__main__":
    # Загрузка модели YOLO v11 (если доступна)
    print("🚀 Загружаем YOLO модель...")

    try:
        # Пытаемся загрузить YOLO v11
        model = YOLO('yolo11n.pt')  # исправлено имя модели
        print("✅ YOLO v11 активирована!")
        print("   📊 Модель: yolo11n.pt (nano)")
        print("   ⚡ Преимущества: быстрее и точнее чем v8")

        # Проверяем какая модель работает в момент выполнения
        print(f"🔍 Проверка модели: {model.ckpt_path}")

    except Exception as e:
        print(f"❌ YOLO v11 недоступна: {e}")
        try:
            # Fallback на YOLO v8
            model = YOLO('yolov8n.pt')
            print("⚠️ YOLO v8 активирована (fallback)")
            print("   📊 Модель: yolov8n.pt (nano)")
            print("   📦 Стабильная версия")

            # Проверяем какая модель работает в момент выполнения
            print(f"🔍 Проверка модели: {model.ckpt_path}")

        except Exception as e2:
            print(f"❌ YOLO v8 тоже недоступна: {e2}")
            print("💡 Убедитесь, что модели скачаны:")
            print("   - yolo11n.pt для YOLO v11")
            print("   - yolov8n.pt для YOLO v8")
            exit(1)

    # Список цветов для различных классов
    colors = [
        (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (0, 255, 255),
        (255, 0, 255), (192, 192, 192), (128, 128, 128), (128, 0, 0), (128, 128, 0),
        (0, 128, 0), (128, 0, 128), (0, 128, 128), (0, 0, 128), (72, 61, 139),
        (47, 79, 79), (47, 79, 47), (0, 206, 209), (148, 0, 211), (255, 20, 147)
    ]

    # Создаем трекер для сглаживания координат и интерполяции с уникализацией объектов
    tracker = BoundingBoxTracker(max_history=10, min_confidence=0.3, interpolation_frames=5, iou_threshold=0.3)

    # Открытие исходного видеофайла
    input_video_path = 'in2.mp4'
    capture = cv2.VideoCapture(input_video_path)

    if not capture.isOpened():
        print(f"❌ Не удалось открыть видео: {input_video_path}")
        exit(1)

    # Чтение параметров видео
    fps = int(capture.get(cv2.CAP_PROP_FPS))
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print(f"📹 Параметры видео: {width}x{height}, {fps} FPS")

    # Настройка выходного файла - РАЗНЫЕ ИМЕНА!
    temp_video_path = 'temp_detect_stabilized.mp4'  # временный файл без звука
    final_video_path = 'detect_stabilized.mp4'      # финальный файл со звуком

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(temp_video_path, fourcc, fps, (width, height))

    print("🎬 Обработка видео со стабилизацией рамок...")
    frame_count = 0

    while True:
        # Захват кадра
        ret, frame = capture.read()
        if not ret:
            break

        # Обработка кадра с помощью модели YOLO
        results = model(frame)[0]

        # Получение данных об объектах
        classes_names = results.names
        classes = results.boxes.cls.cpu().numpy()
        boxes = results.boxes.xyxy.cpu().numpy().astype(np.int32)
        confidences = results.boxes.conf.cpu().numpy()

        # Добавляем обнаружения в трекер
        for class_id, box, conf in zip(classes, boxes, confidences):
            tracker.add_detection(int(class_id), box, conf)

        # Получаем стабилизированные рамки
        stabilized_boxes = tracker.get_boxes()

        # Рисование сглаженных и интерполированных рамок
        for class_id, box, conf, state, object_id in stabilized_boxes:
            if conf > 0.3:  # порог для отображения рамок
                class_name = classes_names[class_id]
                color = colors[object_id % len(colors)]  # используем object_id для цвета
                x1, y1, x2, y2 = box

                # Рисуем рамку
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

                # Рисуем подпись с информацией о состоянии и ID объекта
                label = f"{class_name} #{object_id} {conf:.2f} ({state})"
                cv2.putText(frame, label, (x1, y1 - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # Очищаем старые обнаружения
        tracker.update_frame_count(frame_count)

        # Запись обработанного кадра в выходной файл
        writer.write(frame)

        frame_count += 1
        if frame_count % 30 == 0:
            stats = tracker.get_stats()
            print(f"📊 Обработано кадров: {frame_count}")
            print(f"   🔍 Активных объектов: {stats['active']}")
            print(f"   🔄 Интерполируемых: {stats['interpolating']}")
            print(f"   📦 Всего треков: {stats['total']}")

    # Освобождение ресурсов
    capture.release()
    writer.release()

    print(f"✅ Временное видео со стабилизацией сохранено: {temp_video_path}")

    print("🔊 Добавление звука...")

    # Добавление звука к обработанному видео
    try:
        # Загружаем исходное видео для извлечения звука
        original_video = VideoFileClip(input_video_path)

        # Загружаем обработанное видео (без звука)
        processed_video = VideoFileClip(temp_video_path)

        # Добавляем звук из исходного видео
        final_video = processed_video.set_audio(original_video.audio)

        # Сохраняем финальное видео со звуком в ДРУГОЙ файл
        final_video.write_videofile(final_video_path,
                                   codec='libx264',
                                   audio_codec='aac',
                                   temp_audiofile='temp-audio.m4a',
                                   remove_temp=True)

        # Закрываем видеофайлы
        original_video.close()
        processed_video.close()
        final_video.close()

        # Удаляем временный файл без звука
        if os.path.exists(temp_video_path):
            os.remove(temp_video_path)
            print(f"🗑️ Удален временный файл: {temp_video_path}")

        print(f"🎉 Финальное видео со стабилизацией и звуком сохранено: {final_video_path}")

        # Проверяем, что файл действительно создан
        if os.path.exists(final_video_path):
            file_size = os.path.getsize(final_video_path) / (1024 * 1024)  # размер в МБ
            print(f"📁 Размер файла: {file_size:.1f} МБ")
        else:
            print("❌ Финальный файл не найден!")

    except Exception as e:
        print(f"❌ Ошибка при добавлении звука: {e}")
        print(f"💾 Видео без звука сохранено: {temp_video_path}")

        # Если что-то пошло не так, переименовываем временный файл
        if os.path.exists(temp_video_path):
            os.rename(temp_video_path, final_video_path)
            print(f"🔄 Временный файл переименован в: {final_video_path}")