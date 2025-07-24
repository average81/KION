import os
import cv2
import pandas as pd
import tqdm
import numpy as np
from typing import List, Tuple, Optional, Dict, Any

# Видео обработка
from moviepy import VideoFileClip
import scenedetect
from scenedetect.detectors import ContentDetector
from scenedetect.scene_manager import SceneManager

# Аудио обработка
import librosa
import librosa.display
#from pydub import AudioSegment
#import speech_recognition as sr

# Обработка субтитров


# Детекция и трекинг объектов
from ultralytics import YOLO

from face_recognition import FaceRecognition

from .integrated_tracker_optimized import OptimizedBoundingBoxTracker

script_dir = os.path.dirname(os.path.abspath(__file__))

import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ComplexVideoProcessor:
    def __init__(self, model_path: str = script_dir + "/yolov8n.pt", detector = 'mmod'):
        self.model = YOLO(model_path)
        self.color_cache = {}  # Кэш цветов для ID
        self.face_recognition = FaceRecognition(detector,recognition_value = 0.4)
        self.face_recognition.load_dataset(tomemory = True)
        self.tracker = OptimizedBoundingBoxTracker()
        self.shapes_list = []   #список датафреймов по шотам с треками людей

    def video_short_pretracker(self, clip, clipnum):
        #Предварительный проход по видео для записи в локальную базу дескрипторов опознанных актеров именно из этого видео
        logger.info(f"🎬 Начинаем обработку шота {clipnum}: {clip.n_frames} кадров, {clip.fps:.1f} FPS")

        self.names = {}
        #window = dlib.image_window()
        shapes_df = pd.DataFrame(columns=['id', 'frame', 'shape', 'face_desc'])
        for frame_count in tqdm.tqdm(range(clip.n_frames)):
            frame = clip.get_frame(frame_count/clip.fps)

            # Выполняем трекинг с YOLOv8
            results = self.model.track(frame, stream=True, persist=True, tracker="botsort.yaml", verbose=False)

            # Фильтрация детекций людей
            detections = []

            for result in results:
                boxes = result.boxes
                if boxes is not None:
                    for box in boxes:
                        if box.cls == 0 and box.id != None:  # класс "person"
                            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                            conf = box.conf[0].cpu().numpy()
                            #print(result.boxes.id)
                            detections.append([x1, y1, x2, y2, conf,int(box.id)])

            # Обновление трекера
            #tracked_boxes = self.tracker.update(detections, frame_count)
            tracked_boxes = detections if len(detections) > 0 else []
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
                        human_img = np.array(frame[y1:y2, x1:x2], dtype=np.uint8)

                        #window.set_image(human_img)
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
                    shapes_df.loc[len(shapes_df)] = [int(track_id),frame_count,(x1, y1, x2, y2),det[6]]

        return shapes_df

    def video_short_tracker(self, clip, output_path: str, clipnum):
        """Обрабатывает видео с детекцией и трекомингом объектов"""

        logger.info(f"🎬 Начинаем обработку шота {clipnum}: {clip.n_frames} кадров, {clip.fps:.1f} FPS")
        # Настройка видеозаписи
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, clip.fps, clip.size)

        self.names = {}
        #window = dlib.image_window()
        shapes_df = pd.DataFrame(columns=['id', 'frame', 'shape', 'face_desc', 'name'])
        for frame_count in tqdm.tqdm(range(clip.n_frames)):
            frame = clip.get_frame(frame_count/clip.fps)
            #Берем данные, полученные pretracker
            detections = self.shapes_list[clipnum].loc[self.shapes_list[clipnum]['frame'] == frame_count]


            # Обновление трекера
            #tracked_boxes = self.tracker.update(detections, frame_count)
            tracked_boxes = detections if len(detections) > 0 else []
            # Обработка лиц
            if len(tracked_boxes) > 0:
                for ind,detection in tracked_boxes.iterrows():
                    shape = detection['shape']
                    x1 = shape[0]
                    y1 = shape[1]
                    x2 = shape[2]
                    y2 = shape[3]
                    obj_id = detection['id']
                    face = detection['face_shape']

                    if obj_id not in self.names:
                        #выбираем изображение и меняем BGR на RGB
                        human_img = np.array(frame[y1:y2, x1:x2], dtype=np.uint8)
                        if face is None:
                            name = 'Unknown'
                        else:
                            desc = self.face_recognition.facerec.compute_face_descriptor(human_img, face)
                            name = self.face_recognition.find_name_desc(desc)
                        #window.wait_for_keypress(' ')
                        if name != "Unknown":
                            self.names[obj_id] = name
                            logging.info(f"👤 Обнаружено лицо для объекта {obj_id}: {self.names[obj_id]}")


            #
            if len(detections) > 0:
                for ind,detection in tracked_boxes.iterrows():
                    shapes_df.loc[len(shapes_df)] = detection
        for ind,detection in shapes_df.iterrows():
            if len(self.names) >= detection['id']:
                name = self.names[detection['id']]
            else:
                name = str(detection['id'])
            shapes_df['name'].loc[ind] = name

        for frame_count in tqdm.tqdm(range(clip.n_frames)):
            # Отрисовка результатов
            frame = clip.get_frame(frame_count/clip.fps)[:,:,::-1]
            boxes = shapes_df[shapes_df['frame'] == frame_count]
            processed_frame = self._draw_results(frame, boxes, frame_count)
            # Запись кадра
            out.write(processed_frame)

        out.release()
        #Сохраняем таблицу с треками
        shapes_df = shapes_df.drop(columns=['face_desc'])
        shapes_df.to_csv(output_path[:-4] + '_shapes.csv', index=False)
        return

    def process_video(self, video_path: str, output_path: str):
        self.video_path = video_path
        self.output_path = output_path if output_path else ""
        if not os.path.exists(self.video_path):
            logger.error(f"🚨 Видео не найдено: {self.video_path}")
            return
        self.clip = VideoFileClip(self.video_path)
        # Параметры для нашего случая
        video_weight = 0.7  # больший вес для видео сцен
        audio_weight = 0.3  # меньший вес для аудио сцен
        min_duration = 10.0  # минимальная длительность сцены в секундах
        max_duration = 600.0 # максимальная длительность сцены в секундах
        merged_scenes = None
        video_scenes = self.detect_video_scenes(video_path)
        logger.info(f"Обнаружено {len(video_scenes)} сцен на основе видео контента:")
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
                                         video_weight=video_weight,
                                         audio_weight=audio_weight,
                                         min_scene_duration=min_duration,
                                         max_scene_duration=max_duration)
            logger.info(f"Обнаружено {len(merged_scenes)} сцен после улучшенного объединения:")
            for i, (start, end) in enumerate(merged_scenes):
                logger.info(f"Сцена {i+1}: {start:.2f} - {end:.2f} сек (длительность: {end-start:.2f} сек)")

        #Сохраняем шоты
        shorts_path = output_path + '/shorts'
        prefix = 'input_short_'
        #self.save_scenes_as_videos(video_path, self.clip.fps, video_scenes, shorts_path, prefix)
        '''results_list = self.model.track(video_path, stream=True, persist=True, tracker="bytetrack.yaml", verbose=False)
        for ind,results in enumerate(results_list):
            if len(results) > 0:
                id = results[0].boxes.id
            else:
                id = None
            print(ind,id)'''
        #Сохраняем сцены
        if merged_scenes:
            scene_path = output_path + '/scenes'
            prefix = 'input_scene_'
            #self.save_scenes_as_videos(video_path, self.clip.fps, merged_scenes, scene_path,prefix)

        #обработка шотов
        if not os.path.exists(output_path + f'/shaped_shorts'):
            os.makedirs(output_path + f'/shaped_shorts')
        for i, (start, end) in enumerate(video_scenes):
            short_clip = self.clip.subclipped(start + 2/self.clip.fps, end)
            df = self.video_short_pretracker(short_clip, i)
            self.shapes_list.append(df)
        #Удаляем из локального датасета записи, с именами, которые опознаны менее, чем в 10 кадрах
        for name in self.face_recognition.local_dataset['name'].unique():
            if len(self.face_recognition.local_dataset[self.face_recognition.local_dataset['name'] == name]) < 10:
                self.face_recognition.local_dataset = self.face_recognition.local_dataset[self.face_recognition.local_dataset['name'] != name]
        for i, (start, end) in enumerate(video_scenes):
            short_clip = self.clip.subclipped(start + 2/self.clip.fps, end)
            self.video_short_tracker(short_clip, output_path + f'/shaped_shorts/input_debug_{i}.mp4', i)
        self.clip.close()

    def save_scenes_as_videos(self, video_path, fps, final_scenes, output_dir='./scenes',prefix:str = ''):
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        for i, scene in enumerate(final_scenes):
            output_path = os.path.join(output_dir, f"{prefix}{i+1}.mp4")

            # Используем moviepy для вырезания сцены
            clip = VideoFileClip(video_path).subclipped(scene[0] + 2/fps, scene[1] - 1/3/fps)
            clip.write_videofile(output_path, codec='libx264', audio_codec='aac')
            clip.close()
            logger.info(f"Сохранена сцена {i+1} в {output_path}")

    def analyze_audio(self, frame_length=2048, hop_length=512):
        if self.clip.audio is None:
            logger.error("🚨 Видео не содержит аудио")
            return
        # Загрузка аудио
        y, sr = librosa.load(self.output_path + '/sound.wav')
        #fps = self.clip.audio.fps
        #y, sr = np.array(self.clip.audio.to_soundarray(fps = fps)*(2**16), dtype = "int16"), fps

        # Вычисление RMS энергии
        rms = librosa.feature.rms(y=y, frame_length=frame_length, hop_length=hop_length)[0]

        # Нормализация
        rms_normalized = (rms - np.min(rms)) / (np.max(rms) - np.min(rms))

        # Вычисление спектрального центроида
        spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr, n_fft=frame_length, hop_length=hop_length)[0]
        spectral_centroid_normalized = (spectral_centroid - np.min(spectral_centroid)) / (np.max(spectral_centroid) - np.min(spectral_centroid))

        # Вычисление zero crossing rate
        zcr = librosa.feature.zero_crossing_rate(y, frame_length=frame_length, hop_length=hop_length)[0]

        # Временная ось в секундах
        times = librosa.times_like(rms, sr=sr, hop_length=hop_length, n_fft=frame_length)

        return {
            'times': times,
            'rms': rms_normalized,
            'spectral_centroid': spectral_centroid_normalized,
            'zcr': zcr
        }

    def detect_audio_scenes(self, audio_features, rms_threshold=0.3, centroid_threshold=0.4, min_scene_duration=3.0):
        times = audio_features['times']
        rms = audio_features['rms']
        spectral_centroid = audio_features['spectral_centroid']

        scenes = []
        current_scene_start = 0.0

        for i in range(1, len(times)):
            # Проверяем изменения в RMS и спектральном центроиде
            rms_change = abs(rms[i] - rms[i-1])
            centroid_change = abs(spectral_centroid[i] - spectral_centroid[i-1])

            # Если изменения значительные, считаем это границей сцены
            if rms_change > rms_threshold or centroid_change > centroid_threshold:
                scene_duration = times[i] - current_scene_start
                if scene_duration >= min_scene_duration:
                    scenes.append((current_scene_start, times[i]))
                    current_scene_start = times[i]

        # Добавляем последнюю сцену
        if current_scene_start < times[-1]:
            scenes.append((current_scene_start, times[-1]))

        return scenes

    def detect_video_scenes(self,video_path, threshold=30.0):
        # Создаем менеджер сцен и детектор
        video = scenedetect.VideoManager([video_path])
        scene_manager = SceneManager()
        scene_manager.add_detector(ContentDetector(threshold=threshold))

        # Обрабатываем видео
        video.set_downscale_factor()
        video.start()
        scene_manager.detect_scenes(frame_source=video)

        # Получаем список сцен
        scene_list = scene_manager.get_scene_list()

        # Конвертируем в секунды
        scenes = []
        for scene in scene_list:
            start_time = scene[0].get_seconds()
            end_time = scene[1].get_seconds()
            scenes.append((start_time, end_time))

        return scenes

    def merge_scenes(self,video_scenes, audio_scenes,
                     video_weight=0.7, audio_weight=0.3,
                     min_scene_duration=3.0, max_scene_duration=120.0,
                     overlap_threshold=2.0):
        """
        Улучшенный алгоритм объединения сцен с учетом:
        - весовых коэффициентов для видео и аудио
        - минимальной и максимальной длительности сцен
        - сохранения логических границ
        """
        # Нормализуем и взвешиваем сцены
        weighted_scenes = []

        # Добавляем видео сцены с весом
        for start, end in video_scenes:
            duration = end - start
            if duration >= min_scene_duration:
                weighted_scenes.append({
                    'start': start,
                    'end': end,
                    'weight': video_weight,
                    'type': 'video'
                })

        # Добавляем аудио сцены с весом
        for start, end in audio_scenes:
            duration = end - start
            if duration >= min_scene_duration:
                weighted_scenes.append({
                    'start': start,
                    'end': end,
                    'weight': audio_weight,
                    'type': 'audio'
                })

        # Сортируем все сцены по времени начала
        weighted_scenes.sort(key=lambda x: x['start'])

        if not weighted_scenes:
            return []

        # Алгоритм интеллектуального объединения
        merged_scenes = []
        current_scene = weighted_scenes[0].copy()

        for scene in weighted_scenes[1:]:
            # Проверяем перекрытие или близость сцен
            scene_overlap = (scene['start'] <= current_scene['end'] + overlap_threshold)

            # Проверяем максимальную длительность
            duration_exceeded = (scene['end'] - current_scene['start']) > max_scene_duration

            # Если сцены пересекаются и не превышают максимальную длительность
            if scene_overlap and not duration_exceeded:
                # Объединяем сцены с учетом весов
                if scene['type'] == 'video' and current_scene['type'] != 'video':
                    current_scene['end'] = scene['end']
                    current_scene['weight'] += scene['weight']
                    current_scene['type'] = 'mixed'
                elif scene['weight'] > current_scene['weight']:
                    current_scene['end'] = scene['end']
                    current_scene['weight'] += scene['weight']
                    if scene['type'] != current_scene['type']:
                        current_scene['type'] = 'mixed'
                else:
                    current_scene['end'] = max(current_scene['end'], scene['end'])
                    current_scene['weight'] += scene['weight'] * 0.5  # меньший вес для расширения
            else:
                # Сохраняем текущую сцену и начинаем новую
                if current_scene['end'] - current_scene['start'] >= min_scene_duration:
                    merged_scenes.append((current_scene['start'], current_scene['end']))
                current_scene = scene.copy()

        # Добавляем последнюю сцену
        if current_scene['end'] - current_scene['start'] >= min_scene_duration:
            merged_scenes.append((current_scene['start'], video_scenes[-1][1])) #Для исключения ошибки работы с видео

        # Фильтруем слишком короткие сцены, которые могли появиться после объединения
        merged_scenes = [(start, end) for start, end in merged_scenes
                         if end - start >= min_scene_duration]

        return merged_scenes

    def _draw_results(self, frame: np.ndarray, tracked_boxes: pd.DataFrame,
                      frame_num: int) -> np.ndarray:
        """Отрисовывает результаты на кадре"""
        result_frame = frame.copy()

        # Отладочная информация
        if frame_num % 30 == 0:  # Каждые 30 кадров
            logger.debug(f"🎨 Отрисовка: {len(tracked_boxes)} объектов в кадре {frame_num}")

        # Отрисовка рамок с разноцветными ID
        for row in tracked_boxes.itertuples():
            _,obj_id,_,shape,_,_ = row
            x1, y1, x2, y2 = shape
            if obj_id == None:
                continue
            color = self._get_color_for_id(obj_id)

            # Рамка
            cv2.rectangle(result_frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 3)

            # Фон для текста
            #text = f"ID: {obj_id}"
            text = self.names[obj_id] if obj_id in self.names else f"ID: {obj_id}"
            (text_width, text_height), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
            cv2.rectangle(result_frame, (int(x1), int(y1)),
                          (int(x1)+text_width+10, int(y1)+text_height+10), color, -1)

            # Текст ID
            cv2.putText(result_frame, text, (int(x1)+5, int(y1)+20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            # Показываем уверенность
            #conf_text = f"{conf:.2f}"
            #cv2.putText(result_frame, conf_text, (int(x1), int(y2)-5),
            #            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        return result_frame

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

def main():
    """Главная функция"""
    processor = ComplexVideoProcessor()
    df = processor.process_video("C:/Users/above/IdeaProjects/video/Video_Samples/in2.mp4",
                                 "C:/Users/above/IdeaProjects/video/Video_SamplesC:/Users/above/IdeaProjects/video/Video_Samples/video/Video_Samples/out")
    print(df)

if __name__ == "__main__":
    main()