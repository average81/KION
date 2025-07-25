import os
import cv2
import pandas as pd
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

# Детекция и трекинг объектов
from ultralytics import YOLO

from face_recognition import FaceRecognition


script_dir = os.path.dirname(os.path.abspath(__file__))

import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class BaseVideoProcessor:
    def __init__(self, model_path: str = script_dir + "/yolov8n.pt", detector = 'mmod', force_update = False):
        self.model = YOLO(model_path)
        self.color_cache = {}  # Кэш цветов для ID
        self.face_recognition = FaceRecognition(detector,recognition_value = 0.4)
        self.face_recognition.load_dataset(tomemory = True, force_update = force_update)
        self.shapes_list = []   #список датафреймов по шотам с треками людей

    def video_short_pretracker(self, clip, clipnum):
        #Пустышка
        return

    def video_short_tracker(self, clip, output_path: str, clipnum):
        """Обрабатывает видео с детекцией и трекомингом объектов"""

        logger.info(f"🎬 Начинаем обработку шота {clipnum}: {clip.n_frames} кадров, {clip.fps:.1f} FPS")
        # Настройка видеозаписи
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, clip.fps, clip.size)
        total_frames = clip.n_frames
        length = clip.duration
        fps = total_frames/length
        self.names = {}
        #window = dlib.image_window()
        shapes_df = pd.DataFrame(columns=['id', 'frame', 'shape', 'face_desc', 'name'])
        for frame_count in range(clip.n_frames):
            frame = clip.get_frame(frame_count/fps)
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
                    face = detection['face_desc']

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
            if detection['id'] in self.names:
                name = self.names[detection['id']]
            else:
                name = str(detection['id'])
            shapes_df['name'].loc[ind] = name

        for frame_count in range(clip.n_frames):
            # Отрисовка результатов
            frame = clip.get_frame(frame_count/fps)[:,:,::-1]
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
        #Пустышка
        return


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
        merged_scenes = []
        '''
        # Добавляем видео сцены с весом
        for start, end in video_scenes:
            duration = end - start
            if duration >= 0:#min_scene_duration:
                weighted_scenes.append({
                    'start': start,
                    'end': end,
                    'weight': video_weight,
                    'type': 'video'
                })

        # Добавляем аудио сцены с весом
        for start, end in audio_scenes:
            duration = end - start
            if duration >= 0:#min_scene_duration:
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
                    if scene['type'] != 'audio' and current_scene['type'] != 'audio':
                        current_scene['end'] = max(current_scene['end'], scene['end'])
                    elif scene['type'] != 'audio':
                        current_scene['end'] = scene['end']
                    elif current_scene['type'] != 'audio':
                        current_scene['end'] = current_scene['end'], scene['end']
                    current_scene['weight'] += scene['weight'] * 0.5  # меньший вес для расширения
            else:
                # Сохраняем текущую сцену и начинаем новую
                if current_scene['end'] - current_scene['start'] >= min_scene_duration:
                    merged_scenes.append((current_scene['start'], current_scene['end']))
                current_scene = scene.copy()
        '''
        current_scene = [0,0]
        num_video_scene = 0
        num_audio_scene = 0
        while num_video_scene < len(video_scenes) and num_audio_scene < len(audio_scenes):
            if audio_scenes[num_audio_scene][1] > video_scenes[num_video_scene][1]:
                #конец видеосцены находится в аудиосцене - объединяем
                current_scene[1] = video_scenes[num_video_scene][1]
                num_video_scene += 1
            elif audio_scenes[num_audio_scene][1] > video_scenes[num_video_scene][1] - overlap_threshold:
                #конец видеосцены выходит за границу аудиосцены на допустимое время - объединяем и финализируем
                current_scene[1] = video_scenes[num_video_scene][1]
                num_video_scene += 1
                num_audio_scene += 1
                if current_scene[1] - current_scene[0] > min_scene_duration:
                    merged_scenes.append([current_scene[0], current_scene[1]])
                    current_scene = [current_scene[1] + 0.01,current_scene[1] + 0.01]
            elif audio_scenes[num_audio_scene][1] >= video_scenes[num_video_scene][0] + overlap_threshold:
                #Большая разница границ - объединяем
                current_scene[1] = video_scenes[num_video_scene][1]
                num_video_scene += 1
                num_audio_scene += 1
            elif current_scene[1] - current_scene[0] < min_scene_duration:
                #Слишком короткая сцена - объединяем
                current_scene[1] = video_scenes[num_video_scene][1]
                num_video_scene += 1
                if audio_scenes[num_audio_scene][1] < video_scenes[num_video_scene][0]:
                    num_audio_scene += 1
            else:
                #Финализируем текущую сцену
                merged_scenes.append([current_scene[0], current_scene[1]])
                num_audio_scene += 1
                current_scene = [current_scene[1] + 0.01,current_scene[1] + 0.01]
        merged_scenes.append([current_scene[0], current_scene[1]])
        #Последняя сцена
        if current_scene[1] < video_scenes[-1][1]:
            if video_scenes[-1][1] - current_scene[1] > min_scene_duration:
                merged_scenes.append([current_scene[0], video_scenes[-1][1]])
            else:
                merged_scenes[-1][1] = video_scenes[-1][1]





        ''' 
        # Добавляем последнюю сцену
        if current_scene['end'] - current_scene['start'] >= min_scene_duration:
            merged_scenes.append((current_scene['start'], video_scenes[-1][1])) #Для исключения ошибки работы с видео

        # Фильтруем слишком короткие сцены, которые могли появиться после объединения
        merged_scenes = [(start, end) for start, end in merged_scenes
                         if end - start >= min_scene_duration]
        '''
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
