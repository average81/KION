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

from .base_tracker import BaseVideoProcessor

script_dir = os.path.dirname(os.path.abspath(__file__))

import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ComplexVideoProcessor(BaseVideoProcessor):
    def __init__(self, model_path: str = script_dir + "/yolov8n.pt", detector = 'mmod', force_update = False):
        # Инициализация базового класса
        super().__init__(model_path, detector, force_update)


    def video_short_pretracker(self, clip, clipnum):
        #Предварительный проход по видео для записи в локальную базу дескрипторов опознанных актеров именно из этого видео
        logger.info(f"🎬 Начинаем обработку шота {clipnum}: {clip.n_frames} кадров, {clip.fps:.1f} FPS")

        self.names = {}
        #window = dlib.image_window()
        shapes_df = pd.DataFrame(columns=['id', 'frame', 'shape', 'face_desc'])
        for frame_count in range(clip.n_frames):
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

    def process_video(self, video_path: str, output_path: str):
        self.video_path = video_path
        self.output_path = output_path if output_path else ""
        if not os.path.exists(self.video_path):
            logger.error(f"🚨 Видео не найдено: {self.video_path}")
            return
        self.clip = VideoFileClip(self.video_path)
        fps = self.clip.n_frames / self.clip.duration
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
        self.save_scenes_as_videos(video_path, fps, video_scenes, shorts_path, prefix)
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
            self.save_scenes_as_videos(video_path, fps, merged_scenes, scene_path,prefix)

        #обработка шотов
        if not os.path.exists(output_path + f'/shaped_shorts'):
            os.makedirs(output_path + f'/shaped_shorts')
        logger.info(f"Обработка шотов, 1 проход...")
        for i, (start, end) in enumerate(video_scenes):
            short_clip = self.clip.subclipped(start + 1/fps, end)
            df = self.video_short_pretracker(short_clip, i)
            self.shapes_list.append(df)
        #Удаляем из локального датасета записи, с именами, которые опознаны менее, чем в 10 кадрах
        for name in self.face_recognition.local_dataset['name'].unique():
            if len(self.face_recognition.local_dataset[self.face_recognition.local_dataset['name'] == name]) < 10:
                self.face_recognition.local_dataset = self.face_recognition.local_dataset[self.face_recognition.local_dataset['name'] != name]
        logger.info(f"Обработка шотов, 2 проход...")
        for i, (start, end) in enumerate(video_scenes):
            short_clip = self.clip.subclipped(start + 1/fps, end)
            self.video_short_tracker(short_clip, output_path + f'/shaped_shorts/input_debug_{i}.mp4', i)
        self.clip.close()

        #Сохраняем результаты анализа
        data = []
        prev_frame = 0

        for i, scene in enumerate(video_scenes):
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
        scene_analysis_path = output_path + '/shot_analysis.csv'
        df.to_csv(scene_analysis_path, index=False)
        logger.info(f"📊 Анализ шотов сохранен в {scene_analysis_path}")

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

def main():
    """Главная функция"""
    processor = ComplexVideoProcessor()
    processor.process_video("C:/Users/above/IdeaProjects/video/Video_Samples/in2.mp4",
                                 "C:/Users/above/IdeaProjects/video/Video_SamplesC:/Users/above/IdeaProjects/video/Video_Samples/video/Video_Samples/out")

if __name__ == "__main__":
    main()