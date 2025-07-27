import os
import cv2
import pandas as pd
import numpy as np
from typing import List, Tuple, Optional, Dict, Any
import tqdm

# Видео обработка
from moviepy import VideoFileClip


# Аудио обработка
import librosa
import librosa.display

# Детекция и трекинг объектов
from ultralytics import YOLO

from face_recognition import FaceRecognition

import re
from dataclasses import dataclass
from statistics import mean
import spacy
import subprocess

WEIGHTS = {
    'video': 0.5,               # вес для видео-анализа (ContentDetector)
    'audio': 0.3,               # вес для аудио-анализа
    'subtitles': 0.2,           # вес для анализа субтитров
    'min_duration': 10.0,        # минимальная длительность сцены (сек)
    'max_duration': 600.0,      # максимальная длительность сцены (сек)
    'rms_threshold': 0.3,       # порог для RMS энергии в аудио
    'centroid_threshold': 0.4,  # порог для спектрального центроида
    'sub_min_duration': 10.0,   # мин. длительность сцены по субтитрам
    'sub_time_gap': 3.0,        # макс. разрыв между репликами (сек)
    'sub_similarity': 0.55,     # порог семантической схожести текста
}


@dataclass
class SubtitleLine:
    index: int
    start: float  # в секундах
    end: float    # в секундах
    text: str
    doc: any = None  # spaCy Doc объект

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
        self.nlp = spacy.load("ru_core_news_md")


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
            if i == len(final_scenes) - 1:
                clip = VideoFileClip(video_path).subclipped(scene[0] + 1/fps)
            else:
                clip = VideoFileClip(video_path).subclipped(scene[0] + 1/fps, scene[1] - 1/3/fps)
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
                     subtitle_scenes,
                     min_scene_duration=3.0,
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
        current_scene = [0,0]
        num_video_scene = 0
        num_audio_scene = 0
        num_subtitle_scene = 0
        if len(audio_scenes) >0:
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
        else:
            merged_scenes = video_scenes
        if subtitle_scenes is not None:
            while num_video_scene < len(video_scenes) and num_subtitle_scene < len(subtitle_scenes):
                if subtitle_scenes[num_subtitle_scene][1] > video_scenes[num_video_scene][1]:
                    #конец видеосцены находится в аудиосцене - объединяем
                    current_scene[1] = video_scenes[num_video_scene][1]
                    num_video_scene += 1
                elif subtitle_scenes[num_subtitle_scene][1] > video_scenes[num_video_scene][1] - overlap_threshold:
                    #конец видеосцены выходит за границу аудиосцены на допустимое время - объединяем и финализируем
                    current_scene[1] = video_scenes[num_video_scene][1]
                    num_video_scene += 1
                    num_subtitle_scene += 1
                    if current_scene[1] - current_scene[0] > min_scene_duration:
                        merged_scenes.append([current_scene[0], current_scene[1]])
                        current_scene = [current_scene[1] + 0.01,current_scene[1] + 0.01]
                elif subtitle_scenes[num_subtitle_scene][1] >= video_scenes[num_video_scene][0] + overlap_threshold:
                    #Большая разница границ - объединяем
                    current_scene[1] = video_scenes[num_video_scene][1]
                    num_video_scene += 1
                    num_subtitle_scene += 1
                elif current_scene[1] - current_scene[0] < min_scene_duration:
                    #Слишком короткая сцена - объединяем
                    current_scene[1] = video_scenes[num_video_scene][1]
                    num_video_scene += 1
                    if subtitle_scenes[num_subtitle_scene][1] < video_scenes[num_video_scene][0]:
                        num_subtitle_scene += 1
                else:
                    #Финализируем текущую сцену
                    merged_scenes.append([current_scene[0], current_scene[1]])
                    num_subtitle_scene += 1
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

    def read_srt_file(self,file_path):
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                return file.read()
        except UnicodeDecodeError:
            with open(file_path, 'r', encoding='cp1251') as file:
                return file.read()

    def parse_srt(self,srt_text: str) -> List[SubtitleLine]:
        subtitles = []
        blocks = re.split(r'\n\s*\n', srt_text.strip())

        for block in tqdm.tqdm(blocks):
            lines = block.strip().split('\n')
            if len(lines) < 3:
                continue

            try:
                index = int(lines[0])
                time_match = re.match(r'(\d{2}):(\d{2}):(\d{2}),(\d{3}) --> (\d{2}):(\d{2}):(\d{2}),(\d{3})', lines[1])
                if not time_match:
                    continue

                h1, m1, s1, ms1 = map(int, time_match.groups()[:4])
                h2, m2, s2, ms2 = map(int, time_match.groups()[4:8])
                start = h1 * 3600 + m1 * 60 + s1 + ms1 / 1000
                end = h2 * 3600 + m2 * 60 + s2 + ms2 / 1000

                text = '\n'.join(lines[2:])
                doc = self.nlp(text)  # Анализируем текст с помощью spaCy
                subtitles.append(SubtitleLine(index, start, end, text, doc))
            except Exception as e:
                print(f"Ошибка при обработке блока: {e}")
                continue

        return subtitles

    def calculate_scene_metrics(self, scene: List[SubtitleLine]) -> Tuple[float, float]:
        """Вычисляет длительность сцены и среднюю схожесть реплик"""
        if not scene:
            return 0.0, 0.0

        durations = [sub.end - sub.start for sub in scene]
        similarities = []

        for i in range(1, len(scene)):
            sim = scene[i-1].doc.similarity(scene[i].doc)
            similarities.append(sim)

        total_duration = scene[-1].end - scene[0].start
        avg_similarity = mean(similarities) if similarities else 1.0

        return total_duration, avg_similarity



    def group_into_scenes(self, subtitles: List[SubtitleLine]) -> List[List[SubtitleLine]]:
        if not subtitles:
            return []

        # Сначала группируем по простым правилам
        initial_scenes = []
        current_scene = [subtitles[0]]

        for prev_sub, curr_sub in zip(subtitles, subtitles[1:]):
            time_gap = curr_sub.start - prev_sub.end
            similarity = prev_sub.doc.similarity(curr_sub.doc) if prev_sub.doc and curr_sub.doc else 0

            if time_gap > WEIGHTS['sub_time_gap'] or similarity < WEIGHTS['sub_similarity']:
                initial_scenes.append(current_scene)
                current_scene = [curr_sub]
            else:
                current_scene.append(curr_sub)

        initial_scenes.append(current_scene)

        # Затем объединяем короткие или связанные сцены
        merged_scenes = []

        for scene in initial_scenes:
            if not merged_scenes:
                merged_scenes.append(scene)
                continue

            if self.should_merge_scenes(merged_scenes[-1], scene):
                merged_scenes[-1].extend(scene)
            else:
                merged_scenes.append(scene)

        return merged_scenes

    def should_merge_scenes(self, prev_scene: List[SubtitleLine], current_scene: List[SubtitleLine]) -> bool:
        """Определяет, нужно ли объединять сцены"""
        if not prev_scene or not current_scene:
            return False

        # Проверяем временной промежуток между сценами
        time_gap = current_scene[0].start - prev_scene[-1].end

        # Проверяем схожесть последней реплики предыдущей сцены и первой текущей
        similarity = prev_scene[-1].doc.similarity(current_scene[0].doc)

        # Вычисляем общую длительность объединенной сцены
        merged_duration = current_scene[-1].end - prev_scene[0].start

        # Объединяем, если:
        # 1. Небольшой временной разрыв И хорошая схожесть
        # 2. ИЛИ если одна из сцен слишком короткая
        # 3. И при этом объединенная сцена не станет слишком длинной
        return ((time_gap <= WEIGHTS['sub_time_gap'] and similarity >= WEIGHTS['sub_similarity']) or
                any(self.calculate_scene_metrics(s)[0] < WEIGHTS['min_duration'] for s in [prev_scene, current_scene])) and \
            merged_duration <= WEIGHTS['max_duration']

    def prepare_subtitles_for_final(subtitles: List[SubtitleLine]) -> List[dict]:
        """Подготавливает субтитры для передачи в final_scene_segmentation"""
        return [{'start': sub.start, 'end': sub.end, 'text': sub.text} for sub in subtitles]

    def analyze_subtitles(self, subtitles_path = None, video_path = None):
        subtitle_scenes, subtitles = None, None

        if subtitles_path and os.path.exists(subtitles_path):
            logger.info(f"Используется внешний файл субтитров: {subtitles_path}")
            file = self.read_srt_file(subtitles_path)
            subtitles = self.parse_srt(file)
            subtitle_scenes = self.group_into_scenes(subtitles)
        else:
            embedded_srt = self.extract_embedded_subtitles(video_path)
            if embedded_srt:
                logger.info("Используются встроенные субтитры")
                file = self.read_srt_file(subtitles_path)
                subtitles = self.parse_srt(file)
                subtitle_scenes = self.group_into_scenes(subtitles)
            else:
                logger.info("Субтитры не найдены, анализ будет без учета субтитров")
        return subtitle_scenes

    def extract_embedded_subtitles(self, video_path, output_srt_path='embedded_subtitles.srt'):
        """Извлекает встроенные субтитры из видеофайла с помощью ffmpeg"""
        try:
            cmd = [
                'ffmpeg',
                '-i', video_path,
                '-map', '0:s:0',
                '-c:s', 'srt',
                output_srt_path
            ]
            subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

            if os.path.exists(output_srt_path) and os.path.getsize(output_srt_path) > 0:
                logger.info(f"Извлечены встроенные субтитры в {output_srt_path}")
                return output_srt_path
            else:
                logger.info("Не удалось извлечь встроенные субтитры")
        except subprocess.CalledProcessError as e:
            logger.info(f"Не удалось извлечь встроенные субтитры: {e}")
        except Exception as e:
            logger.info(f"Ошибка при извлечении субтитров: {e}")
        return None