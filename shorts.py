from face_recognition import FaceRecognition
from tracker import *
import logging
import argparse
import time
import os

if __name__ == '__main__':
    start_time = time.time()
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(description="парсер параметров")
    parser.add_argument('--video', type=str, required=True, help='входной файл видео')
    parser.add_argument('--output_dir', type=str, required=False, help='директория для сохранения результата работы')
    # Получаем аргумент из командной строки
    args = parser.parse_args()
    #Подготовка модулей
    logging.info('Start')
    #face_recognition = FaceRecognition()
    #Создание выходной папки
    if args.output_dir:
        if not os.path.exists(args.output_dir):
            os.makedirs(args.output_dir)
    #processor = OptimizedVideoProcessor()
    processor = ComplexVideoProcessor()
    #временно
    file = 'out.mp4'
    #processor.process_video(args.video, args.output_dir + "/" + file)
    processor.process_video(args.video, args.output_dir)
    end_time = time.time()
    logging.info(f'Total Time: {end_time - start_time}')
