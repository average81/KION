from face_recognition import FaceRecognition
from tracker import *
import logging
import argparse

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(description="парсер параметров")
    parser.add_argument('--video', type=str, required=True, help='входной файл видео')
    parser.add_argument('--output_dir', type=str, required=False, help='директория для сохранения результата работы')
    # Получаем аргумент из командной строки
    args = parser.parse_args()
    #Подготовка модулей
    logging.info('Start')
    face_recognition = FaceRecognition()
    processor = OptimizedVideoProcessor()
    processor.process_video(args.video, args.output_dir)
